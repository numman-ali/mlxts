# Inference Optimization Techniques

Catalog of optimization techniques studied from reference inference servers (Rapid-MLX, vLLM-MLX, oMLX) and the research papers behind them. This document informs Phase 9 architecture decisions and records where each technique lives in the reference codebases.

## Philosophy

The reference projects share an approach: study research papers, implement what matters, iterate. Rapid-MLX implements MTP from one paper, speculative decoding from another, SpecPrefill from a third. oMLX implements tiered caching from vLLM's design, adds SSD persistence, then builds mixed-precision quantization on top. They don't commit to a single technique — they build interfaces flexible enough to swap strategies.

**Our approach must match this.** Every optimization boundary in the mlxts stack should be strategy-agnostic:

- **Cache backend**: simple → paged → SSD-persistent, swappable without changing model code
- **Decode strategy**: greedy → speculative → MTP, swappable without changing the serving layer
- **Prefill strategy**: sequential → chunked → sparse (SpecPrefill), swappable via configuration
- **Cache storage**: full precision → quantized → TurboQuant, transparent to the generation loop
- **Scheduling**: single-request → FCFS batched → priority-aware, behind a stable interface

The interfaces must be stable. The implementations behind them evolve as we adopt new techniques.

## Reference Repos

| Repo | Focus | Key files |
|------|-------|-----------|
| `.reference/rapid-mlx` | Inference speed | `vllm_mlx/engine/`, `vllm_mlx/speculative/`, `vllm_mlx/tool_parsers/`, `vllm_mlx/prefix_cache.py`, `vllm_mlx/memory_cache.py` |
| `.reference/vllm-mlx` | Foundational infrastructure | `vllm_mlx/paged_cache.py`, `vllm_mlx/scheduler.py`, `vllm_mlx/prefix_cache.py` |
| `.reference/omlx` | Production serving | `omlx/cache/`, `omlx/engine_pool.py`, `omlx/process_memory_enforcer.py`, `omlx/oq.py` |

All three share lineage — vLLM-MLX is the origin, Rapid-MLX and oMLX forked and diverged into speed and production robustness respectively.

---

## Decode Acceleration

### Multi-Token Prediction (MTP)

**What**: Forward pass returns hidden states alongside logits. MTP head drafts a second token from the hidden state. Both tokens are verified in a single forward pass. If the draft is accepted, the cache advances by 2 tokens per step.

**Speedup**: 1.4x decode (measured on Qwen3-Next).

**Requirements**: Model must expose `mtp_forward()` and `return_hidden=True` in forward. The generation loop needs accept/reject logic with cache trim on rejection. For hybrid RNN+attention models, rejection requires RNN state snapshot restore (see DeltaNet below).

**Paper**: [Multi-Token Prediction](https://arxiv.org/abs/2404.19737)

**Reference**: Rapid-MLX `scheduler.py:541` (`_install_mtp`). Implemented as a monkey-patch on BatchGenerator; in mlxts this should be a first-class strategy.

### Speculative Decoding (Draft Model)

**What**: A small draft model generates N candidate tokens cheaply. The target model verifies all N in a single forward pass (parallel verification). Accepted tokens advance the cache; rejected tokens are discarded with cache trim.

**Speedup**: 1.5-2.3x decode depending on draft quality and acceptance rate.

**Requirements**: Draft and target model share a prompt cache (split by layer count). O(1) cache trim is critical — the acceptance check is greedy left-to-right, and the first mismatch terminates.

**Paper**: [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2302.01318)

**Reference**: Rapid-MLX `models/llm.py:604` delegates to mlx-lm. Draft model loaded at init, both share a single `prompt_cache` list split `[:main_len]` / `[main_len:]`.

### Speculative Decoding (Prompt Lookup)

**What**: Draft-model-free approach. Build an n-gram index over the prompt and generated tokens. Query the last N tokens against the index, propose the longest continuation from a prior occurrence. Verify in one forward pass.

**Speedup**: Modest (benchmarked as minimal benefit on many models). Works best on repetitive content.

**Requirements**: N-gram index (plain Map), O(1) cache trim.

**Reference**: Rapid-MLX `speculative/prompt_lookup.py`. 3-gram default, up to `num_draft_tokens` candidates.

### EAGLE-3

**What**: Feature-level draft generation. A lightweight draft head operates on hidden-state features rather than generating full tokens. Much higher acceptance rates than standard speculative decoding.

**Speedup**: 3-6.5x decode (paper claims).

**Status**: Not implemented in any reference repo. On Rapid-MLX's roadmap.

**Paper**: [EAGLE-3](https://arxiv.org/abs/2503.01840)

### ReDrafter

**What**: Apple's RNN-based draft head. Trained per model.

**Speedup**: 1.4-1.5x.

**Status**: Not implemented in reference repos. Apple has MLX reference code.

**Paper**: [ReDrafter](https://arxiv.org/abs/2403.09919)

---

## Prefill Optimization

### Chunked Prefill with Decode Interleaving

**What**: Break large prefills into chunks (512-8192+ tokens depending on operator mode). Between chunks, run one decode step for all active requests. Prevents decode starvation during long-context prefills. If a request has a cached prefix, align the first chunk to that boundary for optimal cache capture (critical for hybrid RNN models where the snapshot must be taken at the exact prefix boundary). In `@mlxts/serve`, `--prefill-step-size` controls cold prompt-prefill chunks, while `--active-prefill-step-size` controls chunks only when another row is already decoding.

**Reference**: All three servers implement this. Rapid-MLX `scheduler.py:132` (`_install_chunked_prefill`).

### SpecPrefill (Sparse Prefill)

**What**: A small draft model scores token importance via attention capture. Only top-K% of tokens are prefilled by the target model, with manual RoPE at original positions. Reduces TTFT on long prompts.

**Paper**: [SpecPrefill](https://arxiv.org/abs/2502.02789)

**Reference**: oMLX `patches/specprefill.py`. Architecture-specific query extractors for Qwen3.5, Llama/Mistral/Gemma, and Nemotron-H.

---

## KV Cache Management

### Paged KV Cache

**What**: Instead of one contiguous cache per sequence (O(n) concatenation on update), use fixed-size blocks (64 tokens) with reference counting. Doubly-linked free list gives O(1) alloc/free. Copy-on-Write when a shared block diverges — allocate a new block, copy metadata, decrement source ref count. MLX arrays are immutable, so sharing cache data references is safe.

**Reference**: vLLM-MLX `paged_cache.py`. `CacheBlock` (line 84), `FreeKVCacheBlockQueue` (line 158), CoW (line 1029).

### Chain Hashing for Prefix Dedup

**What**: Each block's SHA-256 hash includes its parent block's hash, plus its own token IDs and model name. Identical prefixes always produce identical hash chains, enabling automatic dedup without explicit trie traversal. O(1) lookup per block.

**Reference**: vLLM-MLX `prefix_cache.py:40` (`compute_block_hash`).

### Prompt Cache with LCP Matching

**What**: Memory-aware prefix cache using a sorted key index with binary search for O(log N) prefix lookup. Four match types: exact, shorter prefix, longer prefix (trim excess), and LCP (Longest Common Prefix — divergent sequences sharing a common head). The LCP pattern is critical for agentic multi-turn chat where system prompts are identical but conversation history diverges.

**Reference**: Rapid-MLX `memory_cache.py:409` (`MemoryAwarePrefixCache`). Memory-based eviction (default 20% of available RAM). Supports KV quantization (8-bit) for stored entries.

### DeltaNet State Snapshots

**What**: Qwen3.5 uses a hybrid architecture: 75% Gated DeltaNet layers (non-trimmable RNN) + 25% attention layers (trimmable KV). You can trim KV cache to any prefix, but RNN state is cumulative — you can't "undo" tokens. Solution: deep-copy non-trimmable layers at prefix boundaries (~0.1ms, RNN state is small), restore on multi-turn reuse + trim KV layers to snapshot length.

**Design requirement for mlxts**: The KV cache interface must have an `isTrimmable()` discriminator from the start. Tag cache layers as trimmable or non-trimmable. Snapshot non-trimmable layers at prefix boundaries. This costs nothing to add now and enables hybrid architecture support later.

**Reference**: Rapid-MLX `models/llm.py:261` (`_snapshot_rnn_layers`, `_restore_rnn_layers`).

### SSD-Persistent KV Cache

**What**: Serialize KV blocks to safetensors on SSD. One file per block, hex-prefix sharded (16 subdirectories). Background writer thread uses a pure-Python safetensors writer (no MLX API calls, thread-safe). Write-back RAM hot cache: `save_block()` stores in RAM only; SSD write happens lazily when evicted from RAM. Reads promote SSD blocks back to RAM. Survives server restarts via startup scan of SSD directories.

**Performance**: ~2ms read per 10MB block on NVMe (5 GB/s). Negligible compared to model computation.

**TypeScript implementation notes**: Need a pure-JS safetensors writer for the background worker thread. The bfloat16-as-uint16 trick (view raw bytes since JS buffer protocol doesn't support bfloat16) is essential. Use Bun worker threads for non-blocking SSD writes. Chain hashing is straightforward via Bun's built-in crypto.

**Reference**: oMLX `cache/paged_ssd_cache.py`. Background writer at line 841, tensor extraction at line 131, startup recovery at line 756.

### KV Cache Quantization

**What**: Quantize stored KV entries to 4/8-bit at the prefix cache storage layer. Dequantize on fetch. 8-bit gives >2x memory reduction with <0.05 mean absolute error. Non-KV layers (Mamba state, dict-based state) pass through unmodified.

**Advanced variant (TurboQuant)**: Run attention directly on quantized KV states via custom Metal kernel, without dequantizing during decode (L=1). Uses learned rotation-based codecs. Codec reconstruction is deterministic from (head_dim, bits, seed), enabling cache serialization.

**Reference**: Rapid-MLX `memory_cache.py:368` (`_quantize_cache`). oMLX `turboquant_kv.py`.

---

## Serving Infrastructure

### Dual-Strategy Engine (HybridEngine)

**What**: Single loaded model shared between a serial engine (speculative decoding, max single-user throughput) and a batched engine (continuous batching, concurrent users). Switch on active request count: >= threshold → batched, idle → serial. Model ownership registry prevents concurrent Metal operations.

**Reference**: Rapid-MLX `engine/hybrid.py:47`.

### Continuous Batching

**What**: Scheduler manages request lifecycle (WAITING → RUNNING → FINISHED). Delegates actual GPU batching to a BatchGenerator. FCFS scheduling by default. Configurable max concurrent sequences. Requests enter/leave the batch dynamically without restarting the batch.

**Reference**: vLLM-MLX `scheduler.py:956`.

### Multi-Model Memory Management

**What**: Engine pool manages multiple loaded models. Bookkeeping-based estimates (safetensor file sizes + 25% KV headroom) for pre-load decisions. Real Metal memory polling (`mx.get_active_memory()`) every 1s as safety net. LRU eviction with model pinning and per-model TTL. Cooperative abort via flag checked between loading phases.

**Critical insight**: Do NOT use `mx.set_memory_limit()` — it causes alloc/free churn during model loading. Use bookkeeping + polling instead.

**Reference**: oMLX `engine_pool.py`, `process_memory_enforcer.py`.

### Tool Calling and Structured Output

**Parser registry**: Pluggable parsers registered by model family. Auto-detection via `(regex, config)` first-match list. 17 formats in Rapid-MLX covering Hermes, Llama, DeepSeek, Qwen, GLM, MiniMax, Kimi, etc. Auto-recovery for quantized model degradation.

**Jump-forward decoding**: Logits processor that biases token probabilities toward predictable structural markup during tool call generation. Pre-tokenizes structural patterns, applies `bias_strength` (default 20.0) to expected next tokens. State machine tracks position within patterns. Safety escape after 50 consecutive biased tokens. 2-5x faster structured output.

**Reference**: Rapid-MLX `tool_parsers/`, `api/tool_logits.py`.

### Model Auto-Configuration

**What**: Ordered list of `(regex, ModelConfig)` tuples. `detect_model_config()` does first-match against model path string. More specific patterns go first. Zero user flags — `serve <model>` auto-applies optimal parser, reasoning mode, prefill size.

**Reference**: Rapid-MLX `model_auto_config.py`.

### Pre-Computed SSE Streaming

**What**: Template-compile the JSON SSE envelope at request start (response ID, model name, object type). Only escape and substitute the dynamic `content` per token. 20-30% reduction in server CPU overhead for streaming.

**Reference**: Rapid-MLX `api/streaming.py`.

---

## Quantization

### oQ Mixed-Precision Quantization

**What**: Budget-planned per-layer quantization. Measure layer sensitivity via quantize-dequantize MSE on calibration data (128 samples, 256 seq len). Greedy allocate extra bits to sensitive layers within a bpw budget. MoE routed experts stay at base bits. Mandatory boosts for lm_head/embeddings (8-bit). Protection floor for v_proj, down_proj, o_proj. Streaming quantizer processes one safetensor shard at a time (~3-4GB peak regardless of model size).

**Cheaper proxy path**: Load an already-quantized model, re-quantize each layer at (bits-1), measure delta. Achieves ~90% top-10 overlap with fp16 measurement at 4x less memory.

**Reference**: oMLX `oq.py`. Budget planner at line 427.

### IndexCache (DeepSeek Sparse Attention)

**What**: Adjacent attention layers in DeepSeek-class models share 70-100% of selected tokens in their sparse indexers. "Shared" layers reuse indices from the preceding "Full" layer, skipping Q*K attention + argpartition while keeping the indexer KV cache updated. Reduces redundant computation in MoE models.

**Paper**: [IndexCache](https://arxiv.org/abs/2603.12201)

**Reference**: oMLX `patches/index_cache.py`.

---

## Multimodal Composition

### Vision as Preprocessing

**What**: After the initial vision-aware forward pass (image → embeddings → inject into token stream), text generation is identical to LLM generation on `model.language_model`. The model never sees images during autoregressive decode — only the embedded representations.

**Implication for mlxts**: Vision encoding is a composable prefix step, not a separate model contract. CausalLM stays unchanged. `@mlxts/transformers` holds vision encoders and VLM wrappers alongside text decoders.

**Reference**: vLLM-MLX `mllm_batch_generator.py:12-17`.

### Two-Level Vision Embedding Cache

**What**: L1 caches `prepare_inputs()` output (saves image loading/resizing, ~0.5-1s). L2 caches VLM forward pass output (saves vision encoder computation, ~1-2s). Prompt-independent pixel-only cache allows reusing image tensors across different prompts for the same image. 100x speedup for repeated images in multi-turn chat.

**Reference**: vLLM-MLX `vision_embedding_cache.py:128`.

### Audio and Embeddings Are Not Autoregressive

**What**: STT, TTS, and embedding models don't share the BaseEngine interface. They're separate model types co-located on the same HTTP server as independent singletons. No attempt to unify them under an autoregressive generation contract.

**Implication for mlxts**: Confirms our architecture decision. `@mlxts/transformers` holds autoregressive architectures only. Audio and embedding serving would be separate engine types in `@mlxts/serve`, not extensions of the generation pipeline.

---

## Approach for mlxts

These techniques are not all-or-nothing. The progression is:

1. **Current serving baseline**: OpenAI-compatible completions/chat, narrow text Responses, bounded Anthropic Messages, admission limits, cancellation, streaming, multi-model loading, endpoint benchmarks, and cache-generic continuous batching for eligible LLaMA-like, Qwen 3.6 text, and Gemma 3/4 layer-pattern requests. Continuous routes now share a model-level scheduled-token reservation budget derived from configured total-token and batch-size limits.
2. **Scheduler hardening**: separate prefill/completion budgets, richer fairness controls, and higher-concurrency evidence for sampled/model-native defaults.
3. **Cache pass 1**: prefix cache with LCP matching and explicit Qwen non-trimmable-state handling; rotating/max-KV policy for long contexts.
4. **Cache pass 2**: paged KV, quantized KV, model-aware memory policy, and production metrics for cache hits, tokens saved, evictions, and memory pressure.
5. **Production serving pass**: model pool/eviction, fuller Responses and Anthropic content/tool support, structured output/logprobs, embeddings, and multimodal serving over the same normalized request model.
6. **Advanced speed pass**: whole-batch sampling, MTP, speculative/prompt-lookup decode, jump-forward decoding, SpecPrefill, TurboQuant, and custom native or Metal seams only after evidence proves the strategy.

Each pass adds capability behind stable interfaces. The serving API, model code, and user-facing CLI don't change between passes.
