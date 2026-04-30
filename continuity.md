# Continuity

Compact handoff for long-running `mlxts` work. Durable doctrine lives in
`AGENTS.md`, durable learnings in `MEMORY.md`, and current-phase state that
should survive context compaction lives here.

## Current Focus

Qwen 3.6 27B serving/inference quality is the active critical path. The goal is
staged parity evidence against `mlx-lm`, long-output stability, and long-context
capability — not a single short benchmark. The OpenResponses and Anthropic
Messages adapters are part of the same serving-quality push: usable text,
image, and bounded tool endpoints while benchmark and scheduler work continues.

## Current State

- **Qwen 3.6 27B**: native gated-delta helper at `fast.qwenGatedDeltaUpdate`;
  mixed-dtype linear attention with fp32 recurrent state and contiguous conv
  cache tails; full-attention prefill on causal SDPA marker; quantized `b/a`
  gate projections fused with source-handle invalidation; nested
  `text_config.max_position_embeddings = 262144` advertised.
- **Continuous batching (full-KV greedy safe)**: `ContinuousBatchTokenScheduler`
  owns waiting/active rows, chunk-prefills behind active decode, and is
  endpoint-visible through `serveLoadedModel()`. Async lookahead keeps the
  next-token tensor on-device when active rows are unchanged.
- **Streaming**: SSE flush-cadence fix shipped (macrotask yield between
  processed events). Bun lifecycle handling disables idle timeout for streaming
  responses; client disconnects surface as cancellation; admission gates
  release on early stream failure.
- **Serving telemetry**: `bench:serve` reports `continuous_admissions`
  separately from `static_batches` and `admission_batches`; server prefill
  metrics (`mean_server_prefill_ms`, `mean_server_prefill_tps`) are distinct
  from client-observed TTFT. Reports now also expose protocol usage cache
  tokens plus server-event prompt-prefix cache hits, writes, and read/write
  tokens.
- **Prompt-prefix cache retention**: serve keeps the default at one retained
  prompt-boundary snapshot per served model, but `promptPrefixCacheMaxEntries`
  / `--prompt-prefix-cache-max-entries` and `promptPrefixCacheMaxBytes` /
  `--prompt-prefix-cache-max-bytes` now bound retained snapshots by count and
  live estimated tensor bytes without changing family-owned snapshot/fork cache
  semantics. Full-KV, trimmable retained snapshots share complete 64-token
  tensor blocks across prompt-prefix descendants; active decode storage, Qwen
  hybrid caches, and Gemma sliding/layer-pattern semantics are unchanged.
- **Continuous memory admission**: continuous serving uses the existing
  model-level reservation controller for prompt, completion, aggregate total,
  and estimated memory pressure. Memory estimation remains serve-owned and
  config-derived; scheduler events, metrics, CLI logs, and benchmark reports
  expose scheduled memory pressure when the guard is active.
- **Image serving**: Qwen image transport, host decode, and prepared-prompt
  cache shipped with explicit boundary — serve owns I/O and decode, transformers
  owns preprocessing and prompt expansion. OpenAI Chat/OpenResponses accept
  data-url images; Anthropic Messages accepts local base64 user image blocks
  through the same content route. Remote/file image sources remain rejected
  until transport policy exists.
- **Qwen conditional serving**: top-level Qwen 3.5 / 3.6 conditional
  checkpoints expose the Qwen text batch-cache surface for text-only continuous
  serving. Media/content requests still route as `media_input` and stay off
  continuous scheduling until multimodal batch semantics are implemented
  explicitly.
- **Architectural cleanup (2026-04-28)**: the audit remediation tranches have
  landed end to end. Cross-example data helpers moved into `@mlxts/data`;
  `serve/src/`, Qwen 3.5/3.6, transformer LoRA, align evaluation, tokenizers,
  and nn are in role-named folders; core compile controls are out of the
  top-level barrel; stream writers, supervised-run primitives, trainable-module
  helpers, progress reporting, and cache layer taxonomy are package-owned; and
  package-agent / cross-package-import governance gates are wired into
  `bun run validate`.

## Latest Evidence

Full evidence ladder lives in
`docs/reviews/2026-04-24-qwen-serve-benchmark-ladder.md`. Headline numbers:

- Paired Qwen vs `mlx-lm`: `1024/128` `28.999 / 28.899 tok/s`,
  `10000/128` `26.959 / 27.154`, `1024/1024` `28.352 / 28.448`.
- Qwen `131072/128` streamed end-to-end on 64 GB local
  (`peak=42.55 GB`, `decode_tps=16.019`, exact marker match). Long-context
  capability demonstrated; product-grade TTFT not.
- Qwen long-output streamed at `1024/10000` (`26.850 tok/s`, `19.934 GB peak`)
  and `1024/20000` (`26.597 tok/s`, `19.934 GB peak`).
- Gemma 4 E2B endpoint healthy through `32k` prompt tokens
  (`19.524 tok/s`, `11.192 GB peak`) and through `10000` output tokens
  at flat active memory.
- Mixed long-prefill / short-arrival fairness: Qwen `32768x128 + 128x32`
  short-queue dropped from ~126s starvation to ~2.6s after fairness-biased
  prefill chunking.
- Real repeated-turn protocol health: `regression:qwen-gemma -- --profile real`
  now requires prompt-prefix cache hits for Qwen and Gemma chat, OpenResponses,
  and Anthropic Messages rungs. Latest proof passed with Qwen read evidence
  `139` client cached tokens for chat/responses and `278` server prompt-cache
  read tokens across all three message protocols; Gemma recorded `138` client
  cached tokens for chat/responses and `276` server prompt-cache read tokens.
- Qwen3.6 top-level conditional route smoke on 2026-04-29: direct Chat
  Completions against `mlx-community/Qwen3.6-27B-4bit` logged
  `route=continuous eligible=yes reason=eligible model_type=qwen3_5`; the
  warmed repeat logged `cache hit read_tokens=16 write_tokens=0` and returned
  `prompt_tokens_details.cached_tokens=16`.
- Post-fix `bun run regression:qwen-gemma -- --profile real` passed. Qwen
  decode smoke reported `generation_tps=29.027`; Qwen serve rungs routed
  `continuous:eligible` through protocol health and mixed fairness, including
  `max_continuous_batch=8` and Qwen mixed `32768x128+128x32` passing with
  `max_stream_chunk_gap_ms=658.4`. Gemma real decode and serve rungs passed too.
- Anthropic image-message tranche passed focused serving tests, all
  `packages/serve` tests, `bun run validate`, and the real Qwen/Gemma
  regression. The real protocol rungs kept Anthropic Messages
  `continuous:eligible` for text requests (`30.972` Qwen post-TTFT tok/s,
  `82.759` Gemma post-TTFT tok/s); media-shaped Anthropic requests route
  through the existing single-request content path.
- Prompt-prefix cache retention knob passed focused serve tests, all
  `packages/serve` tests, `bun run validate`, and
  `bun run regression:qwen-gemma -- --profile quick`. Default retention remains
  one snapshot, so existing real-regression prompt-cache requirements stay
  unchanged.
- Prompt-prefix cache byte-budget retention passed focused serve/transformer
  cache tests (`113 pass`), `bun run typecheck`, and
  `bun run regression:qwen-gemma -- --profile quick` (`84` transformer focused
  tests and `205` serve focused tests). `estimatedByteSize` is now exposed on
  cache snapshots, and serve can dispose over-budget prompt-boundary snapshots
  instead of retaining them.
- Full-KV prompt-prefix tensor-block retention passed focused cache/serve tests
  (`42 pass`), focused package tests (`350 pass`), `bun run typecheck`,
  `bun run check:coverage`, `bun run validate`, and
  `bun run regression:qwen-gemma -- --profile quick` (`84` transformer focused
  tests and `205` serve focused tests). Retained full-KV descendants charge
  only private tail blocks while their source snapshot is live; batch cache
  restore/filter/extend/extract keeps source lineage for continuous serving.
- Qwen image serving product regression passed against cached
  `mlx-community/Qwen3.6-27B-4bit` through OpenAI Chat image content,
  OpenResponses image input, and Anthropic Messages base64 image blocks. The
  harness command is `bun run regression:qwen-image`; the latest run described
  a generated 2x2 red/green/blue/yellow grid, routed every request as
  `single:media_input`, kept continuous scheduler phases at `0`, and read `92`
  cached prompt tokens on exact repeats.
- Qwen image direct example proof passed against cached
  `mlx-community/Qwen3.6-27B-4bit` using
  `examples/qwen3_5-image/index.ts --json --greedy --max-tokens 64`. The
  example now defaults to cached/local-only source resolution and disabled
  thinking for short visual descriptions; the proof resized a generated `96x96`
  quadrant BMP to `256x256`, finished with `eos`, generated `34` tokens, and
  named all four quadrants correctly.
- Anthropic Messages tool-use adapter support passed focused protocol and HTTP
  route tests (`51 pass`) and full `bun run validate`. `/v1/messages` now
  accepts client tools, assistant `tool_use` history, and user `tool_result`
  history, formats generated tool calls as Anthropic `tool_use` blocks with
  `stop_reason="tool_use"`, and rejects incomplete tool-result transcripts,
  streaming tool use, tool-result errors, and rich tool-result media until those
  semantics are implemented explicitly.
- OpenResponses function-tool adapter support passed focused protocol and HTTP
  route tests (`56 pass`) and full `bun run validate`. `/v1/responses` now
  accepts flat function tools, normalizes `function_call` /
  `function_call_output` history plus adjacent reasoning items into internal
  assistant/tool turns, formats generated tool-call envelopes as
  `function_call` output items, and rejects streaming tools,
  `parallel_tool_calls=false` with active tools, built-in/custom tools, and
  rich function outputs until those semantics are implemented explicitly.
- Gemma 4 A4B MoE proof passed against the cached
  `mlx-community/gemma-4-26b-a4b-it-4bit` snapshot. Transformer decode at
  `128x128` reported `generation_tps=108.604`, `evals_per_token=1.00`, and
  flat active memory (`14.527 GB` start/end). Serve streamed `128x32@1` through
  `continuous:eligible` with `32` chunks, `active_delta=0.000 GB`, and
  `mean_post_ttft_completion_tps=95.470`. The chat example loaded the same
  15.6 GB snapshot and produced a coherent greedy answer.
- Qwen A3B split-quantized MoE proof passed against the cached
  `unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit` snapshot. Direct `128x128` decode
  reported `generation_tps=89.954`, `evals_per_token=1.00`, and flat active
  memory (`20.816 GB` start/end). Serve streamed `128x32@1` through
  `continuous:eligible` with `mean_post_ttft_completion_tps=79.300` and
  `active_delta=0.004 GB`.

## Next Work

- Remaining Qwen gap is mostly peak memory versus `mlx-lm` plus small
  paired-run variance. Profile full-attention KV representation, cache-buffer
  accounting, and wrapper/FFI overhead before scattering micro-optimizations.
- Next scheduler tranche is cache-semantics work: chunked prefill fairness,
  streaming collectors, sampled batch decode, then Qwen hybrid recurrent /
  full-attention caches and Gemma sliding/global caches.
- Prefix-cache cache-hit seeding now reaches continuous scheduler rows through
  one-row managed batch-cache restore. Hits classify exact, shorter-prefix,
  longer-source trim, and LCP reuse and expose source `CacheLayerKind` /
  trimmability metadata. Serve retains prompt-token block chains with
  ref-counted deduplication and uses a block-hash index to narrow lookup
  candidates before the existing LCP and family-owned `canFork()` gate. The
  served retention limits are explicit runtime/CLI knobs by entry count and
  estimated retained snapshot bytes. Paged attention and cache-tensor block
  deduplication remain later cache-backend work.
- Next memory work is multi-model pool management: model-load estimates, idle
  eviction, pinned models, and active-request abort policy when one loaded model
  needs to shed KV pressure.
- Use `bun run bench:serve --stream` for huge prompt rungs; buffered JSON is
  a poor acceptance shape when client TTFT exceeds a few minutes.
- For publishable parity claims, use
  `bench:generation:parity --require-mlx-lm-reference`. Otherwise a missing
  reference is a local-only profiling run.
- For broad context-window claims, use
  `bench:generation:context --needle-placements all` and require early /
  middle / late marker evidence.
- `--ignore-eos` is an exact-length throughput tool only. Normal serving
  honors EOS; parity claims need explicit ignore-EOS rungs to compare decode
  speed rather than chat stopping behavior.
- The 2026-04-28 architectural cleanup backlog is complete. Future structure
  changes should keep `bun run check:per-package-agents` and
  `bun run check:cross-package-imports` green instead of relying on manual
  package-boundary review.
- Phase 7f has Gemma A4B and Qwen A3B real checkpoint proofs. Choose the next
  model-family proof deliberately, with Mixtral only when there is a clear
  user-facing need. Otherwise the next high-value product areas are Phase 9
  cache backend work, Qwen image product proof, and Phase 8 training proof
  hardening.
