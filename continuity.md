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

The roadmap is now aligned through Phase 10 as the active product horizon:
finish Phase 7 architecture truth, harden Phase 8 fine-tuning/alignment proofs,
complete Phase 9 serving/quantized inference, run Phase 9.5 AXI hardening for
agent-operated CLIs, then complete Phase 10 multimodal understanding and
diffusion/flow generation with package-owned proofs. Phases 8, 9, and 10 fan
out from Phase 7 rather than forming a strict serial chain.

`@mlxts/agent` is an experimental loop harness, not the main product-agent
surface. Keep its finite CLI paths AXI-shaped for testing and reuse, but keep
major product-agent focus on package-owned CLIs and future PI-agent integration.

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
- **Model-load memory preflight**: source-backed `serveModel()` and
  `serveModels()` now estimate local `.safetensors` bytes, including Hugging
  Face snapshot symlinks, with 25% headroom after source resolution and before
  MLX model loading. The check rejects clearly over-budget loads against active
  MLX memory and `gpuMemoryUtilization` when telemetry and sizing are present;
  missing telemetry or sizing skips the preflight rather than making an unsafe
  claim.
- **Lazy source-backed model pool**: `serveModels({ modelLoadPolicy: "lazy" })`
  now publishes configured model ids at startup, loads checkpoints on first
  request, shares concurrent first loads per model, serializes cold loads across
  model ids so memory preflight stays honest, evicts idle non-pinned models
  after `modelIdleTtlMs`, and keeps pinned models resident until shutdown.
  `modelPressurePolicy` defaults to `reject`; `shed_non_pinned` can evict idle
  non-pinned models and abort the oldest eligible active non-pinned request
  scope with `model_pool_memory_pressure`, waiting a bounded
  operator-configured time for release before each retry. The eager policy
  remains the default and single-model eager CLI serving still routes through
  `serveModel()`.
- **Image serving**: Qwen image transport, host decode, and prepared-prompt
  cache shipped with explicit boundary — serve owns I/O and decode, transformers
  owns preprocessing and prompt expansion. OpenAI Chat/OpenResponses accept
  data-url and allowlisted remote HTTP(S) images; Anthropic Messages accepts
  local base64 and allowlisted remote HTTP(S) user image blocks through the same
  content route. Exact repeated Qwen image prompts now short-circuit full visual
  preparation when the media-aware prefix cache covers all expanded image tokens.
  Serve also keeps a per-adapter host-side decoded RGB LRU cache keyed by image
  byte digest plus Qwen preprocessor config; it stores no MLX tensors or visual
  embeddings, and remote URLs still refetch before content-addressed cache
  lookup. Local image `file_id` values are enabled only when the operator
  configures `localImageRoots` / `--local-image-root`; they resolve as relative
  image paths under those roots and do not create a general files API.
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
- **Phase 8 training proof hardening**: `bun run proof:training` is the root
  entrypoint for the canonical official-model proof. Reports now record adapter
  output location, selected LoRA target paths, trainable/total parameter counts,
  peak MLX memory, adapter save/reload/merge greedy-output equality for
  adapter-backed stages, and DPO profile-specific recipe knobs.

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
- Qwen image prefix short-circuit passed against cached
  `mlx-community/Qwen3.6-27B-4bit` with
  `bun run regression:qwen-image -- --report-dir .tmp/qwen-image-prefix-short-circuit`.
  The cold OpenAI Chat probe wrote `92` prompt-cache tokens; exact repeats
  across OpenAI Chat, OpenResponses, and Anthropic Messages read `92` cached
  tokens and stayed on the media route without introducing persistent visual
  embedding storage.
- Qwen decoded-image cache passed focused media/content tests, all
  `packages/serve` tests, full `bun run validate`, and
  `bun run regression:qwen-image -- --report-dir .tmp/qwen-decoded-image-cache`.
  The real run kept media requests on `single:media_input`; the cold OpenAI
  Chat request wrote `92` prompt-cache tokens, and exact repeats across OpenAI
  Chat, OpenResponses, and Anthropic Messages read `92` cached prompt tokens
  while reusing host-side decoded RGB bytes when the image digest and
  preprocessor matched.
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
  tool-result errors, and rich tool-result media until those semantics are
  implemented explicitly. Streaming tool-use output landed in the follow-up
  streaming tranche.
- OpenResponses function-tool adapter support passed focused protocol and HTTP
  route tests (`56 pass`) and full `bun run validate`. `/v1/responses` now
  accepts flat function tools, normalizes `function_call` /
  `function_call_output` history plus adjacent reasoning items into internal
  assistant/tool turns, formats generated tool-call envelopes as
  `function_call` output items, and rejects `parallel_tool_calls=false` with
  active tools, built-in/custom tools, and rich function outputs until those
  semantics are implemented explicitly.
- OpenResponses streaming function-tool output passed focused protocol, stream
  writer, and HTTP route tests (`59 pass`), `bun run check:coverage`, and full
  `bun run validate`. `/v1/responses stream=true` now accepts active function
  tools and emits Responses-shaped `response.output_item.added`,
  `response.function_call_arguments.delta`,
  `response.function_call_arguments.done`, and `response.output_item.done`
  events while keeping malformed/tool-looking text visible when tools are
  inactive.
- Anthropic Messages streaming tool-use passed focused protocol, stream writer,
  and HTTP route tests (`56 pass`), full `bun run validate`, and
  `bun run regression:qwen-gemma -- --profile quick` (`84` transformer-focused
  tests and `220` serve-focused tests). `/v1/messages stream=true` now accepts
  active client tools and emits Anthropic-shaped `tool_use` content blocks with
  `input_json_delta` argument deltas while keeping malformed/tool-looking text
  visible when tools are inactive.
- Phase 8 proof surfaces now have a cheap static gate. `bun run check:training-proofs`
  typechecks `examples/train-proof` and `examples/lora-finetune`, then runs
  helper/report-verifier tests (`17 pass`). The canonical training proof report
  carries machine-checkable verification evidence, and
  `examples/train-proof/verify-report.ts` can check an existing report without
  rerunning training. Tiny live official-model smokes passed individually for
  LoRA, QLoRA, SFT, and DPO; DPO verifier output now includes 41 checks in the
  live runner, including adapter equality and profile knob checks.
- Source-backed lazy model pool lifecycle passed focused lazy/CLI tests
  (`33 pass`), all `packages/serve` tests (`384 pass`), and full
  `bun run validate`. `bun run regression:qwen-gemma -- --profile quick` also
  passed (`84` transformer focused tests and `222` serve focused tests).
  Avicenna reviewed the tranche and identified three lifecycle blockers before
  final gating: racing cold loads across different model ids, lazy setup cleanup
  after model load, and queued cold loads starting after pool shutdown. All
  three are fixed and covered by regression tests; the final follow-up review
  reported no remaining blockers.
- Local model-root discovery now expands flat checkpoint folders and two-level
  `org/model` folders into source-backed lazy serving entries before startup.
  The scanner requires `config.json`, a supported autoregressive `model_type`,
  and safetensor weights; follows safetensor symlinks; defaults
  `--model-root` commands to lazy loading; and rejects empty roots before the
  server starts. `mlxts-serve discover --model-root <directory>` and
  `mlxts-serve status --base-url <url>` are finite AXI-shaped serve commands,
  returning compact structured stdout without starting the server or sending
  generation work.
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
- Source-backed model-load memory preflight passed focused model-loading tests
  (`28 pass`), all `packages/serve` tests (`358 pass`), and full
  `bun run validate`. Bohr's review caught the Hugging Face snapshot symlink
  case before commit; the final scanner stats `.safetensors` symlink targets
  and the regression test covers that path.
- Lazy model-pool pressure policy passed focused pool/source/CLI/metrics tests
  (`54 pass`), all `packages/serve` tests (`424 pass`), and focused serve
  typecheck. The default remains reject-only; `shed_non_pinned` is the explicit
  active guard for memory-pressure relief.
- Bounded lazy model-pool pressure shedding passed focused pool tests
  (`23 pass`) and focused serve typecheck. The policy now sheds one oldest
  eligible active non-pinned lease per pressure pass and fails boundedly if a
  pressure-cancelled lease does not release.
- Lazy model-pool pressure timeout CLI/API plumbing passed focused serve CLI
  and source-loading tests (`32 pass`) plus focused serve typecheck. The
  default pool timeout remains internal unless operators set
  `modelPressureReleaseTimeoutMs` / `--model-pressure-release-timeout-ms`.

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
- Next memory work is real-model `shed_non_pinned` pressure smoke coverage and
  richer placement policy once traces show whether oldest, largest estimated
  request memory, or operator priority should win. Lazy source loading, idle
  eviction, pinned models, TTL policy, local model-root discovery, bounded
  active shedding, and explicit active-guard policy are in place.
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
  cache backend work, live Phase 8 training-proof evidence, and broader Phase
  10 multimodal/diffusion research.
