# Continuity

Compact handoff for long-running `mlxts` work. Durable doctrine lives in
`AGENTS.md`, durable learnings in `MEMORY.md`, and current-phase state that
should survive context compaction lives here.

## Current Focus

Qwen 3.6 27B serving/inference quality is the active critical path. The goal is
staged parity evidence against `mlx-lm`, long-output stability, and long-context
capability — not a single short benchmark. The OpenResponses text slice and
Anthropic Messages adapter are part of the same serving-quality push: usable
text endpoints while benchmark and scheduler work continues.

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
- **Image serving**: Qwen image transport, host decode, and prepared-prompt
  cache shipped with explicit boundary — serve owns I/O and decode, transformers
  owns preprocessing and prompt expansion.
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

## Next Work

- Remaining Qwen gap is mostly peak memory versus `mlx-lm` plus small
  paired-run variance. Profile full-attention KV representation, cache-buffer
  accounting, and wrapper/FFI overhead before scattering micro-optimizations.
- Next scheduler tranche is cache-semantics work: chunked prefill fairness,
  streaming collectors, sampled batch decode, then Qwen hybrid recurrent /
  full-attention caches and Gemma sliding/global caches.
- Prefix-cache cache-hit seeding now reaches continuous scheduler rows through
  one-row managed batch-cache restore. Hits now classify exact, shorter-prefix,
  longer-source trim, and LCP reuse and expose source `CacheLayerKind` /
  trimmability metadata; paged/block-deduplicated storage remains the next
  cache-backend tranche.
- Real serving memory preflight should reject admission deterministically
  based on family geometry + cache layout, not advisory warnings.
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
