# Runtime Review: Prefix Cache Snapshot and Serve Reuse

## Summary

`@mlxts/transformers` now has a family-owned `TransformerCache.snapshot()`
contract that can fork reusable cache state without pretending every model has
the same KV layout. Full KV caches support exact and prefix-trimmed forks.
Sliding and mixed layer-pattern caches support exact forks only, preserving
their ring-buffer cursor so continuation semantics stay intact. Qwen 3.5/3.6
hybrid text caches snapshot full-attention KV state plus linear-attention
recurrent/conv state, and only allow prefix trimming for all-full variants.

`@mlxts/serve` now uses that foundation for message/chat requests. It keeps a
small per-engine prompt-prefix snapshot store, chooses a single-request lane for
message prompts so forked single caches can be used safely, reports cache
hit/miss/write telemetry in logs and `/metrics`, and includes cache read/write
token accounting in OpenAI-compatible usage. The generation layer now separates
sampler history from cache suffix input so prefix reuse does not break
repetition-penalty semantics. When the small prompt-cache store is at capacity,
it evicts the shortest reusable prefix first, with LRU only as a tie-breaker, so
tiny probe prompts do not replace large agent-conversation snapshots.

## Files Reviewed

- `packages/transformers/src/types.ts`
- `packages/transformers/src/index.ts`
- `packages/transformers/src/infrastructure/cache/runtime.ts`
- `packages/transformers/src/infrastructure/cache/single.ts`
- `packages/transformers/src/infrastructure/cache/index.test.ts`
- `packages/transformers/src/infrastructure/generation/runtime.ts`
- `packages/transformers/src/infrastructure/generation/runtime-streaming.ts`
- `packages/transformers/src/infrastructure/generation/helpers.test.ts`
- `packages/transformers/src/generation.test.ts`
- `packages/transformers/src/families/qwen3_5/cache.ts`
- `packages/transformers/src/families/qwen3_5/cache.test.ts`
- `packages/serve/src/types.ts`
- `packages/serve/src/cli.ts`
- `packages/serve/src/protocols/openai-responses-formatting.ts`
- `packages/serve/src/model-router.ts`
- `packages/serve/src/model-router.test.ts`
- `packages/serve/src/model-server.ts`
- `packages/serve/src/request-limits.ts`
- `packages/serve/src/request-limits.test.ts`
- `packages/serve/src/serve-metrics.ts`
- `packages/serve/src/serve-metrics.test.ts`
- `packages/serve/src/transformers-engine.ts`
- `packages/serve/src/transformers-engine-batch.ts`
- `packages/serve/src/transformers-engine-generation.ts`
- `packages/serve/src/transformers-engine-prefix-cache.ts`
- `packages/serve/src/transformers-engine-prefix-cache.test.ts`
- `packages/serve/src/transformers-engine-static.ts`
- `packages/serve/src/transformers-engine-streaming.ts`
- `packages/serve/src/transformers-engine.test.ts`
- `docs/serving-runtime-strategy.md`
- `docs/runtime-optimization-matrix.md`
- `PLAN.md`
- `MEMORY.md`

## Tensor Lifetime Audit

Snapshot arrays are cloned with `cloneCacheArray()`, which materializes an owned
MLX array before the snapshot is exposed and frees the clone if materialization
throws. Multi-array snapshot helpers clean up already-cloned arrays if a later
clone fails. Forked generic caches clone from the snapshot into a fresh cache
state, so later in-place cache writes do not mutate the reusable snapshot.
Forked Qwen full-attention layers materialize `updateAndFetch()` during restore
before temporary source arrays are freed.

Generic cache snapshots dispose all cloned layer arrays through
`disposeLayerStateSnapshot()`. Qwen snapshots dispose full-attention and linear
state arrays through family-local snapshot disposal helpers. Fork failure paths
dispose the partially restored cache before rethrowing.

Sliding-window snapshots preserve physical ring-buffer tensors plus cursor state
for exact continuation. Prefix trimming is deliberately disabled for any cache
with sliding layers or Qwen linear-attention state because the retained state is
not safely representable as an arbitrary shorter prefix.

Serve owns forked prefix-hit caches and disposes them after non-streaming or
streaming generation completes, including synchronous stream setup failures.
Stored prompt snapshots are owned by the `PromptPrefixCache` and disposed on
eviction or cache disposal. Transformer generation snapshot callbacks transfer
ownership to the callback; if the callback throws, the generation layer disposes
the snapshot before rethrowing. Transformer engines now expose an optional
dispose hook, request-limit and model-router wrappers forward it, and model
server shutdown invokes it before optionally disposing model weights.

## Memory / Performance Evidence

Focused correctness and ownership coverage passed:

- `bun test packages/transformers/src/infrastructure/cache/index.test.ts packages/transformers/src/families/qwen3_5/cache.test.ts packages/transformers/src/infrastructure/generation/helpers.test.ts`
- `bun test packages/transformers/src/generation.test.ts packages/transformers/src/infrastructure/cache/index.test.ts packages/transformers/src/families/qwen3_5/cache.test.ts packages/serve/src/transformers-engine.test.ts packages/serve/src/cli.test.ts packages/serve/src/serve-metrics.test.ts packages/serve/src/server.test.ts packages/serve/src/protocols/openai-chat-completions.test.ts packages/serve/src/protocols/openai-completions.test.ts`
- `bun test packages/serve/src/transformers-engine-prefix-cache.test.ts packages/serve/src/transformers-engine.test.ts packages/serve/src/serve-metrics.test.ts`
- `bun test packages/serve/src/transformers-engine-prefix-cache.test.ts packages/serve/src/server-stream-runtime.test.ts packages/serve/src/serve-runtime-strategy.test.ts`
- `bun test packages/serve/src/transformers-engine.test.ts packages/serve/src/request-limits.test.ts packages/serve/src/model-router.test.ts packages/serve/src/transformers-engine-prefix-cache.test.ts packages/transformers/src/infrastructure/cache/index.test.ts packages/transformers/src/families/qwen3_5/cache.test.ts`
- `bun run typecheck`
- `bun run check:tensor-lifetimes`

Lightweight hot-path probes passed after the change:

- `bun run bench:generation --model mlx-community/Llama-3.2-1B-Instruct-4bit --prompt-tokens 16 --generation-tokens 16 --trials 1 --memory-sample-interval 8`
  reported `498.358` generation tok/s, `0.717 GB` peak memory, and `1.00`
  evals/token.
- `bun run bench:generation:parity --model mlx-community/Llama-3.2-1B-Instruct-4bit --prompt-tokens 16 --generation-tokens 16 --trials 1 --memory-sample-interval 8 --skip-mlx-lm-reference`
  reported `509.903` generation tok/s, `0.717 GB` peak memory, and `1.00`
  evals/token. MLX-LM reference capture was skipped for this lightweight local
  probe; longer Qwen/Gemma parity remains part of the serving-prefix reuse
  acceptance tranche.

Unit coverage now proves repeated message prompts reuse the prefix cache for
both buffered and streaming chat requests: the second request skips the prefill
forward, reports `cacheReadTokens`, and still reports the full prompt token
count. `/metrics` coverage records prompt-cache hit/miss/write events plus
prompt/read/write token counters. No long Qwen/Gemma endpoint benchmark was run
in this slice; the next acceptance pass should run the live Pi/Qwen chat loop
and compare repeated-turn TTFT before/after.

## Independent Review

Peirce, Helmholtz, and Euclid independently reviewed the prefix-cache problem
before implementation. Their shared guidance was to make the reusable seam
family-owned rather than a generic KV-array list: LLaMA-style full KV can trim
prefixes, Gemma/Mistral sliding or layer-pattern caches must preserve logical
context separately from retained state, and Qwen hybrid caches must treat linear
recurrent/conv state as exact-boundary only.

Hume reviewed the concrete implementation afterward. That review identified
disposed snapshots as a use-after-dispose risk, called out ambiguous
`isTrimmable()` behavior for sliding and mixed layer-pattern caches, and asked
for mixed layer-pattern plus repeated-Qwen-fork tests. The implementation now
makes disposed snapshots non-forkable, aligns `isTrimmable()` with snapshot
behavior for sliding/mixed caches, and adds those focused tests.

Kepler reviewed the final diff and found three ownership gaps: streaming forked
caches were not protected if stream setup threw synchronously, snapshot cloning
needed partial-clone cleanup, and server shutdown needed to dispose engine-owned
prompt snapshots. The implementation now moves streaming setup inside the
`try/finally`, makes clone helpers clean up on failure, and threads optional
engine disposal through request-limit, model-router, and server shutdown paths.

## Remaining Risks / Follow-ups

The current serve cache is intentionally small and single-request oriented. It
solves the repeated interactive chat path without mixing external single-cache
forks into the continuous batch scheduler. Multi-agent/high-concurrency prompt
reuse should be designed as a later batch-cache or paged-cache layer rather than
bolted onto this first single-lane path.

Batch/continuous cache snapshotting is intentionally out of scope here. The
first prompt-cache reuse path should fork single-request snapshots before
admission into batching, then later decide whether paged or batch-native cache
reuse earns a deeper runtime seam.
