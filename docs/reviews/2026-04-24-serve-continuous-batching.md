# Serve Continuous Batching

## Summary

Added the first scheduler-owned continuous batching tranche for serving:
full-KV, greedy, non-streaming requests can now join an active decode loop
between token steps. This replaces the loaded-model server's previous
`micro-batching -> concurrency gate -> transformers` composition for eligible
requests, because that outer gate serialized the very work continuous batching
needs to own.

The implementation is intentionally narrow. Qwen hybrid caches, Gemma
sliding/global caches, sampled decoding, streaming collectors, prefix/paged
cache, and multimodal batching still fall back to the single model execution
lane until those cache semantics are represented.

## Files Reviewed

- `packages/transformers/src/infrastructure/generation/continuous-batch.ts`
- `packages/transformers/src/generation.ts`
- `packages/transformers/src/index.ts`
- `packages/serve/src/model-execution-lane.ts`
- `packages/serve/src/transformers-engine-continuous.ts`
- `packages/serve/src/transformers-engine-generation.ts`
- `packages/serve/src/transformers-engine.ts`
- `packages/serve/src/model-server.ts`
- `packages/serve/src/cli.ts`
- `packages/serve/src/types.ts`
- `packages/serve/scripts/benchmark-serve.ts`

## Runtime Invariants

- The scheduler owns waiting and active rows, not an HTTP admission wrapper.
- Active rows share one `BatchKVCache`; finished or aborted rows are removed via
  `filter()`, and newly admitted rows are merged via `extend()`.
- The cache invariant is explicit: the cache contains prompt plus already-fed
  generated tokens, while `currentToken` holds the sampled token that has not
  been fed back yet.
- A bounded `ModelExecutionLane` serializes fallback generation and streaming
  against the scheduler by default, supports explicit `maxConcurrentRequests`,
  and lets queued work abort before dispatch.
- `batchWindowMs` controls the initial scheduler admission delay so nearby
  eligible requests can join before the first active decode loop starts.
- Benchmark output now reports `continuous_admissions` separately from
  `static_batches` and `admission_batches`.

## Tensor Lifetime Audit

- `currentToken` is freed whenever it is replaced, filtered, or the loop exits.
- Temporary prompt/input tensors use `using` declarations.
- Extended one-row batch caches transfer their arrays into the active cache and
  are disposed immediately after `extend()`.
- Scheduler failures reject all queued/active requests and dispose active cache
  and token handles.
- The new scheduler file keeps local tensor-producing calls visible rather than
  hiding owned intermediates in nested expressions.

## Independent Review

Sub-agent review compared the tranche against `.reference/mlx-lm`,
`.reference/vllm-mlx`, `.reference/omlx`, and `.reference/rapid-mlx`. The
consistent reference pattern is a scheduler-owned loop with waiting/running
queues, active-row filtering, row insertion between decode steps, and separate
handling for hybrid/sliding caches. That is why this implementation starts with
the full-KV greedy subset instead of claiming Qwen/Gemma batching prematurely.
Follow-up review caught three implementation gaps before handoff: the loaded
model path still needed a bounded lane around fallback generation/streaming,
queued lane acquisition needed abort support, and `batchWindowMs` needed to
affect scheduler admission rather than only being reported. Those are now part
of the implementation. Whole-prompt prefill remains a documented next slice,
not something hidden as solved.

## Memory / Performance Evidence

- `bun test packages/transformers/src/infrastructure/generation/continuous-batch.test.ts`
  passed with scheduler tests proving mid-decode admission, batch-window
  admission delay, mixed-length row filtering, active abort removal,
  all-active-aborted recovery, and abort while waiting behind the model lane.
- `bun test packages/serve/src/model-execution-lane.test.ts packages/serve/src/model-server.test.ts packages/serve/src/transformers-engine.test.ts packages/serve/src/cli.test.ts packages/transformers/src/infrastructure/generation/continuous-batch.test.ts`
  passed: 49 tests.
- Tiny cached live endpoint probe:
  `bun run bench:serve --model mlx-community/Llama-3.2-1B-Instruct-4bit --model-id llama-local --prompt-tokens 16 --generation-tokens 16 --concurrency 1,2 --trials 1 --no-warmup --greedy --ignore-eos --max-batch-size 4 --batch-window-ms 2 --max-concurrent-requests 1 --max-prompt-tokens 16 --max-total-tokens 32 --gpu-memory-utilization 0.85`.
  Concurrency 1 reported `completion_tps=127.140`,
  `continuous_admissions=1`, `static_batches=0`, `admission_batches=0`, and
  `active_delta=0.000 GB`. Concurrency 2 reported `completion_tps=261.027`,
  `continuous_admissions=1`, `static_batches=0`, `admission_batches=0`, and
  `active_delta=0.000 GB`.
- Tiny cached `bench:generation:parity` guard:
  `bun run bench:generation:parity --model mlx-community/Llama-3.2-1B-Instruct-4bit --prompt-tokens 16 --generation-tokens 16 --trials 1 --skip-mlx-lm-reference`.
  This in-process `bench:generation` run reported `generation_tps=378.258`,
  `peak_memory=1.094 GB`, `active_delta=0.000 GB`,
  `active_slope_mb_per_token=0.02`, and `evals_per_token=1.00`.
- `bun run typecheck` passed.
- `bun run lint` passed.
- `bun run check:file-lines` passed.
- `bun run check:tensor-lifetimes` passed.
- `bun run check:assertions` passed.
- `bun run check:runtime-review` passed.
- `bun run check:coverage` passed.

## Remaining Risks / Follow-ups

- No Qwen or Gemma continuous-batching claim is made here. Their cache layouts
  need separate scheduler/cache work.
- Streaming still uses the single-request lane; true streaming collectors are a
  later tranche.
- The scheduler currently uses whole-prompt prefill for admitted rows, with
  abort checks around admission and before decode. Chunked prefill fairness is
  the next scheduler-quality slice before long prompt concurrency claims.
