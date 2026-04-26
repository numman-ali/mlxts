# Runtime Review: Serve Continuous Scheduler Telemetry

## Summary

Added scheduler-owned phase telemetry for the full-KV continuous batching path. The serve layer now receives explicit `generation_scheduler_phase` events for queueing, prefill start, admission, first token, finish, and cancellation. Endpoint benchmark reports use those scheduler-owned timings instead of inferring scheduler behavior from generic event gaps.

This does not broaden the batching claim. Qwen hybrid-cache and Gemma sliding/global-cache requests still route through the single model lane, and the Qwen/Gemma regression matrix now asserts zero continuous scheduler phases for those fallback rungs.

## Files Reviewed

- `packages/transformers/src/generation.ts`
- `packages/transformers/src/index.ts`
- `packages/transformers/src/infrastructure/generation/continuous-batch.ts`
- `packages/transformers/src/infrastructure/generation/continuous-batch-events.ts`
- `packages/transformers/src/infrastructure/generation/continuous-batch-telemetry.ts`
- `packages/transformers/src/infrastructure/generation/continuous-batch-types.ts`
- `packages/serve/src/types.ts`
- `packages/serve/src/transformers-engine-continuous.ts`
- `packages/serve/src/cli.ts`

## Tensor Lifetime Audit

No new tensor-producing operations were added. The scheduler still owns the same `BatchKVCache`, `currentToken`, chunk prefill, filter, and merge paths as before. The new code records timestamps, queue counts, ids, and generated-token counts around existing state transitions.

The active tensor lifecycle remains visible in `continuous-batch.ts`: ready-row token tensors are either installed, combined, or freed; emitted tokens are freed after the next token is sampled or after the final active row completes; prefilling caches are disposed on abort and failure cleanup.

## Memory / Performance Evidence

Focused checks run locally:

- `bun test packages/transformers/src/infrastructure/generation/continuous-batch.test.ts packages/serve/src/transformers-engine.test.ts packages/serve/src/cli.test.ts packages/serve/scripts/benchmark-serve.test.ts packages/serve/scripts/regression-serve-matrix.test.ts`
- `bun run --filter '@mlxts/serve' typecheck`
- `bun run --filter '@mlxts/transformers' typecheck`
- `bun run check:file-lines`
- `bun run check:tensor-lifetimes`
- `bun run lint`
- `bun run check:runtime-review`
- `bun run regression:qwen-gemma -- --profile quick`
- `bun run check:coverage`
- `bun run bench:generation --model mlx-community/Llama-3.2-1B-Instruct-4bit --prompt-tokens 16 --generation-tokens 16 --trials 1 --memory-sample-interval 16`: `generation_tps=508.168`, `peak_memory=0.717 GB`, `active_delta=-0.000 GB`, `evals_per_token=1.00`.
- `bun run bench:generation:parity --model mlx-community/Llama-3.2-1B-Instruct-4bit --prompt-tokens 16 --generation-tokens 16 --trials 1 --memory-sample-interval 16 --skip-mlx-lm-reference`: `generation_tps=517.936`, `peak_memory=0.717 GB`, `active_delta=-0.000 GB`, `evals_per_token=1.00`. External `mlx-lm` reference was intentionally skipped for this tiny local guard run.

This is an observability change. It adds small JS event objects around scheduler state transitions, not extra MLX graph work. The first-token and finish events are request-level; the admitted event is batch-level. There is no per-token operator log line in the default CLI path.

## Independent Review

Sub-agent Aristotle independently reviewed the scheduler telemetry seam before implementation. The recommendation was to avoid reinterpreting `generation_batch_start`, keep Qwen/Gemma fallback honesty, emit scheduler-owned timing from the scheduler itself, and prove benchmark aggregation plus zero scheduler phases on fallback rungs.

## Remaining Risks / Follow-ups

Continuous batching remains intentionally limited to non-streaming greedy full-KV models. The next deeper serving work is cache-semantics support for Qwen hybrid and Gemma sliding/global patterns, then streaming collectors and prefix/paged cache telemetry once those strategies exist.
