# Runtime Review: Qwen/Gemma Serve Regression Matrix

## Summary

This slice adds a tiered Qwen/Gemma regression command that composes existing
model and serving benchmarks instead of inventing a new harness. The default
profile stays cheap and fixture-driven, while `real` and `substantial` profiles
run cached checkpoint smoke tests under the shared MLX runtime lock.

The serving matrix hard-fails on structural endpoint regressions: missing
stream chunks or bytes, missing route decisions, unexpected route/reason,
unexpected finish reasons, collapsed token counts, memory budget failures, and
throughput falling below model-specific budgets.

## Files Reviewed

- `AGENTS.md`
- `MEMORY.md`
- `docs/runtime-safety.md`
- `package.json`
- `packages/serve/README.md`
- `packages/serve/package.json`
- `packages/serve/scripts/benchmark-serve.ts`
- `packages/serve/scripts/regression-serve-matrix.ts`
- `packages/serve/scripts/regression-serve-matrix.test.ts`
- `packages/transformers/README.md`
- `packages/transformers/scripts/benchmark-long-context.ts`
- `packages/transformers/scripts/benchmark-long-context.test.ts`
- `scripts/regression-qwen-gemma.ts`
- `scripts/regression-qwen-gemma.test.ts`

## Tensor Lifetime Audit

No model hot-path tensor code changed. The new code orchestrates existing
benchmark commands, reads JSON reports, and asserts metrics. The long-context
benchmark assertion runs after each result is produced and does not retain
additional MLX arrays.

## Memory / Performance Evidence

- `bun run regression:qwen-gemma -- --profile quick`
  - Transformer focused matrix: `66` tests passed.
  - Serve focused matrix: `120` tests passed.
- `bun run regression:qwen-gemma -- --profile real`
  - Qwen load: `15.134GB` active.
  - Qwen direct decode smoke: `1024x128`, `29.119 tok/s`, `17.184GB` peak,
    `0.14 MB/token` active slope, `1.00` evals/token.
  - Gemma 4 load: `9.295GB` active.
  - Gemma 4 direct decode smoke: `1024x128`, `81.755 tok/s`, `9.893GB` peak,
    `-0.04 MB/token` active slope, `1.00` evals/token.
  - Qwen endpoint stream smoke: `1024x128@1`, `13.976 tok/s` end-to-end,
    `32.367 tok/s` post-TTFT, `17.184GB` peak, route `single:streaming`.
  - Gemma 4 endpoint stream smoke: `1024x128@1`, `75.276 tok/s` end-to-end,
    `9.892GB` peak, route `single:streaming`.

## Independent Review

Two explorer sub-agents reviewed the plan before implementation. The repo-local
review recommended a thin assertion layer over `bench:serve` and the existing
transformer matrix. The reference-repo review recommended tiered smoke/check/full
profiles, separate direct model metrics from serving metrics, and avoiding any
claim of Qwen/Gemma continuous batching until their cache semantics are
implemented.

Those findings were integrated into the command structure and documentation.

## Remaining Risks / Follow-ups

- The `real` profile is a smoke, not the long-context/output ladder.
- The `substantial` profile adds a 32k Qwen retrieval assertion, but 64k/128k
  and 10k/20k output rungs remain overnight evidence, not default gates.
- Qwen and Gemma streaming endpoint smoke intentionally asserts the current
  `single:streaming` route. When true streaming batching lands, this assertion
  should change with the implementation and evidence.
