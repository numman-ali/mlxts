# Runtime Review: Serve Runtime Strategy Seam

## Summary

Serving now has a typed internal runtime-strategy seam for the behavior it
actually supports today. Existing operator knobs resolve into scheduler `auto`,
managed model-precision cache, attention `auto`, model-native decoding,
streaming decode cadence, and admit-only memory preflight when configured.

The seam is intentionally derived from existing options. This does not add
operator-facing flags for paged KV, TurboQuant, speculative decoding, or custom
attention backends before those implementations have correctness and benchmark
proof.

## Files Reviewed

- `packages/serve/src/serve-runtime-strategy.ts`
- `packages/serve/src/model-server.ts`
- `packages/serve/src/server.ts`
- `packages/serve/src/server-info.ts`
- `packages/serve/src/types.ts`
- `packages/serve/src/transformers-engine.ts`
- `packages/serve/src/transformers-engine-routing.ts`
- `packages/serve/src/transformers-engine-continuous.ts`
- `packages/serve/src/transformers-engine-static.ts`
- `packages/serve/src/transformers-engine-streaming.ts`
- `packages/serve/src/transformers-engine-shared.ts`
- `packages/serve/src/cli.ts`
- `packages/serve/scripts/benchmark-serve.ts`
- `packages/serve/src/server.test.ts`
- `packages/serve/src/model-server.test.ts`
- `packages/serve/src/transformers-engine.test.ts`
- `packages/serve/src/cli.test.ts`
- `packages/serve/README.md`
- `docs/serving-runtime-strategy.md`
- `docs/runtime-optimization-matrix.md`

## Tensor Lifetime Audit

This tranche does not add or move MLX tensor-producing operations. The changed
runtime files route existing numeric knobs through a typed strategy object and
report that selected strategy through `/info`, route telemetry, CLI logs, and
benchmark reports.

Memory admission still calls the existing model-context estimator. The only
behavioral change there is reading the resolved `admit_only` memory strategy
instead of directly reading a loose `gpuMemoryUtilization` option.

## Memory / Performance Evidence

Focused serve validation passed:

- `bun test packages/serve/src/server.test.ts packages/serve/src/model-server.test.ts packages/serve/src/cli.test.ts packages/serve/src/transformers-engine.test.ts`

This change is not a decode-kernel or cache-update optimization, so no heavy MLX
benchmark was required. The focused tests cover `/info.runtime_strategy`,
route-decision telemetry, CLI formatting, streaming/static/continuous routing,
and memory-budget behavior over the existing fake model surfaces.

## Independent Review

Galileo reviewed the proposed seam before implementation. The main guidance was
to keep it derived from current behavior, avoid adding new public runtime flags,
avoid future literals such as `paged` or `turboquant` in code unions, preserve
direct engine defaults, and update exact route-event tests when telemetry grows.

The implementation follows that guidance: no new CLI flags or
`ModelServerRuntimeOptions` fields were added, future backends are not in the
runtime unions, direct transformer-engine construction still defaults to
`maxBatchSize=1`, `batchWindowMs=0`, `streamDecodeInterval=1`, and no memory
preflight unless configured.

## Remaining Risks / Follow-ups

The current strategy seam reports today's selected behavior; it is not yet a
policy engine. Future paged cache, TurboQuant-style KV, speculative decode, MTP,
or custom attention work should enter by adding real backend contracts, rejected
unsupported combinations, focused correctness tests, and benchmark evidence
before any operator-facing flag appears.

Benchmark reports now include route strategy fields, but production metrics are
still a separate missing capability. `/metrics` should be the next observability
surface before making stronger production-serving parity claims.
