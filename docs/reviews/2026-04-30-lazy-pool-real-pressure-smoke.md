# Lazy Pool Real Pressure Smoke

## Summary

Added a dedicated real-checkpoint regression command for the lazy source-backed
model pool pressure policy. The smoke starts a lazy multi-model endpoint with a
Gemma streaming request active, constrains the model-load memory budget around
the larger Qwen estimate, then sends a Qwen request that must pressure-abort the
active non-pinned request, retry the blocked load, and complete.

This is evidence scaffolding for the existing `shed_non_pinned` contract, not a
new serving behavior.

## Files Reviewed

- `packages/serve/scripts/regression-lazy-pool-pressure.ts`
- `packages/serve/scripts/regression-lazy-pool-pressure.test.ts`
- `package.json`
- `packages/serve/package.json`

## Tensor Lifetime Audit

The changed code is regression harness and package metadata. It does not create
or retain `MxArray` values directly. Real model execution stays inside existing
`serveModels()` and transformer loading paths.

## Runtime Evidence

- `bun test packages/serve/scripts/regression-lazy-pool-pressure.test.ts`: passed,
  `5` tests / `14` assertions.
- `bun run --filter '@mlxts/serve' typecheck`: passed.
- `bun run check:file-lines`: passed.
- `bun run check:runtime-review`: passed, no runtime-sensitive production
  changes detected.
- `bun run check:coverage`: passed.
- `bun run validate`: passed.
- `bun run regression:lazy-pool-pressure -- --report-dir .tmp/lazy-pool-pressure-real`:
  passed. The report recorded Gemma active streaming, Qwen blocked loading,
  `gpu_memory_utilization=0.330994`, `pressure_events=2`,
  `abort_active=1`, `aborted_requests=1`, the aborted active stream id
  `cmpl-6a789150-85d7-4848-92ca-814d9ca75271`, Qwen blocked request completion
  with `outputChars=35`, and six model-pool pressure metrics lines.

## Independent Review

Lovelace reviewed the first harness shape and flagged three issues: CLI parsing
acquired the runtime lock before finite usage failures, the active stream id was
not tied to the aborted request evidence, and numeric parsing accepted partial
strings. The script now parses before locking, rejects partial numeric values,
records SSE ids and error codes from the active stream, and asserts that the
`abort_active` action targets the blocked model-load request and names the
active stream id.

## Remaining Risks

The smoke depends on cached real checkpoints and MLX memory telemetry. If the
host cannot fit the larger blocked model inside the constrained budget, the
script fails before starting the endpoint instead of producing ambiguous
pressure evidence.

The current streaming pressure-cancel path emits a Bun-visible stack trace to
stderr while still producing correct report evidence and structured stdout
summary. Cleaning the HTTP stream terminal shape is a follow-up product polish
tranche; this smoke intentionally records the current behavior.

## Out-of-scope Drift Noticed

- This does not implement richer placement policy, victim scoring, paged KV, or
  quantized KV. It only proves the current explicit pressure policy with real
  checkpoints.
- The non-streaming active-request variant did not reliably trigger load
  pressure on the local machine because MLX active-memory telemetry stayed below
  the constrained budget until the request had already completed. The harness
  therefore proves the active-stream pressure shape that creates enough live
  request memory to exercise `shed_non_pinned`.
