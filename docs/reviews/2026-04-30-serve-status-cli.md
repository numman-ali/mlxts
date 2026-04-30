# Runtime Review: Serve Status CLI

## Summary

This tranche adds `mlxts-serve status`, a finite AXI-shaped command for
inspecting an already-running `@mlxts/serve` endpoint. The command checks
`/health` and `/info`, emits compact structured stdout for served models,
limits, capabilities, and runtime strategy, and never starts a server or sends
generation work.

## Files Reviewed

- `packages/serve/src/cli.ts`
- `packages/serve/src/cli-usage.ts`
- `packages/serve/src/observability/cli-status-command.ts`

## Tensor Lifetime Audit

No tensor-producing code changed. The new command fetches JSON over HTTP,
normalizes flags and environment fallbacks, validates the `/info` payload, and
formats stdout. It does not construct, retain, fork, mutate, dispose, or eval
`MxArray` instances.

## Runtime Behavior Audit

The server startup and generation paths are unchanged. `runServeCli()` now
dispatches the finite `status` subcommand before parsing model-serving options,
matching the existing `discover` subcommand pattern. The status implementation
uses injected `fetch` in tests and defaults to global `fetch` in real use. It
queries only `/health` and `/info`, applies bearer auth only when provided, and
never logs or echoes access tokens.

## Memory / Performance Evidence

The command does not run model inference, load checkpoints, alter scheduler
state, or touch cache storage. Runtime cost is two bounded HTTP requests with a
default five-second timeout. Generation and serving benchmarks were not run
because no model execution path changed.

Focused validation:

- `bun test packages/serve/src/observability/cli-status-command.test.ts packages/serve/src/cli.test.ts`
- `bun run --filter '@mlxts/serve' typecheck`
- `bun test packages/serve`
- `bun run check:file-lines`
- `bun run check:runtime-review`
- `bun run validate`

## Independent Review

Boyle, a read-only explorer sub-agent, recommended `mlxts-serve status` as the
next narrow AXI tranche because it helps agents answer whether a local endpoint
is usable, which model ids are served, and which limits apply without changing
serving runtime behavior.

## Out-of-scope Drift Noticed

None.

## Remaining Risks / Follow-ups

The long-running server startup stream still uses the legacy human-oriented
terminal format. That migration remains a separate AXI tranche because it needs
status/report surfaces rather than a one-shot data command.
