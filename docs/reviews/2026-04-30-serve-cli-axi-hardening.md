# Runtime Review: Serve CLI AXI Hardening

## Summary

The main `mlxts-serve` CLI boundary now matches the finite `discover` and
`status` commands for pre-start agent ergonomics: usage errors return exit code
2, structured errors go to stdout, help is compact and structured, and flag
values reject missing flag-shaped tokens before server work starts. The
long-running server path keeps serving behavior unchanged while routing load,
ready, warning, shutdown, and generation lifecycle diagnostics away from
agent-consumable stdout.

## Files Reviewed

- `packages/serve/src/cli-discovery-command.ts`
- `packages/serve/src/cli-flag-readers.ts`
- `packages/serve/src/cli-options.ts`
- `packages/serve/src/cli-usage.ts`
- `packages/serve/src/cli.ts`
- `packages/serve/src/observability/cli-status-command.ts`
- `packages/serve/src/cli.test.ts`
- `packages/serve/src/observability/cli-status-command.test.ts`

## Tensor Lifetime Audit

The change is confined to CLI parsing, formatting, output channels, and tests.
No `MxArray` values, native handles, model caches, scheduler rows, request
streams, or disposal paths changed. The server startup path still passes the
same parsed `ServeModelOptions` and `ServeModelsOptions` into the same serving
entry points.

## Memory / Performance Evidence

- `bun test packages/serve/src/cli.test.ts packages/serve/src/observability/cli-status-command.test.ts`: 30 pass, 0 fail.
- `bun test packages/serve`: 460 pass, 0 fail.
- `bun run typecheck`: all workspaces passed.
- `bun run lint`: 694 files checked, no warnings.
- `bun run scripts/check-file-lines.ts`: all active production files are <= 500 lines.
- `bun run validate`: passed.

No generation, cache, batching, model loading, or inference hot path changed.

## Independent Review

Peirce performed a read-only second-opinion review before finalizing the
implementation. The review identified the main serve wrapper as the highest
value AXI gap, recommended exit code 2 plus structured stdout for usage errors,
recommended keeping finite `discover` and `status` intact, and called out the
need to move long-running server diagnostics off stdout without changing
serving semantics.

## Remaining Risks / Follow-ups

The long-running server command still exposes a human-readable readiness block
as diagnostics. A future dedicated finite startup/status command can make the
operator readiness surface fully TOON-shaped without changing the server
process contract.

## Out-of-scope drift noticed

None.
