# Runtime Review: Supervised Run Manager AXI Boundary

## Summary

The supervised-run manager now has an agent-facing command runner with stable
exit codes, structured stdout errors, injectable stdout, typed usage errors,
and compact structured start/stop/cancel acknowledgements. The nanoGPT manager
entrypoint uses that command runner directly. Detached supervisor startup,
control-file semantics, status files, `status --json`, acceptance polling, and
trainer execution remain unchanged.

## Files Reviewed

- `packages/train/src/supervised-run/manager.ts`
- `packages/train/src/supervised-run/manager-args.ts`
- `packages/train/src/supervised-run/manager-run.ts`
- `packages/train/src/supervised-run/manager-status.ts`
- `packages/train/src/supervised-run/index.ts`
- `packages/train/src/supervised-run/supervised-run.test.ts`
- `examples/nanogpt/src/run/manager.ts`
- `examples/nanogpt/src/run/supervised-run-config.ts`
- `examples/nanogpt/src/run/acceptance.ts`
- `examples/nanogpt/src/run/manager.test.ts`

## Tensor Lifetime Audit

The change is confined to command parsing, output routing, error formatting,
and manager acknowledgment text. No tensors, model parameters, gradients,
checkpoint tensors, optimizer state tensors, or MLX eval/synchronize paths
changed. The manager still launches the same detached supervisor command, and
the supervisor still owns trainer stdout/stderr pumping and status/event files.

## Memory / Performance Evidence

- `bun test packages/train/src/supervised-run/supervised-run.test.ts`: 15 pass, 0 fail.
- `bun test examples/nanogpt/src/run/manager.test.ts examples/nanogpt/src/run/acceptance.test.ts`: 27 pass, 0 fail.
- `bun run typecheck`: all workspaces passed.
- `bun run lint`: 694 files checked, no warnings.
- `bun run validate`: passed.

The focused nanoGPT acceptance test completed a tiny supervised run after the
manager output change. No training loop, checkpoint, optimizer, or model hot
path changed.

## Independent Review

Plato performed a read-only second-opinion review before the tranche landed.
The review recommended adding a command runner instead of changing supervisor
lifecycle semantics, typing usage errors separately from runtime/lifecycle
failures, preserving `status --json`, keeping detached no-prompt behavior, and
forwarding AXI manager stdout from the acceptance wrapper on manager start
failure.

## Remaining Risks / Follow-ups

Default text `status` and `watch` output remains human-oriented while
`status --json` remains the stable machine payload. A later CLI-only tranche can
add a compact TOON default status view if downstream agents prefer it over the
existing JSON flag.

## Out-of-scope drift noticed

None.
