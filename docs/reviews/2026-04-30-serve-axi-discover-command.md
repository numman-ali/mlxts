## Summary

Added `mlxts-serve discover --model-root <directory>` as the first finite
AXI-shaped serve CLI command. The command scans local model roots without
starting a server, writes compact TOON-style discovery output to stdout, reports
usage and discovery errors as structured stdout, and leaves the long-running
server operator stream unchanged.

## Files Reviewed

- packages/serve/src/cli-discovery-command.ts
- packages/serve/src/cli.ts
- packages/serve/src/cli-usage.ts
- packages/serve/src/index.ts

## Tensor Lifetime Audit

The changed CLI and formatting files do not allocate, transform, retain, or
dispose `MxArray` values. Local discovery reads filesystem metadata and JSON
checkpoint configs only.

## Memory / Performance Evidence

- `bun test packages/serve/src/cli.test.ts packages/serve/src/model-loading/discovery.test.ts`
  passed with 25 tests.
- `bun run --filter '@mlxts/serve' typecheck` passed.
- `bun run --filter '@mlxts/serve' test` passed with 402 tests.
- `bun run --filter '@mlxts/serve' test:coverage` passed with 402 tests.
- `bun run check:runtime-review` passed.
- `bun run check:file-lines` passed.
- `bun run validate` passed.

No generation hot path changed. The new command exits after filesystem
discovery and does not start model loading or serving.

## Independent Review

Parfit performed a read-only AXI CLI audit and recommended finite output
contracts before any broader CLI stream migration. This tranche follows that
review by adding a bounded finite discovery command and leaving server startup,
generation, scheduler, and REPL behavior out of scope.

## Remaining Risks / Follow-ups

`mlxts-serve` startup still emits human-oriented long-running operator logs.
A later AXI migration tranche must decide the stdout/stderr contract for server
readiness and progress without breaking existing local automation. `mlxts-agent`
still needs a non-TTY guard and structured finite usage/runtime errors.
