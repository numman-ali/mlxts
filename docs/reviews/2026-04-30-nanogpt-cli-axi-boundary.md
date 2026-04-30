# Runtime Review: nanoGPT CLI AXI Boundary

## Summary

The nanoGPT example CLI now treats command parsing, help, and failures as an
agent-facing boundary. Help output is compact structured stdout, usage errors
return exit code 2, runtime errors return exit code 1, and both error classes
emit structured stdout errors. Missing flag values are rejected before data
loading, checkpoint loading, model construction, or training startup.

Successful command contracts are unchanged: `generate` still prints raw text by
default, `generate --json` still prints one JSON object, `export` still prints
the output path, and `train --json` still emits JSONL training events.

## Files Reviewed

- `examples/nanogpt/src/cli.ts`
- `examples/nanogpt/src/cli/commands.ts`
- `examples/nanogpt/src/cli/help.ts`
- `examples/nanogpt/src/cli/shared.ts`
- `examples/nanogpt/src/cli.test.ts`

## Tensor Lifetime Audit

The change is limited to argument parsing, flag arity checks, help formatting,
exit-code mapping, and error output routing. No model forward path, optimizer
step, checkpoint tensor serialization, generated-token loop, training event
handler, or MLX runtime limit application changed. The new missing-value checks
run before any tensor-owning objects are created.

## Memory / Performance Evidence

- `bun test examples/nanogpt/src/cli.test.ts`: 18 pass, 0 fail.
- `bun run --filter nanogpt typecheck`: passed.
- `bun run validate`: passed.

The parser and output changes do not alter training math, generation length,
sampling, checkpoint writes, or optimizer state.

## Independent Review

Ramanujan performed a read-only second-opinion review. The recommendation was
to harden only the `examples/nanogpt/src/cli.ts` parser/error boundary, preserve
successful train/generate/export outputs, add structured stdout errors, fix the
usage/runtime exit-code mapping, reject unknown commands, and catch missing flag
values before runtime work.

## Remaining Risks / Follow-ups

The default `train` progress view remains human-oriented stderr output, while
`train --json` remains the stable machine event stream. A later explicit output
tranche can decide whether the non-JSON training success summary should gain a
compact AXI footer without changing the event stream.

## Out-of-scope drift noticed

`examples/nanogpt/src/bench/memory.ts`, `examples/nanogpt/src/run/acceptance.ts`,
and `examples/nanogpt/src/run/soak.ts` are separate agent-facing CLI surfaces
with their own output contracts. They were not changed in this tranche.
