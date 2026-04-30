# Runtime Review: nanoGPT Memory Benchmark AXI Boundary

## Summary

The nanoGPT memory benchmark now parses help and usage errors before acquiring
the shared MLX runtime lock. Help output is compact structured stdout, usage
errors return exit code 2 with structured stdout, runtime failures return exit
code 1 with structured stdout, and lock conflicts still prevent the benchmark
from running when another heavy command owns the runtime.

Benchmark semantics are unchanged: the same scenarios run with the same model
and tensor paths, `--json` still emits the benchmark result object, and the
default success output remains the existing compact key/value summary.

## Files Reviewed

- `examples/nanogpt/src/bench/memory.ts`
- `examples/nanogpt/src/bench/memory.test.ts`

## Tensor Lifetime Audit

The changed code is the CLI boundary around the benchmark. Tensor-producing
scenario runners, explicit `using` lifetimes, model disposal, attention forward
calls, loss computation, synchronization, memory sampling, and allocator cache
reset behavior are unchanged. The only runtime-order change is that parse/help
and usage validation run before the heavy-command lock is acquired.

## Memory / Performance Evidence

- `bun test examples/nanogpt/src/bench/memory.test.ts`: 4 pass, 0 fail.
- `bun run --filter nanogpt typecheck`: passed.
- `bun run lint`: passed.
- `bun run check:tensor-lifetimes`: passed.
- `bun run check:runtime-review`: passed.
- `bun run validate`: passed.

The focused success test still runs the `reshape-transpose` scenario under the
benchmark path and emits the same JSON result fields.

## Independent Review

Hypatia performed a read-only second-opinion review for this tranche. The review
confirmed that parse/help/errors now happen before the runtime lock, usage and
runtime errors use structured stdout with distinct exit codes, lock conflicts
remain runtime failures, default success output and `--json` output are
preserved, and the focused tests cover the intended boundary. The review
recommended keeping this tranche narrow and not widening it into a broader TOON
success-output rewrite.

## Remaining Risks / Follow-ups

The benchmark's default success output remains compact key/value text rather
than full TOON. That preserves the existing operator-facing contract. A later
CLI-only tranche can standardize benchmark success summaries across nanoGPT and
the package benchmark scripts.

## Out-of-scope drift noticed

The supervised-run soak and acceptance wrappers are separate heavy MLX command
surfaces. They were not changed in this tranche.
