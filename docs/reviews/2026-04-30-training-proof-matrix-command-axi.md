# Training Proof Matrix Command AXI Boundary

## Summary

The training proof matrix wrapper now behaves as a finite agent-facing command.
It parses help and matrix-owned usage errors before spawning child proofs,
routes child proof stdout and stderr to matrix stderr as progress, and reserves
matrix stdout for compact structured help, success, and error output.

## Files Reviewed

- `package.json`
- `examples/train-proof/matrix.ts`
- `examples/train-proof/matrix.test.ts`
- `examples/train-proof/README.md`

## Runtime Sensitivity

The change is a command-boundary migration around sequential child proof runs.
It does not change the canonical `proof:training` command, stage behavior,
training math, dataset preparation, adapter handling, report verification, or
the child proof report schema. It changes how the matrix wrapper exposes child
output: child stdout and stderr are progress evidence on matrix stderr, while
matrix stdout contains one summary.

## Tensor Lifetime Audit

No tensor-producing operation or retained tensor owner is introduced. The matrix
wrapper spawns child proof processes and formats their planned artifact paths.
Model and tensor ownership remain inside each child `proof:training` run.

## Memory / Performance Evidence

No performance claim is made. Help and usage errors now return before child
processes start. Successful matrix runs still execute the same child proof
commands sequentially.

## Tests

- `bun test examples/train-proof/matrix.test.ts`
- `bun run check:training-proofs`
- `bun run examples/train-proof/matrix.ts --help`
- `bun run examples/train-proof/matrix.ts --source --dataset-source`

## Independent Review

Locke performed a read-only second opinion and recommended the same narrow
shape: keep the child `proof:training` contract unchanged, make repeated
`--source` the only matrix-owned option, pass all other flags through, capture
child stdout and stderr as progress, and emit one structured matrix summary on
stdout.

## Out-of-scope Drift Noticed

No additional Phase 8 training behavior drift was changed in this tranche.
Long-running official-model matrix proof remains an operator-triggered smoke,
not a cheap validation gate.

## Remaining Risks / Follow-ups

Any external consumer that scraped child `training_proof:` summaries from matrix
stdout must now read the matrix `runs[]` report paths or run `proof:training`
directly. This is the intended AXI boundary: matrix stdout is one command
result, while child logs are progress.
