# Runtime Review: Phase 8 Training Loss Traces

## Summary

Phase 8 recipe runners now return per-step training loss traces alongside the
existing average-loss scalar. The canonical training proof and readable LoRA
example include those traces in JSON reports, and their verifiers require one
loss entry per configured optimizer step plus average-loss consistency.

This tranche does not change optimizer settings, batch selection, model
forwards, adapter application, quantized-base checks, DPO reward metrics, or
sample generation. It records evidence already produced by the existing
training-step calls.

## Files Reviewed

- `packages/align/src/recipes.ts`
- `packages/align/src/index.ts`
- `packages/align/src/recipes.test.ts`
- `examples/train-proof/runtime.ts`
- `examples/train-proof/stages.ts`
- `examples/train-proof/types.ts`
- `examples/train-proof/report-schema.ts`
- `examples/train-proof/verification.ts`
- `examples/train-proof/verification.test.ts`
- `examples/train-proof/verify-report.test.ts`
- `examples/train-proof/cli.test.ts`
- `examples/lora-finetune/runtime.ts`
- `examples/lora-finetune/workflow.ts`
- `examples/lora-finetune/args.ts`
- `examples/lora-finetune/verification.ts`
- `examples/lora-finetune/verify-report.test.ts`
- `examples/lora-finetune/cli.test.ts`

## Tensor Lifetime Audit

`runSupervisionTrainingSteps` and `runPreferenceTrainingSteps` still call
`sftTrain` and `dpoTrain` once per configured step. The new trace stores the
returned numeric average loss for each step and does not retain tensors,
gradients, batches, or model-owned arrays.

The proof and verifier changes are JSON/report plumbing over numbers and do not
introduce MLX operations.

## Memory / Performance Evidence

The runtime path performs no extra loss recomputation and no extra model
forward. The only added work is pushing one small `{ step, loss }` record per
optimizer step and checking the trace during report verification.

Focused validation passed:

```bash
bun test packages/align/src/recipes.test.ts examples/train-proof/verification.test.ts examples/train-proof/verify-report.test.ts examples/train-proof/cli.test.ts examples/lora-finetune/verify-report.test.ts examples/lora-finetune/cli.test.ts
bun run check:training-proofs
bun run typecheck
bun run lint
bun run check:assertions
bun run check:tensor-lifetimes
bun run check:runtime-review
bun run validate
```

## Independent Review

Bacon reviewed the tranche before finalization. The review confirmed that
`@mlxts/align` recipes are the right layer for the trace, that `@mlxts/train`
should remain untouched, and that verifiers should require shape and mean
consistency without requiring monotonic loss.

## Out-of-Scope Drift Noticed

- Full official-model Phase 8 proof reruns are still manual Apple Silicon
  acceptance evidence, not an every-commit CI gate.
- Higher-level training CLI ergonomics remain future product work after the
  proof surfaces stay stable.

## Remaining Risks / Follow-ups

The trace proves transparency of the executed optimizer steps. It does not
claim convergence quality by itself, and small-batch loss can legitimately
increase between individual steps.
