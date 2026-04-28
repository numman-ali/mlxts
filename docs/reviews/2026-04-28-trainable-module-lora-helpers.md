# Runtime Review: Trainable Module And QLoRA Helper Extraction

## Summary

Tranche 11 moved the example-local CausalLM-to-Module narrowing into
`@mlxts/transformers` as `expectTrainableModule`, without changing the
`CausalLM` contract. It also moved the QLoRA merged-base invariant into
`@mlxts/lora` as `assertQuantizedBasePreserved`, replacing the training proof's
private `Reflect.get` path walk with package-owned module traversal.

## Files Reviewed

- packages/transformers/src/index.ts
- packages/transformers/src/lora/module-traversal.ts

## Tensor Lifetime Audit

The changed transformer files only expose and delegate module-tree inspection;
they do not allocate tensors, call MLX ops, trigger `mx.eval`, or change native
handle ownership. The new lora helper reuses existing module traversal and only
checks the merged target module identity.

## Memory / Performance Evidence

No generation hot path changed. Focused validation completed with
`bun test packages/lora/src/apply-module.test.ts`, `bun run --filter
'@mlxts/lora' typecheck`, `bun run --filter '@mlxts/transformers' typecheck`,
`bun run typecheck`, and `bun run lint`. The first full `bun run validate`
reached `check:runtime-review` and stopped only because this review artifact was
missing.

## Independent Review

Wegener, a GPT-5.5 xhigh explorer sub-agent, reviewed the tranche 11 diff for
contract leaks, missed exports, brittle path assumptions, and capability or
performance regressions. Wegener identified a hardcoded LLaMA q-projection path
in the first extraction pass; the implementation was updated to assert the
actual merged QLoRA target paths instead.

## Remaining Risks / Follow-ups

The QLoRA proof now checks merged target paths rather than assuming a single
family-shaped projection name. No additional follow-up is required for this
tranche.
