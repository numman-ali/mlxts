# Transformers LoRA Folder Consolidation Review

## Summary

Moved the transformer-owned CausalLM LoRA helpers from loose top-level
`lora-*` files into `packages/transformers/src/lora/` without changing adapter
I/O, traversal, target resolution, or PEFT naming behavior.

## Files Reviewed

- `packages/transformers/src/lora-adapters.ts`
- `packages/transformers/src/lora-module-traversal.ts`
- `packages/transformers/src/lora-targets.ts`
- `packages/transformers/src/lora/adapters.ts`
- `packages/transformers/src/lora/module-traversal.ts`
- `packages/transformers/src/lora/targets.ts`
- `packages/transformers/src/index.ts`

## Tensor Lifetime Audit

No tensor-producing expressions, adapter tensor ownership, safetensors loading
or saving behavior, module traversal logic, or LoRA wrapper state handling were
changed intentionally. The changed production lines are import/export path
updates caused by the file move.

## Memory / Performance Evidence

This tranche makes no performance claim. `bench:generation` and
`bench:generation:parity` were not run because the diff does not touch
generation hot paths or model execution logic.

## Independent Review

The audit already identified this as a folder-discipline move: transformer
LoRA stays responsible for CausalLM-specific PEFT I/O and target naming, while
`@mlxts/lora` keeps generic Module-level adapter mechanics. I checked the moved
files against that boundary and left the generic traversal consolidation for
the later helper tranche called out in the audit.

## Remaining Risks / Follow-ups

`packages/transformers/src/lora/module-traversal.ts` still partially overlaps
with `@mlxts/lora/src/traversal.ts`. That consolidation is deliberately out of
scope for this tranche because the audit only asks for the folder move here.
