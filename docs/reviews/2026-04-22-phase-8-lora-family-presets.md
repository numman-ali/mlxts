# Runtime Review: Phase 8 LoRA Family Presets, PEFT Interop, and Example Surfaces

## Summary

This slice turns LoRA from a single-family proof path into a readable
cross-family surface without adding a new training framework.

The key changes are:

1. `@mlxts/transformers` now owns readable LoRA target presets for causal
   decoder families.
2. Transformer-owned adapter save/load now supports strict first-pass PEFT
   causal-LM interoperability alongside the native mlxts format.
3. The proof and example surfaces now compose around those shared presets
   instead of hardcoded module names.

The architectural goal stayed the same throughout: keep family truth and
checkpoint path translation in transformers, keep generic adapter mechanics in
`@mlxts/lora`, and keep the user story visible in examples rather than hiding
it behind a new pipeline package.

## Files Reviewed

- `packages/transformers/src/index.ts`
- `packages/transformers/src/lora-module-traversal.ts`
- `packages/transformers/src/lora-targets.ts`
- `packages/transformers/src/lora-adapters.ts`
- `examples/train-proof/index.ts`
- `examples/lora-finetune/index.ts`

## Tensor Lifetime Audit

`resolveLoRATargets()` and the traversal helper are module-tree inspection only.
They do not allocate or retain native tensors.

`saveCausalLMAdapters()` reads active adapter wrapper state and either delegates
to the existing mlxts-native save path or writes a PEFT-shaped safetensors
bundle from already-owned LoRA parameters. No new native ownership protocol was
introduced there.

`loadCausalLMAdapters()` keeps ownership explicit:

- native mlxts adapter load still delegates to the existing `@mlxts/lora`
  loader
- PEFT load translates tensor names into the existing module-tree adapter
  layout, applies wrappers first, then updates the module tree with translated
  tensors
- if PEFT load fails after tensors are read, the loaded tensors are freed in
  the catch path before rethrowing

The proof and example surfaces reuse the existing training, generation, and
merge/remove flows. They did not add new hidden tensor-retention behavior; they
only changed how LoRA targets are resolved and how adapters are serialized.

## Memory / Performance Evidence

This patch does not change generation hot-path math, cache handling, or runtime
strategy under `packages/transformers/src/families/` or
`packages/transformers/src/infrastructure/`. The main performance-sensitive
question here is operational correctness: can the new surfaces train, save,
reload, merge, and sample without regressing behavior?

Focused validation and live smoke evidence:

- structural family preset coverage:
  - `bun test packages/transformers/src/lora.test.ts`
  - covers `llama`, `mistral`, `gemma`, `phi3`, `mistral3`, `gemma3_text`,
    `gemma4_text`, and `gemma4`
- adapter/training foundation coverage:
  - `bun test packages/lora/src/apply-module.test.ts`
  - `bun test packages/lora/src/io.test.ts`
  - `bun test packages/nn/src/lora-linear.test.ts`
  - `bun test packages/align/src/sft.test.ts`
  - `bun test packages/align/src/dpo.test.ts`
  - `bun test examples/train-proof/helpers.test.ts`
- workspace typing:
  - `bun run typecheck`

Real end-to-end example runs on cached official checkpoints:

- dense LoRA example, native adapter format:
  - command:
    `bun run example:lora-finetune --source meta-llama/Llama-3.2-1B-Instruct --dataset-source tiny --train-limit 8 --eval-limit 4 --batch-size 2 --steps 1 --report .tmp/lora-finetune-smoke-report.json --output-dir .tmp/lora-finetune-smoke --adapter-format mlxts`
  - result:
    - `eval_loss_before=5.9142`
    - `eval_loss_after=5.9077`
    - `target_count=8`
    - trained / reloaded / merged sample text matched
- dense LoRA example on a second family, native adapter format:
  - command:
    `bun run example:lora-finetune --source google/gemma-3-1b-it --dataset-source tiny --train-limit 8 --eval-limit 4 --batch-size 2 --steps 1 --report .tmp/lora-finetune-gemma3-smoke-report.json --output-dir .tmp/lora-finetune-gemma3-smoke --adapter-format mlxts`
  - result:
    - `eval_loss_before=7.3211`
    - `eval_loss_after=7.3125`
    - `target_count=8`
    - trained / reloaded / merged sample text matched
- dense LoRA example, PEFT adapter format:
  - command:
    `bun run example:lora-finetune --source meta-llama/Llama-3.2-1B-Instruct --dataset-source tiny --train-limit 8 --eval-limit 4 --batch-size 2 --steps 1 --report .tmp/lora-finetune-peft-smoke-report.json --output-dir .tmp/lora-finetune-peft-smoke --adapter-format peft`
  - result:
    - `eval_loss_before=5.9142`
    - `eval_loss_after=5.9077`
    - `target_count=8`
    - trained / reloaded / merged sample text matched

Interpretation:

- target preset resolution is working across the current supported causal LM
  family set
- the end-to-end example remains readable while proving the actual train →
  save → reload → merge loop
- first-pass PEFT interop is not just unit-correct; it survives the full example
  workflow too

## Independent Review

Two focused high-capability reviews informed the implementation boundaries
before the final code landed.

Family-target review:

- standard decoder families share the same effective `attention` and
  `attention+mlp` target shapes
- Phi needs packed `qkvProjection` and `gateUpProjection` handling
- Gemma 4 is the only current family with extra decoder-stack linear targets
  outside ordinary attention and MLP
- `lastLayers` needs to slice layer-local targets without dropping the Gemma 4
  decoder-root auxiliary projection

PEFT interoperability review:

- first pass should support only standard single-adapter causal LM LoRA
- adapter config should be strict and reject `modules_to_save`, DoRA, RSLoRA,
  rank-pattern overrides, embedding payloads, and other PEFT features we do not
  actually implement yet
- Hugging Face naming should be translated family-by-family at the transformer
  ownership layer rather than pushed down into generic LoRA traversal code

The final implementation follows those recommendations directly.

## Remaining Risks / Follow-ups

- PEFT support is intentionally narrow. It does not yet support RSLoRA, DoRA,
  `modules_to_save`, multi-adapter bundles, embedding adapters, or non-causal-LM
  tasks.
- The proof matrix runner exists, but the full official-model matrix has not yet
  been run as a single long acceptance sweep in this patch. The structural test
  matrix plus the two live example smokes were the lighter-weight validation
  layer for landing the implementation safely first.
- `examples/lora-finetune/` is the primary readable end-to-end teaching surface
  for this phase. If we later discover a genuinely shared orchestration seam
  across LoRA, SFT, DPO, serving, and multimodal flows, that should be earned by
  reuse rather than introduced pre-emptively as a new pipeline framework now.
