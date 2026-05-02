# Runtime Review: LTX-2 Proof Assembly

## Summary

This tranche adds the LTX-2 example proof path that joins Gemma 3 all-layer prompt conditioning, LTX-2 text connectors, paired video/audio denoising, video VAE decode, audio VAE decode, and vocoder WAV output. The production transformer change is bounded to Gemma 3 hidden-state extraction and tokenizer attention masks; `CausalLM` remains unchanged.

The reference audit used Diffusers `src/diffusers/pipelines/ltx2/pipeline_ltx2.py`, the live Lightricks/LTX-2 `model_index.json`, and the live Lightricks/LTX-2 `text_encoder/config.json` / safetensors index shape to confirm the `gemma3` top-level wrapper and `language_model.*` text weight prefix.

## Files Reviewed

- `packages/transformers/src/families/gemma3/config.ts`
- `packages/transformers/src/families/gemma3/model.ts`
- `packages/transformers/src/families/gemma3/weights.ts`
- `packages/transformers/src/index.ts`
- `packages/transformers/src/infrastructure/masks.ts`

## Tensor Lifetime Audit

Gemma 3 hidden-state extraction retains each exposed hidden state explicitly with `retainArray`, returns a disposable model output, and frees hidden states on error. Tokenizer attention masks are owned per shared mask variant and released through a de-duplicating mask release path so full/sliding layers can share mask tensors without double-free.

LTX-2 example conditioning disposes token id tensors, tokenizer masks, Gemma outputs, connector outputs, denoised video/audio latents, decoded media tensors, and waveform tensors at the owning call boundary. The audio WAV writer evaluates the final waveform once before copying PCM samples to host bytes.

`bun run check:tensor-lifetimes` passed with no suspicious nested tensor-producing calls.

## Memory / Performance Evidence

Targeted checks run:

- `bun test examples/ltx-video packages/transformers/src/families/config.test.ts packages/transformers/src/families/gemma3/model.test.ts packages/transformers/src/infrastructure/masks.test.ts`
- `bun run typecheck`
- `bun run check:tensor-lifetimes`

The changed Gemma 3 path does not change the existing generation `forward()` call shape or steady-state cache path. The new tokenizer attention mask path is used by LTX-2 prompt conditioning through `runWithHiddenStates`, not normal decode.

`bench:generation` and `bench:generation:parity` were not run for this tranche because the implementation does not alter decode scheduling, cache update semantics, sampling, or logits projection. They remain the required evidence for future steady-state generation hot-path changes.

## Independent Review

Tesla the 2nd performed an independent read-only pass before implementation. The review identified the same narrow tranche boundary: add Gemma 3 hidden-state and padding-mask support, keep `CausalLM` unchanged, assemble LTX-2 in the example, and leave STG, modality isolation, guidance rescale, prompt enhancement, and LTX-2.3-specific prompt modulation out of scope.

## Remaining Risks / Follow-ups

Real-checkpoint LTX-2 proof execution still needs an operator run with a local LTX-2 snapshot and the shared runtime lock. The code path now supports the proof assembly, but this commit validates it with focused fixtures and compile-time gates rather than a full 12B text encoder plus video/audio denoise run.

LTX-2.3-only branches, prompt enhancement, STG, modality isolation guidance, and guidance rescale remain deliberately unsupported until their reference behavior is implemented as separate runtime tranches.
