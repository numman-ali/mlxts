# Runtime Review: Stable Diffusion 3 Prompt Conditioning

## Summary

Added the application-owned Stable Diffusion 3 / 3.5 prompt-conditioning bridge under `examples/stable-diffusion-3`. The tranche composes two projected CLIP text encoders plus one T5 encoder into `StableDiffusion3Conditioning` without changing `@mlxts/diffusion` package boundaries.

The new workbook mirrors Diffusers prompt semantics: CLIP penultimate hidden states, optional positive `clipSkip`, projected CLIP `textEmbeds` for pooled projections, T5 sequence embeddings padded to the transformer joint-attention dimension, and negative conditioning only when classifier-free guidance is active.

## Files Reviewed

- `examples/stable-diffusion-3/conditioning-runtime.ts`
- `examples/stable-diffusion-3/conditioning.ts`
- `examples/stable-diffusion-3/conditioning-types.ts`
- `examples/stable-diffusion-3/conditioning-result.ts`
- `examples/stable-diffusion-3/conditioning.test.ts`
- `examples/stable-diffusion-3/AGENTS.md`
- `examples/stable-diffusion-3/README.md`
- `package.json`
- `tsconfig.phase10-examples.json`

## Reference Audit

- `.reference/diffusers/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py` confirms the SD3 `encode_prompt` contract: two CLIP projection branches, one T5 branch, CLIP hidden concat, CLIP-to-T5 hidden padding, sequence-axis concat with T5, and pooled projection concat.
- The same file confirms `max_sequence_length` defaults to `256`, rejects values above `512`, applies positive `clip_skip`, and uses default penultimate CLIP hidden states for negatives.
- Existing local examples were used for package-boundary shape: `examples/stable-diffusion` for CLIP hidden-state and projection handling, `examples/flux` for CLIP+T5 loading, and `examples/flux2` for disposable application-owned conditioner wrappers.

## Tensor Lifetime Audit

Returned conditioning tensors are owned by `StableDiffusion3PromptConditioningResult`. Intermediate CLIP hidden states are retained before encoder outputs are disposed. Projected `textEmbeds` are transferred into the returned pooled projections or freed on failure. T5 last hidden states are transferred into the returned encoder hidden states or freed on failure. Unreturned hidden states, pooled outputs, and token-id arrays are disposed in all branches.

## Memory / Performance Evidence

No real checkpoint generation benchmark was run. The change is an example-owned tensor composition step and does not alter package denoising hot paths. The composition keeps branch tensors explicit and avoids hidden package-level tokenizer or text-encoder dependencies.

## Independent Review

Boole the 2nd (`019de3c5-27c1-7e42-8b6a-044ed0bc04a3`) performed a read-only audit of Diffusers SD3 prompt conditioning and local example patterns. The review confirmed that the bridge belongs in an example, not `@mlxts/diffusion`, and called out the critical correctness traps: use projected CLIP `textEmbeds`, pad CLIP hidden states to the T5 joint dimension, ignore negative prompts unless CFG is active, and keep negative CLIP encoding on the default penultimate hidden state.

## Coverage

Focused tests cover:

- Positive and negative CFG conditioning across two CLIP branches plus one T5 branch.
- Positive `clipSkip` while negative conditioning stays on the default hidden-state selection.
- `prompt_2` / `prompt_3` fallback and distinct prompt routing.
- `numImagesPerPrompt` repetition.
- CLIP-to-T5 padding and pooled projection concatenation.
- Prompt batch mismatch, max sequence bounds, invalid `clipSkip`, hidden-size mismatch, pooled-size mismatch, and disposed-conditioner errors.

## Out-of-scope Drift Noticed

None.

## Remaining Risks / Follow-ups

Official SD3 / SD3.5 real image proof still requires authenticated Stability checkpoints. T5 tokenizer loading may need a fallback if a target snapshot lacks `tokenizer.model` or `spiece.model`. Direct `prompt_embeds` APIs, LoRA scaling, IP-Adapter, ControlNet, skip-layer guidance, img2img, inpainting, and PAG variants remain separate tranches.
