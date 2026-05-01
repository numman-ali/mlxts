# Runtime Review: Qwen-Image 2512 Real Checkpoint Proof

## Summary

Official `Qwen/Qwen-Image-2512` now has a bounded real checkpoint proof through
`examples/qwen-image`. The proof resolved the current Diffusers
`QwenImagePipeline` snapshot, loaded the Qwen2.5-VL text encoder, Qwen2
tokenizer sidecars, Qwen-Image transformer, FlowMatch Euler scheduler, and
Qwen-specific 3D causal VAE, encoded positive and single-space negative prompt
conditioning, ran two true-CFG denoising steps, and wrote a 256x256 BMP
artifact.

This is a product capability proof, not a throughput or image-quality
benchmark.

## Files Reviewed

- `packages/tokenizers/src/bpe/vocab-merges.ts`
- `packages/tokenizers/src/bpe/bpe-load.ts`
- `packages/tokenizers/src/bpe/bpe.ts`
- `packages/tokenizers/src/index.ts`
- `packages/tokenizers/src/load.ts`
- `packages/transformers/src/load.ts`
- `packages/transformers/src/pretrained/snapshot-inspection.ts`
- `packages/transformers/src/pretrained/snapshot-supported-files.ts`
- `packages/transformers/src/pretrained/types.ts`
- `packages/diffusion/src/families/qwen-image/pipeline.ts`
- `packages/diffusion/src/families/qwen-image/transformer.ts`
- `packages/diffusion/src/families/qwen-image/autoencoder.ts`
- `packages/diffusion/src/families/qwen-image/weights.ts`
- `examples/qwen-image/index.ts`
- `examples/qwen-image/conditioning.ts`
- `examples/qwen-image/conditioning-runtime.ts`
- `examples/qwen-image/image-output.ts`

## Snapshot Evidence

The resolved snapshot is:

```text
Qwen/Qwen-Image-2512@25468b98e3276ca6700de15c6628e51b7de54a26
```

`model_index.json` declares:

- pipeline: `QwenImagePipeline`
- scheduler: `FlowMatchEulerDiscreteScheduler`
- text encoder: `Qwen2_5_VLForConditionalGeneration`
- tokenizer: `Qwen2Tokenizer`
- transformer: `QwenImageTransformer2DModel`
- VAE: `AutoencoderKLQwenImage`

The selected Diffusers snapshot contained 28 files and `57,704,574,910` bytes.
The transformer config used `num_layers=60`, `num_attention_heads=24`,
`attention_head_dim=128`, `joint_attention_dim=3584`,
`axes_dims_rope=[16,56,56]`, and `in_channels=64`. The scheduler config used
`shift=1.0`, `shift_terminal=0.02`, and `num_train_timesteps=1000`.

## Tokenizer Fix

The first proof attempt found a product bug before denoising: the Qwen-Image
tokenizer directory has no `tokenizer.json`, declares `Qwen2Tokenizer`, and
ships `vocab.json`, `merges.txt`, `added_tokens.json`, and
`added_tokens_decoder`. Auto-detection fell through to CLIP vocab/merges loading
and failed because Qwen does not define CLIP's `<|startoftext|>` BOS token.

`@mlxts/tokenizers` now has a generic byte-level vocab/merges loader for
GPT/Qwen-style tokenizers. The automatic loader keeps CLIP-specific behavior on
`CLIPTokenizer` / `CLIPTokenizerFast`, routes `Qwen2Tokenizer` /
`Qwen2TokenizerFast` to the generic byte-level path, preserves added token IDs,
uses `added_tokens_decoder.special` for true special-token classification, and
uses the Qwen2 pre-tokenization regex from the Hugging Face reference.

## Tensor Lifetime Audit

The tokenizer fix does not create tensor resources. The proof exercised the
existing Qwen-Image runtime path with explicit ownership boundaries around text
conditioning outputs, generated latents, transformer outputs, VAE outputs, and
the final image tensor. The proof command holds the shared runtime lock for the
full model-load and denoising run.

## Memory / Performance Evidence

The proof loaded a selected snapshot of `57,704,574,910` bytes from the local
cache after the initial download. The proof was intentionally bounded to two
256x256 denoising steps and completed in `69,035.96 ms` from the cached
snapshot. No throughput or default-step performance claim is made.

## Validation

Focused tokenizer and Qwen-Image tests:

```bash
bun test packages/tokenizers/src/load.test.ts packages/tokenizers/src/coverage.test.ts packages/transformers/src/load.test.ts examples/qwen-image/conditioning.test.ts examples/qwen-image/index.test.ts
```

Result: 66 pass, 0 fail.

Focused typecheck:

```bash
bun run --filter @mlxts/tokenizers typecheck
bun run --filter @mlxts/transformers typecheck
```

Result: both passed.

Real checkpoint proof command:

```bash
bun run examples/qwen-image/index.ts Qwen/Qwen-Image-2512 \
  --cache-dir .tmp/hf-diffusion-proof-cache \
  --local-files-only \
  --prompt "a small red apple on a white table, product photo" \
  --negative-prompt " " \
  --output .tmp/qwen-image/qwen-image-2512-official-proof.bmp \
  --steps 2 \
  --height 256 \
  --width 256 \
  --true-cfg-scale 4 \
  --seed 7 \
  --dtype bfloat16 \
  --json
```

Result:

- resolved revision:
  `25468b98e3276ca6700de15c6628e51b7de54a26`
- output: `.tmp/qwen-image/qwen-image-2512-official-proof.bmp`
- output bytes: `196,662`
- image size: `256x256`
- prompt truncated: `false`
- negative prompt truncated: `false`
- elapsed: `69,035.96 ms` from cached snapshot

The BMP artifact was verified as a `256 x 256 x 24` Windows BMP. A temporary PNG
conversion opened successfully for visual inspection.

## Independent Review

Galileo performed a read-only tokenizer/load-path review and recommended the
Qwen2-gated byte-level vocab/merges loader instead of changing CLIP fallback
globally. The review also called out `added_tokens.json` propagation,
`added_tokens_decoder.special`, and Qwen2 pre-tokenization parity; this tranche
implements those follow-ups.

## Remaining Risks / Follow-ups

- This proof is bounded to `256x256`, two denoising steps, batch size 1, and
  true CFG scale `4`.
- It is not a throughput benchmark and does not claim 1024px/default-step image
  quality.
- Qwen-Image edit, ControlNet, image-to-image, inpainting, layered composition,
  multi-batch, and quantized sidecars remain unsupported until their runtime
  semantics are designed deliberately.
- FLUX.2 Klein and Stable Diffusion 3 / 3.5 remain separate later families.
