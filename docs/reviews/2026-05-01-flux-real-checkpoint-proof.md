# FLUX Real Checkpoint Proof

## Summary

`black-forest-labs/FLUX.1-schnell` now has a bounded real checkpoint image
proof through `examples/flux`. The proof exposed that FLUX.1's VAE config sets
`use_quant_conv=false` and `use_post_quant_conv=false`; the shared
AutoencoderKL now honors those Diffusers config flags while Stable Diffusion
continues to require both projections.

## Files Reviewed

- `packages/diffusion/src/families/stable-diffusion/autoencoder.ts`
- `packages/diffusion/src/families/stable-diffusion/config.ts`
- `packages/diffusion/src/families/flux/autoencoder.ts`
- `packages/diffusion/src/families/flux/config.ts`
- `examples/flux/README.md`

## Reference Audit

- The cached FLUX.1-schnell VAE config at revision
  `741f7c3ce8b383c54771c7003378a50191e9efe9` declares
  `use_quant_conv: false` and `use_post_quant_conv: false`.
- The same config keeps standard `AutoencoderKL` shape fields:
  `latent_channels: 16`, `block_out_channels: [128, 256, 512, 512]`,
  `scaling_factor: 0.3611`, and `shift_factor: 0.1159`.
- The checkpoint's VAE safetensors do not contain `quant_conv.*` or
  `post_quant_conv.*`, so requiring those parameters is not Diffusers-parity
  behavior for this model.

## Implementation Notes

- `StableDiffusionAutoencoderConfig` now carries `useQuantConv` and
  `usePostQuantConv`.
- Stable Diffusion config parsing still rejects disabled quant projections and
  sets both booleans to `true`.
- FLUX config parsing accepts the Diffusers booleans, defaulting to `true` only
  when the fields are absent.
- Autoencoder parameter scanning omits disabled projections because the module
  fields are `null`, not enumerable child modules.
- Decode bypasses `postQuantConv` when disabled; encode transfers the encoder
  output directly when `quantConv` is disabled.

## Tensor Lifetime Audit

- `encodeMoments()` keeps the encoded tensor in a local nullable owner, returns
  it only after transferring ownership, and frees it on all projection paths.
- `decode()` keeps the existing `using projected` lifetime when `postQuantConv`
  exists and sends caller-owned latents directly into the decoder when it does
  not.
- No disposable tensor handles are hidden inside nested tensor-producing calls.

## Memory / Performance Evidence

- This tranche removes nonexistent VAE projection modules for FLUX.1-schnell
  instead of adding work to the hot denoise loop.
- Stable Diffusion checkpoint configs still construct the same quant and
  post-quant Conv2d modules, so existing Stable Diffusion encode/decode paths
  keep their prior projection behavior.
- The proof run is a capability check, not a benchmark; no performance claim is
  made from its elapsed time.

## Proof Evidence

Initial proof run downloaded the Hub snapshot into `.tmp/hf-diffusion-proof-cache`
and failed at VAE loading with:

```text
loadDiffusionWeights: checkpoint is missing required parameters: postQuantConv.bias, postQuantConv.weight, quantConv.bias, quantConv.weight.
```

After the fix, the local-only proof passed:

```bash
bun run examples/flux/index.ts black-forest-labs/FLUX.1-schnell \
  --cache-dir .tmp/hf-diffusion-proof-cache \
  --local-files-only \
  --prompt "a small red apple on a white table, product photo" \
  --output .tmp/flux/flux1-schnell-proof.bmp \
  --steps 2 \
  --height 256 \
  --width 256 \
  --max-sequence-length 128 \
  --seed 7 \
  --dtype float16 \
  --json
```

Result:

- Resolved revision: `741f7c3ce8b383c54771c7003378a50191e9efe9`
- Snapshot size: `33,725,923,002` bytes across `23` files
- Output: `.tmp/flux/flux1-schnell-proof.bmp`
- Output bytes: `196,662`
- Image size: `256x256`
- Denoise steps: `2`
- Guidance scale: `null`
- Prompt truncated: `false`
- Prompt 2 truncated: `false`

## Validation

- `bun test packages/diffusion/src/families/stable-diffusion/autoencoder.test.ts packages/diffusion/src/families/stable-diffusion/weights.test.ts packages/diffusion/src/families/stable-diffusion/config.test.ts packages/diffusion/src/families/flux/autoencoder.test.ts packages/diffusion/src/families/flux/config.test.ts packages/diffusion/src/families/z-image/config.test.ts`
- Real FLUX.1-schnell local-only proof command above.
- `bun run check:tensor-lifetimes`
- `bun run check:runtime-review`
- `bun run validate`

## Independent Review

Second-opinion explorer review found two pre-commit issues: the initial edit
removed the `add` import still used by `StableDiffusionVaePosterior`, and the
review artifact initially missed the required `Memory / Performance Evidence`
heading. Both were fixed before commit. The reviewer found no issue with the
optional quant-projection config flow or the `encodeMoments()` / `decode()`
ownership shape.

## Out-of-scope Drift Noticed

- FLUX dev-style guidance checkpoints remain supported only by the existing
  guidance-embedding path; this proof does not claim dev checkpoint licensing
  or quality parity.
- Z-Image and Qwen-Image runtime tensor execution remain separate Phase 10
  tranches.

## Remaining Risks / Follow-ups

- Full FLUX image quality parity against Diffusers is still a separate
  numerical/visual parity tranche.
- The proof uses a bounded two-step 256x256 run, not a performance benchmark.
