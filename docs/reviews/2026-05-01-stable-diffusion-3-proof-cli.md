# Runtime Review: Stable Diffusion 3 Proof CLI

## Summary

Added the `examples/stable-diffusion-3` finite proof command. The command resolves
local or Hub Diffusers snapshots, validates the `StableDiffusion3Pipeline`
manifest, loads FlowMatch Euler, the SD3 transformer, the SD3 VAE, and the
example-owned CLIP/T5 prompt conditioner, then writes BMP artifact evidence
through the shared image-proof verifier path.

No official gated checkpoint evidence or throughput claim is made in this
tranche.

## Files Reviewed

- `examples/stable-diffusion-3/index.ts`
- `examples/stable-diffusion-3/index.test.ts`
- `examples/stable-diffusion-3/image-output.ts`
- `examples/stable-diffusion-3/image-output.test.ts`
- `examples/stable-diffusion-3/conditioning-runtime.ts`
- `examples/stable-diffusion-3/README.md`
- `package.json`
- `tsconfig.phase10-examples.json`

## Reference Audit

- Local Diffusers `pipeline_stable_diffusion_3.py` defaults the text-to-image
  path to 28 inference steps, guidance scale `7.0`, T5 max sequence length 256,
  and default image dimensions from transformer sample size times VAE scale.
- The same Diffusers pipeline validates image dimensions against
  `vae_scale_factor * patch_size`, uses FlowMatch Euler, and treats CFG as active
  only when guidance scale is greater than one.
- The command keeps skip-layer guidance, IP-Adapter, ControlNet, img2img,
  inpainting, and PAG out of scope because their package runtime contracts do
  not exist yet.

## Tensor Lifetime Audit

The command owns only application-layer orchestration. Loaded transformer, VAE,
prompt conditioner, prompt-conditioning result, RNG key, and generated image
tensors are held with `using` declarations. The BMP writer consumes the final
image without taking ownership.

## Memory / Performance Evidence

The command acquires the shared runtime command lock before model loading or
generation. No performance claim is made. The focused tests cover help, parsing,
usage errors before lock acquisition, structured stdout/stderr separation,
runtime error formatting, output validation, and BMP artifact evidence.

## Coverage

`bun test examples/stable-diffusion-3` passed with 15 tests and 124 assertions
after implementation.

## Independent Review

Boole the 2nd previously reviewed the SD3 prompt-conditioning tranche and
confirmed the example-owned application boundary. This command follows that
boundary and does not add `@mlxts/diffusion` dependencies on transformers or
tokenizers.

## Out-of-scope Drift Noticed

None.

## Remaining Risks / Follow-ups

Official `stabilityai/*` SD3 and SD3.5 image proof still requires authenticated
access to gated checkpoints. Larger/default-step characterization, skip-layer
guidance, IP-Adapter, ControlNet, img2img, inpainting, PAG, LoRA, and throughput
work remain separate tranches.
