# Stable Diffusion 3 Real Checkpoint Proof Blocked

## Summary

Attempted the bounded official Stable Diffusion 3.5 Medium checkpoint proof after
the finite proof command landed. The checkpoint is visible through the Hub API,
but the configured token is not authorized for the gated Stability AI repo, so
the proof is blocked before snapshot download completes.

This is an access blocker, not a loader, scheduler, conditioning, or denoising
failure.

## Files Reviewed

- `examples/stable-diffusion-3/index.ts`
- `examples/stable-diffusion-3/conditioning.ts`
- `packages/diffusion/src/pretrained/snapshot-source.ts`

## Attempted Command

```bash
bun run examples/stable-diffusion-3/index.ts stabilityai/stable-diffusion-3.5-medium \
  --variant fp16 \
  --prompt "a small ceramic robot holding a red apple on a wooden desk" \
  --negative-prompt "blurry, low quality" \
  --output .tmp/stable-diffusion-3/sd35-medium-proof.bmp \
  --steps 2 \
  --height 256 \
  --width 256 \
  --max-sequence-length 128 \
  --dtype bfloat16 \
  --json
```

## Result

The Hub API returned HTTP access denial during `paths-info` resolution:

```text
Access to model stabilityai/stable-diffusion-3.5-medium is restricted and you are not in the authorized list.
```

The command failed before any selected safetensor file downloaded or any MLX
runtime component loaded.

## Follow-up

Rerun the same command after the local Hugging Face token is accepted for
`stabilityai/stable-diffusion-3.5-medium`, or provide a local Diffusers SD3/SD3.5
snapshot directory with equivalent component files.
