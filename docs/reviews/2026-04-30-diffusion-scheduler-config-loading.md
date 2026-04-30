# Diffusion Scheduler Config Loading

## Summary

Added the first local checkpoint metadata boundary for `@mlxts/diffusion`.
The package now parses Diffusers-style `scheduler/scheduler_config.json`
payloads, supports explicit scheduler subfolders, instantiates only the
supported DDIM and Euler scheduler classes, and rejects checkpoint knobs whose
scheduler math is not implemented yet.

This tranche does not add UNet, VAE, text conditioning, image generation, Hub
download, or serving routes.

## Files Reviewed

- `packages/diffusion/src/errors.ts`
- `packages/diffusion/src/index.ts`
- `packages/diffusion/src/pretrained/scheduler-config.ts`

## Tensor Lifetime Audit

This tranche adds config parsing and scheduler construction only. It does not
introduce new tensor-producing operations. Constructed scheduler instances keep
the existing tensor lifetime behavior from the scheduler foundation tranche.

## Memory / Performance Evidence

No generation hot-path benchmark is required. The changed production code reads
JSON metadata and constructs scheduler objects; no denoising loop, model
forward path, or serving route changed.

Focused tests cover DDIM and Euler config parsing, local snapshot loading,
scheduler instantiation, unsupported scheduler class rejection, unsupported
prediction type rejection, unsupported beta schedule rejection, and unsupported
Euler sigma variant rejection.

## Independent Review

Copernicus was asked for a read-only second pass on the scheduler config
boundary. The requested direction was to keep this tranche limited to real
checkpoint metadata and supported scheduler semantics, preserve the config path
and raw metadata in the loader result, support a scheduler subfolder, and avoid
pipeline or model-backbone claims.

## Remaining Risks / Follow-ups

- `prediction_type="v_prediction"`, trained betas, zero-SNR beta rescaling,
  Karras/exponential/beta sigma variants, ancestral Euler, and DPM-Solver
  schedulers remain unsupported and fail closed.
- The loader currently consumes local snapshot directories only. Hugging Face
  Hub resolution belongs with the broader diffusion checkpoint-loading tranche.
- Stable Diffusion and Flux pipelines still need model-owned UNet/VAE,
  conditioning, and image output evidence before any user-facing image command
  lands.
