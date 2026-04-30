# FlowMatch Euler Scheduler

## Summary

Added a package-owned FlowMatch Euler scheduler for Flux-style rectified-flow
denoising. The scheduler is separate from the Stable Diffusion Euler scheduler
because FlowMatch uses unscaled normal priors, identity model-input scaling,
forward noise `(1 - sigma) * sample + sigma * noise`, and deterministic reverse
steps `sample + (nextSigma - sigma) * modelOutput`.

Diffusers `FlowMatchEulerDiscreteScheduler` metadata now parses into
`@mlxts/diffusion`, and local Flux manifests fail only on unsupported scheduler
knobs instead of failing because the scheduler class is unknown.

## Files Reviewed

- `packages/diffusion/src/schedulers/flow-match-euler.ts`
- `packages/diffusion/src/index.ts`
- `packages/diffusion/src/pretrained/flow-match-scheduler-config.ts`
- `packages/diffusion/src/pretrained/scheduler-config.ts`
- `packages/diffusion/src/pretrained/snapshot-manifest.ts`
- `packages/diffusion/src/families/stable-diffusion/pipeline-loading.ts`

## Reference Audit

- `.reference/diffusers/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py`
  defines FlowMatch forward noise and Euler reverse stepping.
- `.reference/diffusers/src/diffusers/pipelines/flux/pipeline_flux.py` defines
  Flux resolution-dependent shift calculation and the `linspace(1, 1 / steps)`
  sigma schedule used by the pipeline.
- `.reference/mlx-examples/flux/flux/sampler.py` confirms MLX Flux uses plain
  normal prior sampling, optional dynamic time shift, flow forward noise, and
  `x_t + (t_prev - t) * pred` reverse stepping.
- Public FLUX.1 scheduler configs currently use
  `_class_name: "FlowMatchEulerDiscreteScheduler"` with `base_image_seq_len`,
  `base_shift`, `max_image_seq_len`, `max_shift`, `num_train_timesteps`,
  `shift`, `time_shift_type`, and inactive sigma-variant flags.

## Tensor Lifetime Audit

`FlowMatchEulerScheduler.step()` stores the velocity-scaled delta in a named
`using` binding before returning the updated sample. `scaleNoise()` stores both
the sample and noise products in named `using` bindings before adding them.
`scaleInitialNoise()` and `scaleModelInput()` use `retainArray()` because
FlowMatch does not rescale those tensors.

`pipeline-loading.ts` changed only the scheduler type boundary for Stable
Diffusion bundles. It rejects FlowMatch scheduler configs before constructing a
Stable Diffusion bundle and does not introduce new tensor ownership.

## Memory / Performance Evidence

- `bun test packages/diffusion/src/schedulers/flow-match-euler.test.ts packages/diffusion/src/pretrained/scheduler-config.test.ts packages/diffusion/src/pretrained/model-index.test.ts`: 22 pass, 0 fail.
- `bun run --filter @mlxts/diffusion typecheck`: passed.

The tranche adds scheduler math and config parsing only. It does not add a Flux
transformer, text encoder composition, latent packing, or a full image sampling
loop, so it makes no image quality or throughput claim.

## Independent Review

Volta completed a read-only second pass before implementation. The review
recommended a separate FlowMatch scheduler, explicit step records, Flux
dynamic-shift support, fail-closed config parsing, and a runtime review artifact
because the scheduler owns tensor step and noise methods.

## Remaining Risks / Follow-ups

- Custom Diffusers `timesteps` and stochastic/per-token FlowMatch paths remain
  unsupported until a real model path needs them.
- Flux transformer config translation, latent packing, CLIP plus T5
  conditioning, and real checkpoint image proof remain follow-on Phase 10
  tranches.

## Out-of-scope drift noticed

None.
