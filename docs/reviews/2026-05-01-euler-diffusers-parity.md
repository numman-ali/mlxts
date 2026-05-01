# Euler Diffusers Parity

## Summary

Aligned the Stable Diffusion Euler scheduler with Diffusers sigma-space
semantics so real SD/SDXL Euler checkpoint metadata can load without accepting
wrong denoising math. Euler now supports Diffusers `linspace`, `leading`, and
`trailing` timestep spacing, `steps_offset`, and `final_sigmas_type` values
while continuing to reject unimplemented Karras, exponential, and beta sigma
variants.

## Files Reviewed

- `packages/diffusion/src/schedulers/euler.ts`
- `packages/diffusion/src/families/stable-diffusion/pipeline.ts`
- `packages/diffusion/src/pretrained/scheduler-config.ts`

## Reference Audit

- `.reference/diffusers/src/diffusers/schedulers/scheduling_euler_discrete.py`
  confirms Euler timesteps are `0..num_train_timesteps - 1`, `linspace`
  preserves fractional timesteps, `leading` applies `steps_offset`, and
  `trailing` does not.
- Diffusers `init_noise_sigma` uses the active inference sigma ladder. Leading
  spacing uses `sqrt(max_sigma^2 + 1)`; linspace and trailing use `max_sigma`.
- Diffusers epsilon stepping updates sigma-space latents with
  `sample + model_output * (next_sigma - sigma)` after model-input scaling.

## Tensor Lifetime Audit

`EulerScheduler.step()` keeps the only tensor-producing intermediate, the
epsilon delta, in a local `using` binding and returns a caller-owned tensor.
`samplePrior()` frees internally-created random noise after scaling. Stable
Diffusion pipeline denoising continues to free each previous latent after the
next scheduler step is created, and the Euler branch now passes explicit sigma
pairs through the existing visible scheduler-step object.

## Memory / Performance Evidence

This tranche changes scalar scheduler math and does not add host/GPU
synchronization, extra model evaluations, or new tensor allocations in the
denoising loop beyond the existing Euler delta tensor. Focused tests cover the
Diffusers reference timestep ladders, final sigma policies, active-schedule
initial noise scale, and epsilon step numbers.

## Independent Review

Lagrange reviewed the scheduler boundary before implementation and identified
that simply relaxing the parser would create a false green: the existing Euler
implementation followed the MLX examples' normalized latent sampler, while real
Diffusers Stable Diffusion checkpoints expect sigma-space latents. The landed
change follows that recommendation and keeps unsupported sigma variants
rejected.

## Validation

- `bun test packages/diffusion/src/schedulers/euler.test.ts`
- `bun test packages/diffusion/src/pretrained/scheduler-config.test.ts`
- `bun test packages/diffusion/src/families/stable-diffusion/pipeline.test.ts`

## Remaining Risks / Follow-ups

- PNDM, DPM-Solver, Euler ancestral, Karras, exponential, and beta sigma
  variants remain unsupported.
- Real checkpoint image proof remains the next tranche now that SD/SDXL Euler
  metadata can be represented truthfully.
