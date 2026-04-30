# Diffusion Scheduler Foundation

## Summary

Created the initial `@mlxts/diffusion` package boundary and landed scheduler
infrastructure for Phase 10 diffusion/flow generation. The tranche adds
reference-backed Euler and deterministic DDIM scheduler primitives, package
registration, coverage/runtime-review governance, and package doctrine.

Stable Diffusion, Flux, VAE/backbone loading, text conditioning, and image
writing are not claimed by this tranche.

## Files Reviewed

- `packages/diffusion/src/index.ts`
- `packages/diffusion/src/schedulers/ddim.ts`
- `packages/diffusion/src/schedulers/euler.ts`
- `packages/diffusion/src/schedulers/schedule.ts`

## Tensor Lifetime Audit

Scheduler step functions keep tensor-producing intermediates in local `using`
bindings and return only caller-owned output tensors. `DDIMScheduler.step`
explicitly frees the unclipped predicted-original sample when clipping produces
a replacement tensor and frees returned intermediate ownership on error.

`samplePrior` frees internally-created random noise after creating the scaled
latent output. Schedule arrays stay as JavaScript numeric state; they are small
scalar coefficient tables, not MLX tensors.

## Memory / Performance Evidence

No generation hot-path benchmark is required. This tranche adds a new diffusion
package and scheduler primitives only; no transformer generation path, serving
route, UNet/VAE module, or model checkpoint hot path changed.

Focused tests cover numeric schedule generation, Euler step/update formulas,
and deterministic DDIM step/noise/clip formulas. Full validation remains the
required gate before commit.

## Independent Review

Second-opinion request sent to Locke before implementation to compare next
tranche candidates. The review recommended a diffusion-first Phase 10 slice
and explicitly warned against an empty package. This tranche follows that
boundary by landing tested scheduler primitives and package doctrine only,
rather than widening `@mlxts/transformers` or claiming a pipeline before
backbones and VAE are package-owned.

## Remaining Risks / Follow-ups

- The scheduler surface covers Euler and deterministic DDIM formulas only.
  Ancestral noise, Karras/exponential/beta sigma variants, DPM-Solver, and flow
  matching remain follow-on scheduler tranches.
- `@mlxts/diffusion` does not yet load checkpoints, own UNet/VAE modules, or
  generate images.
- The first real text-to-image proof must add a paired runtime artifact with
  real checkpoint evidence and memory numbers.
