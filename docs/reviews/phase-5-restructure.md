# Runtime Review: Phase 5 Ecosystem Restructure

## Summary

This artifact tracks the runtime-sensitive review for the Phase 5 package
restructure from `mlx-ts` + `nanogpt` into the `@mlxts/*` package family. The
current implementation focus is the reusable package surface: tensor, nn,
optimizer, training, data, and tokenizer code are extracted and split into
smaller files without changing their runtime behavior. The old `packages/nanogpt`
surface remains a temporary validation fixture rather than the long-term
canonical example.

## Current State vs Deferred Work

The package-first extraction is the current Phase 5 target and is reflected in
the repo today. The extracted packages are the canonical surfaces. The deferred
work is a later dedicated examples pass, including a ground-up nanoGPT rewrite
and deletion of the temporary `packages/mlx-ts` shim and `packages/nanogpt`
fixture once replacement surfaces are ready.

The review is intentionally incremental. The `Files Reviewed` section is updated
as runtime-sensitive files move or split, and the evidence sections are
refreshed as the migration reaches new validation checkpoints.

## Files Reviewed

- `packages/core/src/array-data.ts`
- `packages/core/src/array-ffi-data.ts`
- `packages/core/src/array.ts`
- `packages/core/src/device.ts`
- `packages/core/src/dtype.ts`
- `packages/core/src/error.ts`
- `packages/core/src/fast.ts`
- `packages/core/src/ffi/closure-bridge.ts`
- `packages/core/src/ffi/index.ts`
- `packages/core/src/ffi/lib.ts`
- `packages/core/src/ffi/pointer.ts`
- `packages/core/src/ffi/symbols.ts`
- `packages/core/src/format-shape.ts`
- `packages/core/src/index.ts`
- `packages/core/src/io.ts`
- `packages/core/src/memory.ts`
- `packages/core/src/metal.ts`
- `packages/core/src/ops/arithmetic.ts`
- `packages/core/src/ops/comparison.ts`
- `packages/core/src/ops/index.ts`
- `packages/core/src/ops/linalg.ts`
- `packages/core/src/ops/reduction.ts`
- `packages/core/src/ops/shape.ts`
- `packages/core/src/random.ts`
- `packages/core/src/transforms-autograd.ts`
- `packages/core/src/transforms-base.ts`
- `packages/core/src/transforms-compile.ts`
- `packages/core/src/transforms.ts`
- `packages/core/src/tree.ts`
- `packages/data/src/index.ts`
- `packages/data/src/text.ts`
- `packages/mlx-ts/src/index.ts`
- `packages/nn/src/activations.ts`
- `packages/nn/src/checkpoint.ts`
- `packages/nn/src/dropout.ts`
- `packages/nn/src/embedding.ts`
- `packages/nn/src/index.ts`
- `packages/nn/src/layer-norm.ts`
- `packages/nn/src/linear.ts`
- `packages/nn/src/losses.ts`
- `packages/nn/src/module.ts`
- `packages/nn/src/value-and-grad.ts`
- `packages/optimizers/src/adam.ts`
- `packages/optimizers/src/index.ts`
- `packages/optimizers/src/optimizer.ts`
- `packages/optimizers/src/sgd.ts`
- `packages/tokenizers/src/char.ts`
- `packages/tokenizers/src/index.ts`
- `packages/train/src/checkpoint-io.ts`
- `packages/train/src/checkpoint-manifest.ts`
- `packages/train/src/checkpoint-serialization.ts`
- `packages/train/src/checkpoint-types.ts`
- `packages/train/src/checkpoint.ts`
- `packages/train/src/gradients.ts`
- `packages/train/src/index.ts`
- `packages/train/src/loop.ts`
- `packages/train/src/schedule.ts`

## Tensor Lifetime Audit

The restructure starts from the current Phase 4 runtime model and preserves the
existing tensor-lifetime rules while files move or split:

- keep disposable tensor lifetimes visible during refactors
- preserve `using` / explicit free behavior when splitting oversized files
- keep FFI output-slot ownership per call during the core extraction
- rerun `bun run check:tensor-lifetimes` after each migration checkpoint

## Memory / Performance Evidence

Current evidence:

- `bun run validate` passes on the current package-first layout
- `cd packages/core && bun test` passes after splitting `array.ts` and `transforms.ts`
- `cd packages/train && bun test` passes after extracting the generic training
  schedule, loop, gradient, and checkpoint surfaces
- `bun run check:file-lines` passes for the canonical `@mlxts/*` package sources

Throughput-sensitive long-run evidence for the legacy `packages/nanogpt`
fixture remains a follow-up once the package extraction settles and the example
strategy is finalized.

## Independent Review

Implementation is being done by Codex. Independent review is still required
before the Phase 5 migration is considered complete.

## Remaining Risks / Follow-ups

- Runtime-sensitive file moves can accidentally hide tensor lifetimes if large
  files are split without preserving ownership clarity.
- The temporary `mlx-ts` shim still coexists with the extracted packages, so
  there is a short-term risk of drift until the old surface is deleted.
- The `packages/nanogpt` fixture is no longer the primary Phase 5 target, so
  example and operator docs still need a dedicated cleanup pass to match the new
  package-first direction.
- Long-run throughput evidence still flows through the temporary validation
  fixture until the later examples strategy is implemented.
