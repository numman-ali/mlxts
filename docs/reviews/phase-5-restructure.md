# Runtime Review: Phase 5 Ecosystem Restructure

## Summary

This artifact tracks the runtime-sensitive review for the Phase 5 package
restructure from the legacy monolith into the `@mlxts/*` package family. The
current implementation focus is the reusable package surface: tensor, nn,
optimizer, training, data, and tokenizer code are extracted and split into
smaller files without changing their runtime behavior. The old `examples/nanogpt`
surface remains a temporary validation example rather than the long-term
canonical example.

## Current State vs Deferred Work

The package-first extraction is the current Phase 5 target and is reflected in
the repo today. The extracted packages are the canonical surfaces, and the old
compatibility shim has been removed. The deferred work is a later dedicated
examples pass, including a ground-up nanoGPT rewrite and eventual deletion of
the temporary `examples/nanogpt` fixture once replacement surfaces are ready.

The current checkpointing, data, tokenizer, and step-orchestration direction is
now package-canonical: `@mlxts/train`, `@mlxts/data`, and `@mlxts/tokenizers`
own the reusable implementations, while `examples/nanogpt` carries only GPT-
specific adapters and the operator-facing validation harness.

That fixture has now been split down below the repo-wide 500-line production
file cap. The runtime-sensitive CLI and run-manager code is still present, but
it is broken into smaller modules so tensor ownership, checkpoint behavior, and
operator-state transitions remain inspectable by eye.

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
- `examples/nanogpt/src/bench/memory.ts`
- `examples/nanogpt/src/checkpoint.ts`
- `examples/nanogpt/src/cli/commands.ts`
- `examples/nanogpt/src/cli/help.ts`
- `examples/nanogpt/src/cli/session.ts`
- `examples/nanogpt/src/cli/shared.ts`
- `examples/nanogpt/src/cli/train-events.ts`
- `examples/nanogpt/src/cli.ts`
- `examples/nanogpt/src/generate.ts`
- `examples/nanogpt/src/index.ts`
- `examples/nanogpt/src/model/causal-self-attention.ts`
- `examples/nanogpt/src/model/gpt.ts`
- `examples/nanogpt/src/model/init.ts`
- `examples/nanogpt/src/model/mlp.ts`
- `examples/nanogpt/src/model/transformer-block.ts`
- `examples/nanogpt/src/optimizer-defaults.ts`
- `examples/nanogpt/src/run/acceptance.ts`
- `examples/nanogpt/src/run/acceptance-options.ts`
- `examples/nanogpt/src/run/acceptance-runtime.ts`
- `examples/nanogpt/src/safetensors.ts`
- `examples/nanogpt/src/train.ts`
- `examples/nanogpt/src/run/files-health.ts`
- `examples/nanogpt/src/run/files-json.ts`
- `examples/nanogpt/src/run/files-paths.ts`
- `examples/nanogpt/src/run/files-types.ts`
- `examples/nanogpt/src/run/files.ts`
- `examples/nanogpt/src/run/manager-args.ts`
- `examples/nanogpt/src/run/manager-run.ts`
- `examples/nanogpt/src/run/manager-status.ts`
- `examples/nanogpt/src/run/manager.ts`
- `examples/nanogpt/src/run/supervisor-events.ts`
- `examples/nanogpt/src/run/supervisor-streams.ts`
- `examples/nanogpt/src/run/supervisor.ts`
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
- `packages/train/src/step.ts`

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
- `bun run release:check` passes for the current package manifests, build
  pipeline, TypeDoc config, and public tarball dry-runs
- `cd packages/core && bun test` passes after splitting `array.ts` and `transforms.ts`
- `cd packages/train && bun test` passes after extracting the generic training
  schedule, loop, gradient, checkpoint, and step-orchestration surfaces
- `cd examples/nanogpt && bun test` passes after rewiring the temporary fixture
  directly onto `@mlxts/core`, `@mlxts/nn`, `@mlxts/optimizers`,
  `@mlxts/train`, `@mlxts/data`, and `@mlxts/tokenizers`
- `cd examples/nanogpt && bun run acceptance:gpt-tiny` passes on the current package-first fixture
  with the re-baselined tiny-run loss target
- `bun run check:file-lines` passes for all active production source, including
  the temporary fixture

Throughput-sensitive long-run evidence for the legacy `examples/nanogpt`
fixture remains a follow-up once the package extraction settles and the example
strategy is finalized.

## Independent Review

Implementation is being done by Codex. Independent review is still required
before the Phase 5 migration is considered complete.

## Remaining Risks / Follow-ups

- Runtime-sensitive file moves can accidentally hide tensor lifetimes if large
  files are split without preserving ownership clarity.
- Rewiring the temporary `examples/nanogpt` fixture onto generic checkpoint
  metadata and package-owned training helpers touches training, checkpoint, and
  generation paths that still carry the long-run operational surface.
- The `examples/nanogpt` fixture is no longer the primary Phase 5 target, so
  example and operator docs still need a dedicated cleanup pass to match the new
  package-first direction.
- Long-run throughput evidence still flows through the temporary validation
  fixture until the later examples strategy is implemented.
