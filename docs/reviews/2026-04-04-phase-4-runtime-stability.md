# Runtime Review: Phase 4 Runtime Stability Hardening

## Summary

This review covers the Phase 4 runtime-sensitive work around long-run GPT training, checkpoint control, runtime telemetry, and the leak investigation that identified anonymous disposable tensor intermediates as a concrete failure class. The goal of this artifact is to make the review evidence durable and to keep the repo's new runtime-safety posture anchored in an actual incident.

The same review now also covers the fused fast-attention migration, module-backed checkpoint transforms, the visible-lifetime validation gate, and the new benchmark/soak surfaces that turn the incident response into repeatable engineering practice.

The latest follow-up also covers the removal of the shared FFI output buffer pattern, explicit disposal on transform-returning helpers, the fast fused layer-norm path, and model-weight safetensors interop.

The current follow-up also covers the supervised soak wrapper itself. A live `gpt-small` soak run exposed that the default throughput window expected more logged step events than the default `50`-step soak can emit. The review for this change focused on making the acceptance math self-consistent with the actual log cadence rather than silently weakening the soak criteria.

## Files Reviewed

- `packages/nanogpt/src/model/causal-self-attention.ts`
- `packages/nanogpt/src/model/transformer-block.ts`
- `packages/nanogpt/src/train.ts`
- `packages/nanogpt/src/checkpoint.ts`
- `packages/nanogpt/src/bench/memory.ts`
- `packages/nanogpt/src/run/acceptance.ts`
- `packages/nanogpt/src/run/manager.ts`
- `packages/nanogpt/src/run/soak.ts`
- `packages/nanogpt/src/run/supervisor.ts`
- `packages/nanogpt/src/run/files.ts`
- `packages/mlx-ts/src/core/array.ts`
- `packages/mlx-ts/src/core/device.ts`
- `packages/mlx-ts/src/core/fast.ts`
- `packages/mlx-ts/src/core/io.ts`
- `packages/mlx-ts/src/core/dtype.ts`
- `packages/mlx-ts/src/core/transforms.ts`
- `packages/mlx-ts/src/core/memory.ts`
- `packages/mlx-ts/src/core/metal.ts`
- `packages/mlx-ts/src/core/ffi/symbols.ts`
- `packages/mlx-ts/src/core/ffi/closure-bridge.ts`
- `packages/mlx-ts/src/core/ffi/pointer.ts`
- `packages/mlx-ts/src/core/ffi/index.ts`
- `packages/mlx-ts/src/core/ffi/lib.ts`
- `packages/mlx-ts/src/core/ops/arithmetic.ts`
- `packages/mlx-ts/src/core/ops/comparison.ts`
- `packages/mlx-ts/src/core/ops/index.ts`
- `packages/mlx-ts/src/core/ops/linalg.ts`
- `packages/mlx-ts/src/core/ops/reduction.ts`
- `packages/mlx-ts/src/core/ops/shape.ts`
- `packages/mlx-ts/src/core/random.ts`
- `packages/mlx-ts/src/nn/checkpoint.ts`
- `packages/mlx-ts/src/nn/dropout.ts`
- `packages/mlx-ts/src/nn/index.ts`
- `packages/mlx-ts/src/nn/embedding.ts`
- `packages/mlx-ts/src/nn/layer-norm.ts`
- `packages/mlx-ts/src/nn/linear.ts`
- `packages/mlx-ts/src/nn/module.ts`
- `packages/mlx-ts/src/nn/losses.ts`
- `packages/mlx-ts/src/nn/value-and-grad.ts`
- `packages/mlx-ts/src/optimizers/adam.ts`
- `packages/mlx-ts/src/optimizers/index.ts`
- `packages/mlx-ts/src/optimizers/optimizer.ts`
- `packages/mlx-ts/src/optimizers/sgd.ts`
- `packages/nanogpt/src/cli.ts`
- `packages/nanogpt/src/config.ts`
- `packages/nanogpt/src/data.ts`
- `packages/nanogpt/src/generate.ts`
- `packages/nanogpt/src/index.ts`
- `packages/nanogpt/src/model/gpt.ts`
- `packages/nanogpt/src/model/init.ts`
- `packages/nanogpt/src/model/mlp.ts`
- `packages/nanogpt/src/optimizer-defaults.ts`
- `packages/nanogpt/src/safetensors.ts`
- `packages/nanogpt/src/run/acceptance.ts`
- `packages/nanogpt/src/tokenizer.ts`

## Tensor Lifetime Audit

The runtime review focused on places where `MxArray` ownership could be lost by composition rather than by explicit names. The concrete incident was the nested-disposable pattern in attention code, where intermediate tensors created by `reshape(...)` and `ones(...)` were easy to miss when wrapped immediately by other ops. That hot path now uses fused fast attention plus named `using` intermediates, and the repo backs the rule with `bun run check:tensor-lifetimes`.

The follow-up audit also covered two lower-level ownership surfaces that can silently undermine otherwise clean code:

- result-pointer handling at the FFI boundary now uses per-call `OutSlot` allocation rather than a shared module-level output buffer
- transform-returning helpers now expose explicit disposal so native closures and compiled transform handles do not rely on eventual GC

The same audit was applied to the new checkpoint and soak surfaces:

- module-backed checkpoint transforms must restore original parameter handles in `finally`
- gradient checkpointing should only wrap training-time execution, not eval
- long-run operator code must stay under package source and emit enough structured evidence to debug throughput, memory, and checkpoint health
- soak acceptance math must derive its default throughput window from the number of step events the configured `maxSteps` and `logInterval` can actually produce

## Memory / Performance Evidence

The underlying incident showed a reproducible `gpt-small` degradation pattern: throughput falling from roughly `10.3k tok/s` to roughly `1.6k tok/s` by step `130`, with active memory climbing in a way that matched leaked anonymous intermediates. Additional investigation separated allocator cache behavior from live-memory growth and confirmed that `set_memory_limit()` is not a hard cap and `clear_cache()` does not fix live tensor retention.

The concrete follow-up evidence for this pass is:

- a fresh-process memory benchmark surface (`bun run bench:memory`)
- direct tests for fused fast attention and checkpoint transforms
- a supervised soak surface (`bun run soak:gpt-tiny`, `bun run soak:gpt-small`)
- a static validation gate for suspicious anonymous tensor intermediates
- direct round-trip coverage for safetensors weight export/import
- targeted review of the fast fused layer-norm path and comparison-scalar coercion semantics
- a corrected `gpt-small` 50-step soak run that now completes successfully with a self-consistent throughput window
- a corrected `gpt-small` 250-step soak run that completed cleanly with flat active memory (~1.03 GB), stable throughput (~8.6k tok/s early vs ~7.9k tok/s late), and a final validation loss of about `2.448`

## Independent Review

Codex performed this review as the independent reviewer for the runtime-sensitive Phase 4 changes authored elsewhere in the branch. This artifact records the review intent, the files examined, and the specific runtime risk class that triggered the new guardrails. The follow-up implementation was then re-reviewed against the same runtime criteria rather than treated as self-justifying.

## Remaining Risks / Follow-ups

- The long-run soak ladder still needs to stay green after the runtime fixes land; a review artifact is necessary but not sufficient.
- The soak wrapper now judges the `50`-step rung correctly, but longer rungs still need to keep earning trust through actual supervised evidence rather than inherited confidence from shorter runs.
- Compiled training-state capture is still a follow-on optimization path. The repo now exposes compile/checkpoint primitives cleanly, but the training loop should only adopt more aggressive compilation after the new benchmark and soak evidence stays healthy.
- Broader `mlx-c` family coverage should continue to be evaluated deliberately rather than symbol-by-symbol.
- If future runtime incidents show the same class of failure, the next fix should strengthen the static checker or benchmark coverage rather than only patching the local code again.
