# @mlxts/train

`trainLoop` is a function. A `Trainer` base class is forbidden. Per-step orchestration runs in user code that imports `applyGradientStep`, `materializeTrainingState`, and the chosen optimizer step.

Lifecycle hook surfaces — `before_optimizer_step`, `on_epoch_end`, `on_batch_end` — are forbidden. Callbacks belong to the caller's loop, not to a framework registry.

`applyGradientStep` and `materializeTrainingState` keep ownership and `mxEval`/`synchronize` calls visible. Hidden async eval is forbidden — readers see when the GPU is being driven.

Snapshot checkpoints (lightweight model saves) and resume checkpoints (optimizer state for exact continuation) are distinct. `CheckpointKind` discriminates them. Manifest parsing is typed and validated.

Schedules are pure `(step) => lr` functions. Training state on the schedule object is forbidden.

Gradient helpers — `accumulateGradients`, `accumulateGradientTrees`, `clipGradientTree`, `gradientNorm`, `freeGradientTree`, `evalGradientTree`, `scaleGradientTree` — own gradient lifetime. Callers using these helpers do not call `mxEval` on grads directly.
