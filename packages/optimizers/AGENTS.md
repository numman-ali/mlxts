# @mlxts/optimizers

Optimizers extend `Optimizer` and implement `applySingle(key, param, grad, prevState)`. Cross-parameter coordination is the responsibility of `Optimizer.update`, not `applySingle`.

State is `Map<string, Record<string, MxArray>>` keyed by dot-joined parameter path. Parameter paths match the `@mlxts/nn` parameter-scan output exactly.

`update(model, gradients)` orchestrates path-keyed gradient lookup, atomic state staging, and old-array cleanup. Sub-step ordering is fixed; new optimizers do not override `update`.

Schedules are `(step) => lr` pure functions and live in `@mlxts/train`. Optimizers expose `setLearningRate(lr)`. Storing schedule state on the optimizer is forbidden.

Future fused or compiled optimizer steps replace `update()`, never `applySingle()`. The single-parameter contract is the boundary across optimizer variants.

`AdamWCheckpoint` is the canonical AdamW checkpoint shape. Checkpoint serialization for new optimizers grows as a sibling type, not as additions to `AdamWCheckpoint`.
