# `@mlxts/train`

Training orchestration and checkpointing for mlxts.

`@mlxts/train` builds on `@mlxts/core`, `@mlxts/nn`, and `@mlxts/optimizers` to provide learning-rate schedules, gradient utilities, typed checkpoint metadata, and reusable training-loop helpers.

```ts
import { getLearningRate, warmupCosineSchedule } from "@mlxts/train";

const learningRate = getLearningRate(100, {
  maxSteps: 1_000,
  learningRate: 3e-4,
  minLearningRate: 3e-5,
  warmupSteps: 100,
});
```
