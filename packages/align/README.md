# `@mlxts/align`

Recipe-level supervised fine-tuning and preference-optimization helpers.

`@mlxts/align` builds on `@mlxts/train`, `@mlxts/data`, and `@mlxts/lora`
without changing the lower-level training or model contracts.

It is intentionally the recipe layer, not a trainer framework. The package owns
alignment math, chat-example shaping, raw-chat normalization helpers, and small
reusable SFT/DPO loops so that examples and future CLIs can stay thin without
hiding control flow behind a black-box `Trainer`.

```ts
import {
  prepareChatSupervisionExamples,
  evaluatePreferenceMetrics,
  evaluateSupervisionDatasetLoss,
  runSupervisionTrainingSteps,
} from "@mlxts/align";
import { Adam } from "@mlxts/optimizers";

const optimizer = new Adam({ learningRate: 1e-4 });

const prepared = prepareChatSupervisionExamples(tokenizer, template, rawMessages, {
  limit: 128,
  maxSequenceLength: 1024,
});

const evalLoss = evaluateSupervisionDatasetLoss(model, {
  examples: prepared.examples,
  padTokenId,
  batchSize: 4,
});

const result = runSupervisionTrainingSteps(model, {
  optimizer,
  examples: prepared.examples,
  padTokenId,
  batchSize: 4,
  steps: 32,
  seed: 7,
  learningRate: 1e-4,
});

const dpoMetrics = evaluatePreferenceMetrics(policyModel, {
  referenceModel,
  examples: preferenceExamples,
  padTokenId,
  batchSize: 4,
  beta: 0.1,
});
```
