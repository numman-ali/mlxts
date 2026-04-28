# @mlxts/align

Recipe layer above `@mlxts/train`. SFT, DPO, and future ORPO/KTO are per-step trainer functions composed of `valueAndGrad` plus `applyGradientStep`. Growing into framework lifecycle is forbidden — recipes are functions that take a step and return the next state.

`loss-utils.ts` is shared math, kept pure. `preferenceLogProbSums` and `preferenceRewardSums` are reused across DPO, ORPO, and KTO without copying.

Dataset evaluation lives separately from training step runners. `evaluateSupervisionDatasetLoss`, `evaluatePreferenceDatasetLoss`, and `evaluatePreferenceMetrics` do not share files with `runSupervisionTrainingSteps` or `runPreferenceTrainingSteps`.

Chat-template-aware example construction lives here. Composing tokenizer plus `ChatTemplate` to turn raw rows into trainable shape is align's responsibility, not data's.

Seven internal dependencies (`core`, `data`, `lora`, `nn`, `tokenizers`, `train`, `transformers`) is by design. New internal dependencies require a recipe-layer reason. Wire-format and serving dependencies are forbidden.

DPO reporting uses reference-aware reward metrics: reward accuracy plus reward margin, with chosen/rejected rewards and log-probs recorded alongside loss. Policy-only `raw_pref_acc` is a supplemental debug signal, never the canonical readout.

`recipes.ts` carries cap pressure. New recipes split into siblings, not additions to that file.
