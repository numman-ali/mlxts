import { applyLoRAToModule, mergeLoRAInModule } from "@mlxts/lora";
import { QuantizedLinear } from "@mlxts/nn";
import {
  type CausalLM,
  type InteractionProfile,
  loadCausalLM,
  resolveLoRATargets,
} from "@mlxts/transformers";

import type { TrainingProofArgs } from "./args";
import {
  ensureQuantizedSnapshot,
  evaluatePreferenceAccuracy,
  evaluatePreferenceDatasetLoss,
  evaluateSupervisionDatasetLoss,
  expectTrainableModule,
  readPadTokenId,
  runPreferenceTrainingSteps,
  runSupervisionTrainingSteps,
  sampleText,
  summarizeMetric,
} from "./runtime";
import type {
  AppliedLoRA,
  LoadedAssets,
  LoRAStageMode,
  PreparedTrainingProofData,
  StageReport,
} from "./types";

function applyTrainingLoRA(model: CausalLM, mode: LoRAStageMode): AppliedLoRA {
  const preset = mode === "qlora" ? "all-linear" : "attention";
  const resolved = resolveLoRATargets(model, {
    preset,
    lastLayers: 2,
  });
  applyLoRAToModule(expectTrainableModule(model), {
    paths: resolved.paths,
    rank: 8,
    alpha: 16,
    dropout: 0,
  });
  return {
    preset,
    targets: resolved.paths,
  };
}

export async function runLoRAStage(
  source: string,
  tokenizer: LoadedAssets["tokenizer"],
  profile: InteractionProfile,
  data: PreparedTrainingProofData,
  args: TrainingProofArgs,
): Promise<StageReport> {
  using model = await loadCausalLM(source);
  const padTokenId = readPadTokenId(tokenizer);
  const before = evaluateSupervisionDatasetLoss(
    model,
    data.supervisionEval,
    padTokenId,
    args.batchSize,
  );
  const appliedLoRA = applyTrainingLoRA(model, "lora");
  const averageTrainingLoss = runSupervisionTrainingSteps(
    model,
    data.supervisionTrain,
    padTokenId,
    args.batchSize,
    args.steps,
    args.seed,
    1e-4,
  );
  const merged = mergeLoRAInModule(expectTrainableModule(model));
  const after = evaluateSupervisionDatasetLoss(
    model,
    data.supervisionEval,
    padTokenId,
    args.batchSize,
  );

  return {
    stage: "lora",
    evalLoss: summarizeMetric(before, after),
    averageTrainingLoss,
    sampleText: sampleText(model, tokenizer, profile, data.samplePromptMessages),
    notes: [
      `preset=${appliedLoRA.preset}`,
      `target_count=${appliedLoRA.targets.length}`,
      `merged_targets=${merged.targets.length}`,
      `skipped=${merged.skipped.length}`,
      `train_examples=${data.supervisionTrain.length}`,
      `eval_examples=${data.supervisionEval.length}`,
    ],
  };
}

export async function runQLoRAStage(
  source: string,
  quantizedOutputDir: string,
  tokenizer: LoadedAssets["tokenizer"],
  profile: InteractionProfile,
  data: PreparedTrainingProofData,
  args: TrainingProofArgs,
): Promise<StageReport> {
  const qloraSource = await ensureQuantizedSnapshot(source, quantizedOutputDir);
  using model = await loadCausalLM(qloraSource);
  const padTokenId = readPadTokenId(tokenizer);
  const before = evaluateSupervisionDatasetLoss(
    model,
    data.supervisionEval,
    padTokenId,
    args.batchSize,
  );
  const appliedLoRA = applyTrainingLoRA(model, "qlora");
  const averageTrainingLoss = runSupervisionTrainingSteps(
    model,
    data.supervisionTrain,
    padTokenId,
    args.batchSize,
    args.steps,
    args.seed + 1,
    1e-4,
  );
  const merged = mergeLoRAInModule(expectTrainableModule(model));
  const after = evaluateSupervisionDatasetLoss(
    model,
    data.supervisionEval,
    padTokenId,
    args.batchSize,
  );
  const lastLayer = model.config.numHiddenLayers - 1;
  const lastProjection = readLastLlamaQProjection(model, lastLayer);
  const preservedQuantizedBase = lastProjection instanceof QuantizedLinear;

  return {
    stage: "qlora",
    evalLoss: summarizeMetric(before, after),
    averageTrainingLoss,
    sampleText: sampleText(model, tokenizer, profile, data.samplePromptMessages),
    notes: [
      `preset=${appliedLoRA.preset}`,
      `target_count=${appliedLoRA.targets.length}`,
      `merged_targets=${merged.targets.length}`,
      `quantized_base_preserved=${preservedQuantizedBase}`,
      `train_examples=${data.supervisionTrain.length}`,
      `eval_examples=${data.supervisionEval.length}`,
    ],
  };
}

function readLastLlamaQProjection(model: CausalLM, layerIndex: number): unknown {
  const backbone = Reflect.get(model, "model");
  if (typeof backbone !== "object" || backbone === null) {
    return null;
  }

  const layers = Reflect.get(backbone, "layers");
  if (!Array.isArray(layers)) {
    return null;
  }

  const layer = layers[layerIndex];
  if (typeof layer !== "object" || layer === null) {
    return null;
  }

  const selfAttention = Reflect.get(layer, "selfAttention");
  if (typeof selfAttention !== "object" || selfAttention === null) {
    return null;
  }

  return Reflect.get(selfAttention, "qProjection");
}

export async function runSFTStage(
  source: string,
  tokenizer: LoadedAssets["tokenizer"],
  profile: InteractionProfile,
  data: PreparedTrainingProofData,
  args: TrainingProofArgs,
): Promise<StageReport> {
  using model = await loadCausalLM(source);
  const padTokenId = readPadTokenId(tokenizer);
  const before = evaluateSupervisionDatasetLoss(
    model,
    data.supervisionEval,
    padTokenId,
    args.batchSize,
  );
  const averageTrainingLoss = runSupervisionTrainingSteps(
    model,
    data.supervisionTrain,
    padTokenId,
    args.batchSize,
    args.steps,
    args.seed + 2,
    5e-6,
  );
  const after = evaluateSupervisionDatasetLoss(
    model,
    data.supervisionEval,
    padTokenId,
    args.batchSize,
  );

  return {
    stage: "sft",
    evalLoss: summarizeMetric(before, after),
    averageTrainingLoss,
    sampleText: sampleText(model, tokenizer, profile, data.samplePromptMessages),
    notes: [
      "dense_model=true",
      `train_examples=${data.supervisionTrain.length}`,
      `eval_examples=${data.supervisionEval.length}`,
    ],
  };
}

export async function runDPOStage(
  source: string,
  tokenizer: LoadedAssets["tokenizer"],
  profile: InteractionProfile,
  data: PreparedTrainingProofData,
  args: TrainingProofArgs,
): Promise<StageReport> {
  using policyModel = await loadCausalLM(source);
  using referenceModel = await loadCausalLM(source);
  const padTokenId = readPadTokenId(tokenizer);
  const beforeLoss = evaluatePreferenceDatasetLoss(
    policyModel,
    referenceModel,
    data.preferenceEval,
    padTokenId,
    args.batchSize,
  );
  const beforeAccuracy = evaluatePreferenceAccuracy(
    policyModel,
    data.preferenceEval,
    padTokenId,
    args.batchSize,
  );
  const appliedLoRA = applyTrainingLoRA(policyModel, "dpo");
  const averageTrainingLoss = runPreferenceTrainingSteps(
    policyModel,
    referenceModel,
    data.preferenceTrain,
    padTokenId,
    args.batchSize,
    args.steps,
    args.seed + 3,
    5e-5,
  );
  const merged = mergeLoRAInModule(expectTrainableModule(policyModel));
  const afterLoss = evaluatePreferenceDatasetLoss(
    policyModel,
    referenceModel,
    data.preferenceEval,
    padTokenId,
    args.batchSize,
  );
  const afterAccuracy = evaluatePreferenceAccuracy(
    policyModel,
    data.preferenceEval,
    padTokenId,
    args.batchSize,
  );

  return {
    stage: "dpo",
    evalLoss: summarizeMetric(beforeLoss, afterLoss),
    preferenceAccuracy: summarizeMetric(beforeAccuracy, afterAccuracy),
    averageTrainingLoss,
    sampleText: sampleText(policyModel, tokenizer, profile, data.samplePromptMessages),
    notes: [
      "reference_model=frozen_copy",
      `preset=${appliedLoRA.preset}`,
      `target_count=${appliedLoRA.targets.length}`,
      `merged_targets=${merged.targets.length}`,
      `train_examples=${data.preferenceTrain.length}`,
      `eval_examples=${data.preferenceEval.length}`,
    ],
  };
}

export function printStage(report: StageReport): void {
  const evalLoss =
    report.evalLoss === undefined
      ? ""
      : ` eval_loss_before=${report.evalLoss.before.toFixed(4)} eval_loss_after=${report.evalLoss.after.toFixed(4)} delta=${report.evalLoss.delta.toFixed(4)}`;
  const preferenceAccuracy =
    report.preferenceAccuracy === undefined
      ? ""
      : ` pref_acc_before=${report.preferenceAccuracy.before.toFixed(4)} pref_acc_after=${report.preferenceAccuracy.after.toFixed(4)} delta=${report.preferenceAccuracy.delta.toFixed(4)}`;
  const trainingLoss =
    report.averageTrainingLoss === undefined
      ? ""
      : ` train_loss=${report.averageTrainingLoss.toFixed(4)}`;
  console.log(`[${report.stage}]${evalLoss}${preferenceAccuracy}${trainingLoss}`);
  for (const note of report.notes) {
    console.log(`  - ${note}`);
  }
  if (report.sampleText !== undefined) {
    console.log(`  sample: ${report.sampleText}`);
  }
}
