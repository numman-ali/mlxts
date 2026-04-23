import { applyLoRAToModule, mergeLoRAInModule } from "@mlxts/lora";
import { QuantizedLinear } from "@mlxts/nn";
import {
  type CausalLM,
  type InteractionProfile,
  loadCausalLM,
  resolveLoRATargets,
} from "@mlxts/transformers";

import type { DPOProofProfile, TrainingProofArgs } from "./args";
import {
  DEFAULT_DPO_BETA,
  ensureQuantizedSnapshot,
  evaluatePreferenceDatasetLoss,
  evaluatePreferenceMetrics,
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
  DPOProofConfig,
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

function resolveDPOProofConfig(profile: DPOProofProfile): DPOProofConfig {
  if (profile === "handbook") {
    return {
      profile,
      preset: "attention+mlp",
      lastLayers: null,
      rank: 32,
      alpha: 16,
      dropout: 0.05,
      learningRate: 1e-5,
      beta: 0.01,
    };
  }

  return {
    profile,
    preset: "attention",
    lastLayers: 2,
    rank: 8,
    alpha: 16,
    dropout: 0,
    learningRate: 5e-5,
    beta: DEFAULT_DPO_BETA,
  };
}

function applyDPOTrainingLoRA(model: CausalLM, profile: DPOProofProfile): AppliedLoRA {
  const config = resolveDPOProofConfig(profile);
  const resolved = resolveLoRATargets(model, {
    preset: config.preset,
    lastLayers: config.lastLayers ?? undefined,
  });
  applyLoRAToModule(expectTrainableModule(model), {
    paths: resolved.paths,
    rank: config.rank,
    alpha: config.alpha,
    dropout: config.dropout,
  });
  return {
    preset: config.preset,
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
  const dpoConfig = resolveDPOProofConfig(args.dpoProfile);
  const padTokenId = readPadTokenId(tokenizer);
  const beforeLoss = evaluatePreferenceDatasetLoss(
    policyModel,
    referenceModel,
    data.preferenceEval,
    padTokenId,
    args.batchSize,
  );
  const beforeMetrics = evaluatePreferenceMetrics(
    policyModel,
    referenceModel,
    data.preferenceEval,
    padTokenId,
    args.batchSize,
    dpoConfig.beta,
  );
  const appliedLoRA = applyDPOTrainingLoRA(policyModel, args.dpoProfile);
  const averageTrainingLoss = runPreferenceTrainingSteps(
    policyModel,
    referenceModel,
    data.preferenceTrain,
    padTokenId,
    args.batchSize,
    args.steps,
    args.seed + 3,
    dpoConfig.learningRate,
    dpoConfig.beta,
  );
  const merged = mergeLoRAInModule(expectTrainableModule(policyModel));
  const afterLoss = evaluatePreferenceDatasetLoss(
    policyModel,
    referenceModel,
    data.preferenceEval,
    padTokenId,
    args.batchSize,
    dpoConfig.beta,
  );
  const afterMetrics = evaluatePreferenceMetrics(
    policyModel,
    referenceModel,
    data.preferenceEval,
    padTokenId,
    args.batchSize,
  );

  return {
    stage: "dpo",
    evalLoss: summarizeMetric(beforeLoss, afterLoss),
    rewardAccuracy: summarizeMetric(beforeMetrics.rewardAccuracy, afterMetrics.rewardAccuracy),
    rewardMargin: summarizeMetric(beforeMetrics.rewardMargin, afterMetrics.rewardMargin),
    chosenReward: summarizeMetric(beforeMetrics.chosenReward, afterMetrics.chosenReward),
    rejectedReward: summarizeMetric(beforeMetrics.rejectedReward, afterMetrics.rejectedReward),
    chosenLogProb: summarizeMetric(beforeMetrics.chosenLogProb, afterMetrics.chosenLogProb),
    rejectedLogProb: summarizeMetric(beforeMetrics.rejectedLogProb, afterMetrics.rejectedLogProb),
    rawPreferenceAccuracy: summarizeMetric(
      beforeMetrics.rawPreferenceAccuracy,
      afterMetrics.rawPreferenceAccuracy,
    ),
    averageTrainingLoss,
    sampleText: sampleText(policyModel, tokenizer, profile, data.samplePromptMessages),
    notes: [
      "reference_model=frozen_copy",
      `dpo_profile=${dpoConfig.profile}`,
      `preset=${appliedLoRA.preset}`,
      `target_count=${appliedLoRA.targets.length}`,
      `merged_targets=${merged.targets.length}`,
      `rank=${dpoConfig.rank}`,
      `alpha=${dpoConfig.alpha}`,
      `dropout=${dpoConfig.dropout}`,
      `learning_rate=${dpoConfig.learningRate}`,
      `beta=${dpoConfig.beta}`,
      dpoConfig.lastLayers === null ? "last_layers=all" : `last_layers=${dpoConfig.lastLayers}`,
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
  const rewardAccuracy =
    report.rewardAccuracy === undefined
      ? ""
      : ` reward_acc_before=${report.rewardAccuracy.before.toFixed(4)} reward_acc_after=${report.rewardAccuracy.after.toFixed(4)} delta=${report.rewardAccuracy.delta.toFixed(4)}`;
  const rewardMargin =
    report.rewardMargin === undefined
      ? ""
      : ` reward_margin_before=${report.rewardMargin.before.toFixed(4)} reward_margin_after=${report.rewardMargin.after.toFixed(4)} delta=${report.rewardMargin.delta.toFixed(4)}`;
  const rawPreferenceAccuracy =
    report.rawPreferenceAccuracy === undefined
      ? ""
      : ` raw_pref_acc_before=${report.rawPreferenceAccuracy.before.toFixed(4)} raw_pref_acc_after=${report.rawPreferenceAccuracy.after.toFixed(4)} delta=${report.rawPreferenceAccuracy.delta.toFixed(4)}`;
  const trainingLoss =
    report.averageTrainingLoss === undefined
      ? ""
      : ` train_loss=${report.averageTrainingLoss.toFixed(4)}`;
  console.log(
    `[${report.stage}]${evalLoss}${rewardAccuracy}${rewardMargin}${rawPreferenceAccuracy}${trainingLoss}`,
  );
  if (report.chosenReward !== undefined && report.rejectedReward !== undefined) {
    console.log(
      `  - chosen_reward_before=${report.chosenReward.before.toFixed(4)} chosen_reward_after=${report.chosenReward.after.toFixed(4)} delta=${report.chosenReward.delta.toFixed(4)}`,
    );
    console.log(
      `  - rejected_reward_before=${report.rejectedReward.before.toFixed(4)} rejected_reward_after=${report.rejectedReward.after.toFixed(4)} delta=${report.rejectedReward.delta.toFixed(4)}`,
    );
  }
  if (report.chosenLogProb !== undefined && report.rejectedLogProb !== undefined) {
    console.log(
      `  - chosen_logp_before=${report.chosenLogProb.before.toFixed(4)} chosen_logp_after=${report.chosenLogProb.after.toFixed(4)} delta=${report.chosenLogProb.delta.toFixed(4)}`,
    );
    console.log(
      `  - rejected_logp_before=${report.rejectedLogProb.before.toFixed(4)} rejected_logp_after=${report.rejectedLogProb.after.toFixed(4)} delta=${report.rejectedLogProb.delta.toFixed(4)}`,
    );
  }
  for (const note of report.notes) {
    console.log(`  - ${note}`);
  }
  if (report.sampleText !== undefined) {
    console.log(`  sample: ${report.sampleText}`);
  }
}
