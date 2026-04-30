import { getPeakMemoryBytes, resetPeakMemory, treeFlatten } from "@mlxts/core";
import { applyLoRAToModule, assertQuantizedBasePreserved, mergeLoRAInModule } from "@mlxts/lora";
import type { Module } from "@mlxts/nn";
import {
  type CausalLM,
  expectTrainableModule,
  type InteractionProfile,
  loadCausalLM,
  loadCausalLMAdapters,
  resolveLoRATargets,
  saveCausalLMAdapters,
} from "@mlxts/transformers";
import { rmSync } from "fs";
import { join } from "path";

import type { DPOProofProfile, TrainingProofArgs } from "./args";
import {
  DEFAULT_DPO_BETA,
  ensureQuantizedSnapshot,
  evaluatePreferenceDatasetLoss,
  evaluatePreferenceMetrics,
  evaluateSupervisionDatasetLoss,
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

type AdapterCheck = NonNullable<StageReport["adapterCheck"]>;
type SavedAdapterCheck = Pick<AdapterCheck, "directory" | "trainedSampleText">;
type ParameterCounts = NonNullable<StageReport["parameterCounts"]>;
type MemoryReport = NonNullable<StageReport["memory"]>;

function countParameterElements(module: Module): ParameterCounts {
  const total = treeFlatten(module.parameters()).reduce(
    (sum, [, parameter]) => sum + parameter.size,
    0,
  );
  const trainable = treeFlatten(module.trainableParameters()).reduce(
    (sum, [, parameter]) => sum + parameter.size,
    0,
  );
  return { total, trainable };
}

function memoryReport(): MemoryReport {
  return {
    peakBytes: getPeakMemoryBytes(),
  };
}

function adapterDirectory(args: TrainingProofArgs, stage: LoRAStageMode): string {
  return join(args.adapterOutputDir, stage);
}

async function runAdapterCheck(
  stage: LoRAStageMode,
  loadSource: string,
  baseSource: string,
  model: CausalLM,
  tokenizer: LoadedAssets["tokenizer"],
  profile: InteractionProfile,
  data: PreparedTrainingProofData,
  args: TrainingProofArgs,
): Promise<AdapterCheck> {
  const saved = await saveAdapterForCheck(stage, baseSource, model, tokenizer, profile, data, args);
  return completeAdapterCheck(loadSource, saved, tokenizer, profile, data);
}

async function saveAdapterForCheck(
  stage: LoRAStageMode,
  baseSource: string,
  model: CausalLM,
  tokenizer: LoadedAssets["tokenizer"],
  profile: InteractionProfile,
  data: PreparedTrainingProofData,
  args: TrainingProofArgs,
): Promise<SavedAdapterCheck> {
  const directory = adapterDirectory(args, stage);
  rmSync(directory, { recursive: true, force: true });
  const trainedSampleText = sampleText(model, tokenizer, profile, data.samplePromptMessages);
  await saveCausalLMAdapters(model, directory, {
    baseModelNameOrPath: baseSource,
  });
  return {
    directory,
    trainedSampleText,
  };
}

async function completeAdapterCheck(
  loadSource: string,
  saved: SavedAdapterCheck,
  tokenizer: LoadedAssets["tokenizer"],
  profile: InteractionProfile,
  data: PreparedTrainingProofData,
): Promise<AdapterCheck> {
  using reloadedModel = await loadCausalLM(loadSource);
  await loadCausalLMAdapters(reloadedModel, saved.directory);
  const reloadedSampleText = sampleText(
    reloadedModel,
    tokenizer,
    profile,
    data.samplePromptMessages,
  );
  const reloadedMerge = mergeLoRAInModule(expectTrainableModule(reloadedModel));
  const reloadedMergedSampleText = sampleText(
    reloadedModel,
    tokenizer,
    profile,
    data.samplePromptMessages,
  );

  return {
    directory: saved.directory,
    reloadedMergeTargets: reloadedMerge.targets,
    trainedSampleText: saved.trainedSampleText,
    reloadedSampleText,
    reloadedMergedSampleText,
  };
}

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
  const resolveOptions: {
    preset: DPOProofConfig["preset"];
    lastLayers?: number;
  } = {
    preset: config.preset,
  };
  if (config.lastLayers !== null) {
    resolveOptions.lastLayers = config.lastLayers;
  }
  const resolved = resolveLoRATargets(model, resolveOptions);
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
  resetPeakMemory();
  using model = await loadCausalLM(source);
  const padTokenId = readPadTokenId(tokenizer);
  const before = evaluateSupervisionDatasetLoss(
    model,
    data.supervisionEval,
    padTokenId,
    args.batchSize,
  );
  const appliedLoRA = applyTrainingLoRA(model, "lora");
  const trainableModule = expectTrainableModule(model);
  const parameterCounts = countParameterElements(trainableModule);
  const averageTrainingLoss = runSupervisionTrainingSteps(
    model,
    data.supervisionTrain,
    padTokenId,
    args.batchSize,
    args.steps,
    args.seed,
    1e-4,
  );
  const adapterCheck = await runAdapterCheck(
    "lora",
    source,
    source,
    model,
    tokenizer,
    profile,
    data,
    args,
  );
  const merged = mergeLoRAInModule(trainableModule);
  const after = evaluateSupervisionDatasetLoss(
    model,
    data.supervisionEval,
    padTokenId,
    args.batchSize,
  );
  const sample = sampleText(model, tokenizer, profile, data.samplePromptMessages);
  const memory = memoryReport();

  return {
    stage: "lora",
    evalLoss: summarizeMetric(before, after),
    averageTrainingLoss,
    sampleText: sample,
    targets: appliedLoRA.targets,
    parameterCounts,
    memory,
    adapterCheck,
    notes: [
      `preset=${appliedLoRA.preset}`,
      `target_count=${appliedLoRA.targets.length}`,
      `merged_targets=${merged.targets.length}`,
      `skipped=${merged.skipped.length}`,
      `adapter_reloaded_targets=${adapterCheck.reloadedMergeTargets.length}`,
      `trainable_parameters=${parameterCounts.trainable}`,
      `total_parameters=${parameterCounts.total}`,
      `peak_memory_bytes=${memory.peakBytes}`,
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
  resetPeakMemory();
  using model = await loadCausalLM(qloraSource);
  const padTokenId = readPadTokenId(tokenizer);
  const before = evaluateSupervisionDatasetLoss(
    model,
    data.supervisionEval,
    padTokenId,
    args.batchSize,
  );
  const appliedLoRA = applyTrainingLoRA(model, "qlora");
  const trainableModule = expectTrainableModule(model);
  const parameterCounts = countParameterElements(trainableModule);
  const averageTrainingLoss = runSupervisionTrainingSteps(
    model,
    data.supervisionTrain,
    padTokenId,
    args.batchSize,
    args.steps,
    args.seed + 1,
    1e-4,
  );
  const adapterCheck = await runAdapterCheck(
    "qlora",
    qloraSource,
    source,
    model,
    tokenizer,
    profile,
    data,
    args,
  );
  const merged = mergeLoRAInModule(trainableModule);
  const after = evaluateSupervisionDatasetLoss(
    model,
    data.supervisionEval,
    padTokenId,
    args.batchSize,
  );
  for (const target of merged.targets) {
    assertQuantizedBasePreserved(trainableModule, target);
  }
  const sample = sampleText(model, tokenizer, profile, data.samplePromptMessages);
  const memory = memoryReport();

  return {
    stage: "qlora",
    evalLoss: summarizeMetric(before, after),
    averageTrainingLoss,
    sampleText: sample,
    targets: appliedLoRA.targets,
    parameterCounts,
    memory,
    adapterCheck,
    notes: [
      `preset=${appliedLoRA.preset}`,
      `target_count=${appliedLoRA.targets.length}`,
      `merged_targets=${merged.targets.length}`,
      `adapter_reloaded_targets=${adapterCheck.reloadedMergeTargets.length}`,
      "quantized_base_preserved=true",
      `trainable_parameters=${parameterCounts.trainable}`,
      `total_parameters=${parameterCounts.total}`,
      `peak_memory_bytes=${memory.peakBytes}`,
      `train_examples=${data.supervisionTrain.length}`,
      `eval_examples=${data.supervisionEval.length}`,
    ],
  };
}

export async function runSFTStage(
  source: string,
  tokenizer: LoadedAssets["tokenizer"],
  profile: InteractionProfile,
  data: PreparedTrainingProofData,
  args: TrainingProofArgs,
): Promise<StageReport> {
  resetPeakMemory();
  using model = await loadCausalLM(source);
  const trainableModule = expectTrainableModule(model);
  const parameterCounts = countParameterElements(trainableModule);
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
  const sample = sampleText(model, tokenizer, profile, data.samplePromptMessages);
  const memory = memoryReport();

  return {
    stage: "sft",
    evalLoss: summarizeMetric(before, after),
    averageTrainingLoss,
    sampleText: sample,
    parameterCounts,
    memory,
    notes: [
      "dense_model=true",
      `trainable_parameters=${parameterCounts.trainable}`,
      `total_parameters=${parameterCounts.total}`,
      `peak_memory_bytes=${memory.peakBytes}`,
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
  resetPeakMemory();
  const dpoConfig = resolveDPOProofConfig(args.dpoProfile);
  const padTokenId = readPadTokenId(tokenizer);
  const trained = await (async () => {
    using policyModel = await loadCausalLM(source);
    using referenceModel = await loadCausalLM(source);
    const beforeLoss = evaluatePreferenceDatasetLoss(
      policyModel,
      referenceModel,
      data.preferenceEval,
      padTokenId,
      args.batchSize,
      dpoConfig.beta,
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
    const trainableModule = expectTrainableModule(policyModel);
    const parameterCounts = countParameterElements(trainableModule);
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
    const savedAdapterCheck = await saveAdapterForCheck(
      "dpo",
      source,
      policyModel,
      tokenizer,
      profile,
      data,
      args,
    );
    const merged = mergeLoRAInModule(trainableModule);
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
      dpoConfig.beta,
    );
    const sample = sampleText(policyModel, tokenizer, profile, data.samplePromptMessages);
    return {
      beforeLoss,
      beforeMetrics,
      appliedLoRA,
      parameterCounts,
      averageTrainingLoss,
      savedAdapterCheck,
      merged,
      afterLoss,
      afterMetrics,
      sample,
    };
  })();
  const adapterCheck = await completeAdapterCheck(
    source,
    trained.savedAdapterCheck,
    tokenizer,
    profile,
    data,
  );
  const memory = memoryReport();

  return {
    stage: "dpo",
    evalLoss: summarizeMetric(trained.beforeLoss, trained.afterLoss),
    rewardAccuracy: summarizeMetric(
      trained.beforeMetrics.rewardAccuracy,
      trained.afterMetrics.rewardAccuracy,
    ),
    rewardMargin: summarizeMetric(
      trained.beforeMetrics.rewardMargin,
      trained.afterMetrics.rewardMargin,
    ),
    chosenReward: summarizeMetric(
      trained.beforeMetrics.chosenReward,
      trained.afterMetrics.chosenReward,
    ),
    rejectedReward: summarizeMetric(
      trained.beforeMetrics.rejectedReward,
      trained.afterMetrics.rejectedReward,
    ),
    chosenLogProb: summarizeMetric(
      trained.beforeMetrics.chosenLogProb,
      trained.afterMetrics.chosenLogProb,
    ),
    rejectedLogProb: summarizeMetric(
      trained.beforeMetrics.rejectedLogProb,
      trained.afterMetrics.rejectedLogProb,
    ),
    rawPreferenceAccuracy: summarizeMetric(
      trained.beforeMetrics.rawPreferenceAccuracy,
      trained.afterMetrics.rawPreferenceAccuracy,
    ),
    averageTrainingLoss: trained.averageTrainingLoss,
    sampleText: trained.sample,
    targets: trained.appliedLoRA.targets,
    parameterCounts: trained.parameterCounts,
    memory,
    adapterCheck,
    notes: [
      "reference_model=frozen_copy",
      `dpo_profile=${dpoConfig.profile}`,
      `preset=${trained.appliedLoRA.preset}`,
      `target_count=${trained.appliedLoRA.targets.length}`,
      `merged_targets=${trained.merged.targets.length}`,
      `adapter_reloaded_targets=${adapterCheck.reloadedMergeTargets.length}`,
      `rank=${dpoConfig.rank}`,
      `alpha=${dpoConfig.alpha}`,
      `dropout=${dpoConfig.dropout}`,
      `learning_rate=${dpoConfig.learningRate}`,
      `beta=${dpoConfig.beta}`,
      dpoConfig.lastLayers === null ? "last_layers=all" : `last_layers=${dpoConfig.lastLayers}`,
      `trainable_parameters=${trained.parameterCounts.trainable}`,
      `total_parameters=${trained.parameterCounts.total}`,
      `peak_memory_bytes=${memory.peakBytes}`,
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
