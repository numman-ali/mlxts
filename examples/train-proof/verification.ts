import type { TrainingProofArgs, TrainingProofDatasetSource, TrainingProofStageName } from "./args";
import type {
  MetricPair,
  StageReport,
  TrainingProofReport,
  TrainingProofVerification,
  TrainingProofVerificationCheck,
} from "./types";

export { parseTrainingProofReport } from "./report-schema";

export type TrainingProofVerificationOptions = {
  expectedSource?: string;
  expectedDatasetSource?: TrainingProofDatasetSource;
  expectedTrainLimit?: number;
  expectedEvalLimit?: number;
  expectedBatchSize?: number;
  expectedSteps?: number;
  expectedMaxSequenceLength?: number;
  expectedSeed?: number;
  requiredStages?: readonly TrainingProofStageName[];
  expectedDPOProfile?: TrainingProofArgs["dpoProfile"];
  requireMetricImprovement?: boolean;
};

function isStageName(value: unknown): value is TrainingProofStageName {
  return value === "lora" || value === "qlora" || value === "sft" || value === "dpo";
}

function check(
  checks: TrainingProofVerificationCheck[],
  id: string,
  passed: boolean,
  message: string,
): void {
  checks.push({ id, passed, message });
}

function noteValue(report: StageReport, key: string): string | undefined {
  const prefix = `${key}=`;
  const note = report.notes.find((entry) => entry.startsWith(prefix));
  return note?.slice(prefix.length);
}

function hasNote(report: StageReport, note: string): boolean {
  return report.notes.includes(note);
}

function readPositiveNote(report: StageReport, key: string): number | undefined {
  const value = noteValue(report, key);
  if (value === undefined) {
    return undefined;
  }
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : undefined;
}

function readNumberNote(report: StageReport, key: string): number | undefined {
  const value = noteValue(report, key);
  if (value === undefined) {
    return undefined;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function isKnownPreset(value: string | undefined): boolean {
  return value === "attention" || value === "attention+mlp" || value === "all-linear";
}

function metricPairIsConsistent(metric: MetricPair): boolean {
  const expected = metric.after - metric.before;
  return Math.abs(metric.delta - expected) <= 1e-9;
}

function metricPairIsImproved(metric: MetricPair): boolean {
  return metric.after <= metric.before;
}

function numbersAreClose(left: number, right: number): boolean {
  return Math.abs(left - right) <= 1e-9;
}

function trainingStepLossesAreSequential(report: StageReport): boolean {
  return report.trainingStepLosses?.every((entry, index) => entry.step === index + 1) === true;
}

function trainingStepLossesAreFinite(report: StageReport): boolean {
  return (
    report.trainingStepLosses?.every((entry) => Number.isFinite(entry.loss) && entry.loss >= 0) ===
    true
  );
}

function averageMatchesTrainingTrace(report: StageReport): boolean {
  if (
    report.averageTrainingLoss === undefined ||
    report.trainingStepLosses === undefined ||
    report.trainingStepLosses.length === 0
  ) {
    return false;
  }
  const total = report.trainingStepLosses.reduce((sum, entry) => sum + entry.loss, 0);
  return numbersAreClose(report.averageTrainingLoss, total / report.trainingStepLosses.length);
}

function isMetricPair(value: unknown): value is MetricPair {
  return (
    typeof value === "object" &&
    value !== null &&
    !Array.isArray(value) &&
    "before" in value &&
    typeof value.before === "number" &&
    Number.isFinite(value.before) &&
    "after" in value &&
    typeof value.after === "number" &&
    Number.isFinite(value.after) &&
    "delta" in value &&
    typeof value.delta === "number" &&
    Number.isFinite(value.delta)
  );
}

function checkOptionalMetric(
  checks: TrainingProofVerificationCheck[],
  stage: StageReport,
  key: keyof StageReport,
): void {
  const metric = stage[key];
  const id = `${stage.stage}.${String(key)}`;
  if (metric === undefined) {
    check(checks, id, false, `${stage.stage} missing ${String(key)}.`);
    return;
  }
  if (!isMetricPair(metric)) {
    check(checks, id, false, `${stage.stage} ${String(key)} is not a metric pair.`);
    return;
  }
  check(
    checks,
    id,
    metricPairIsConsistent(metric),
    `${stage.stage} ${String(key)} records consistent before/after/delta values.`,
  );
}

function checkStageCommon(
  checks: TrainingProofVerificationCheck[],
  report: StageReport,
  options: TrainingProofVerificationOptions,
  reportSteps: number,
): void {
  check(
    checks,
    `${report.stage}.known`,
    isStageName(report.stage),
    `${report.stage} is a known proof stage.`,
  );
  checkOptionalMetric(checks, report, "evalLoss");
  if (options.requireMetricImprovement && report.evalLoss !== undefined) {
    check(
      checks,
      `${report.stage}.eval_loss_improves`,
      metricPairIsImproved(report.evalLoss),
      `${report.stage} held-out loss does not increase.`,
    );
  }
  check(
    checks,
    `${report.stage}.training_loss`,
    report.averageTrainingLoss !== undefined && report.averageTrainingLoss > 0,
    `${report.stage} records a positive average training loss.`,
  );
  check(
    checks,
    `${report.stage}.training_step_losses`,
    report.trainingStepLosses !== undefined && report.trainingStepLosses.length === reportSteps,
    `${report.stage} records one training loss per configured optimizer step.`,
  );
  check(
    checks,
    `${report.stage}.training_step_sequence`,
    trainingStepLossesAreSequential(report),
    `${report.stage} records step losses in one-based optimizer-step order.`,
  );
  check(
    checks,
    `${report.stage}.training_step_loss_values`,
    trainingStepLossesAreFinite(report),
    `${report.stage} records finite non-negative step losses.`,
  );
  check(
    checks,
    `${report.stage}.training_loss_average`,
    averageMatchesTrainingTrace(report),
    `${report.stage} average training loss matches the step-loss trace.`,
  );
  check(
    checks,
    `${report.stage}.training_steps_note`,
    readPositiveNote(report, "training_steps") === reportSteps,
    `${report.stage} notes record the configured optimizer-step count.`,
  );
  check(
    checks,
    `${report.stage}.sample`,
    typeof report.sampleText === "string" && report.sampleText.trim() !== "",
    `${report.stage} records a non-empty sample.`,
  );
  check(
    checks,
    `${report.stage}.parameter_counts`,
    report.parameterCounts !== undefined &&
      report.parameterCounts.total > 0 &&
      report.parameterCounts.trainable >= 0 &&
      report.parameterCounts.trainable <= report.parameterCounts.total,
    `${report.stage} records valid total and trainable parameter counts.`,
  );
  check(
    checks,
    `${report.stage}.memory_peak`,
    report.memory !== undefined && report.memory.peakBytes > 0,
    `${report.stage} records peak MLX memory evidence.`,
  );
  check(
    checks,
    `${report.stage}.train_examples`,
    readPositiveNote(report, "train_examples") !== undefined,
    `${report.stage} records positive train example evidence.`,
  );
  check(
    checks,
    `${report.stage}.eval_examples`,
    readPositiveNote(report, "eval_examples") !== undefined,
    `${report.stage} records positive eval example evidence.`,
  );
}

function checkAdapterStage(
  checks: TrainingProofVerificationCheck[],
  report: StageReport,
  expectedPreset: string,
): void {
  const targetCount = readPositiveNote(report, "target_count");
  const mergedTargets = readPositiveNote(report, "merged_targets");
  const preset = noteValue(report, "preset");
  check(
    checks,
    `${report.stage}.preset`,
    preset === expectedPreset && isKnownPreset(preset),
    `${report.stage} uses the ${expectedPreset} LoRA preset.`,
  );
  check(
    checks,
    `${report.stage}.target_count`,
    targetCount !== undefined,
    `${report.stage} records positive LoRA target count.`,
  );
  check(
    checks,
    `${report.stage}.merged_targets`,
    mergedTargets !== undefined && mergedTargets === targetCount,
    `${report.stage} merges every selected LoRA target.`,
  );
  check(
    checks,
    `${report.stage}.targets`,
    report.targets !== undefined &&
      targetCount !== undefined &&
      report.targets.length === targetCount,
    `${report.stage} records every selected LoRA target path.`,
  );
  check(
    checks,
    `${report.stage}.adapter_reload`,
    report.adapterCheck !== undefined &&
      targetCount !== undefined &&
      report.adapterCheck.reloadedMergeTargets.length === targetCount &&
      report.adapterCheck.trainedSampleText.trim() !== "" &&
      report.adapterCheck.reloadedSampleText.trim() !== "" &&
      report.adapterCheck.reloadedMergedSampleText.trim() !== "" &&
      report.adapterCheck.reloadedSampleText === report.adapterCheck.trainedSampleText &&
      report.adapterCheck.reloadedMergedSampleText === report.adapterCheck.reloadedSampleText,
    `${report.stage} saves, reloads, samples, and merges the trained adapter without changing greedy output.`,
  );
}

function checkDPOStage(
  checks: TrainingProofVerificationCheck[],
  report: StageReport,
  options: TrainingProofVerificationOptions,
): void {
  for (const key of [
    "rewardAccuracy",
    "rewardMargin",
    "chosenReward",
    "rejectedReward",
    "chosenLogProb",
    "rejectedLogProb",
    "rawPreferenceAccuracy",
  ] satisfies (keyof StageReport)[]) {
    checkOptionalMetric(checks, report, key);
  }
  check(
    checks,
    "dpo.reference_model",
    hasNote(report, "reference_model=frozen_copy"),
    "dpo records frozen reference model evidence.",
  );
  if (options.expectedDPOProfile !== undefined) {
    check(
      checks,
      "dpo.profile",
      noteValue(report, "dpo_profile") === options.expectedDPOProfile,
      `dpo profile is ${options.expectedDPOProfile}.`,
    );
  }
  const profile = options.expectedDPOProfile ?? noteValue(report, "dpo_profile");
  if (profile !== "canonical" && profile !== "handbook") {
    check(checks, "dpo.profile_known", false, "dpo records a known proof profile.");
    return;
  }
  const expected =
    profile === "handbook"
      ? {
          rank: 32,
          alpha: 16,
          dropout: 0.05,
          learningRate: 1e-5,
          beta: 0.01,
          lastLayers: "all",
        }
      : {
          rank: 8,
          alpha: 16,
          dropout: 0,
          learningRate: 5e-5,
          beta: 0.1,
          lastLayers: "2",
        };
  check(
    checks,
    "dpo.rank",
    readNumberNote(report, "rank") === expected.rank,
    `dpo ${profile} rank matches the proof recipe.`,
  );
  check(
    checks,
    "dpo.alpha",
    readNumberNote(report, "alpha") === expected.alpha,
    `dpo ${profile} alpha matches the proof recipe.`,
  );
  check(
    checks,
    "dpo.dropout",
    readNumberNote(report, "dropout") === expected.dropout,
    `dpo ${profile} dropout matches the proof recipe.`,
  );
  check(
    checks,
    "dpo.learning_rate",
    readNumberNote(report, "learning_rate") === expected.learningRate,
    `dpo ${profile} learning rate matches the proof recipe.`,
  );
  check(
    checks,
    "dpo.beta",
    readNumberNote(report, "beta") === expected.beta,
    `dpo ${profile} beta matches the proof recipe.`,
  );
  check(
    checks,
    "dpo.last_layers",
    noteValue(report, "last_layers") === expected.lastLayers,
    `dpo ${profile} layer targeting matches the proof recipe.`,
  );
}

function expectedDPOPreset(report: StageReport, options: TrainingProofVerificationOptions): string {
  if (options.expectedDPOProfile === "handbook") {
    return "attention+mlp";
  }
  if (options.expectedDPOProfile === "canonical") {
    return "attention";
  }
  return noteValue(report, "preset") ?? "";
}

function checkStageSpecific(
  checks: TrainingProofVerificationCheck[],
  report: StageReport,
  options: TrainingProofVerificationOptions,
): void {
  if (report.stage === "lora") {
    checkAdapterStage(checks, report, "attention");
    return;
  }
  if (report.stage === "qlora") {
    checkAdapterStage(checks, report, "all-linear");
    check(
      checks,
      "qlora.quantized_base",
      hasNote(report, "quantized_base_preserved=true"),
      "qlora records quantized base preservation after merge.",
    );
    return;
  }
  if (report.stage === "sft") {
    check(checks, "sft.dense_model", hasNote(report, "dense_model=true"), "sft uses dense model.");
    return;
  }
  if (report.stage === "dpo") {
    checkAdapterStage(checks, report, expectedDPOPreset(report, options));
    checkDPOStage(checks, report, options);
  }
}

function checkExpectedNumber(
  checks: TrainingProofVerificationCheck[],
  id: string,
  actual: number,
  expected: number | undefined,
): void {
  if (expected === undefined) {
    return;
  }
  check(checks, id, actual === expected, `${id} is ${expected}.`);
}

function checkReportShape(
  checks: TrainingProofVerificationCheck[],
  report: TrainingProofReport,
  options: TrainingProofVerificationOptions,
): void {
  if (options.expectedSource !== undefined) {
    check(
      checks,
      "source",
      report.source === options.expectedSource,
      "source matches proof input.",
    );
  }
  if (options.expectedDatasetSource !== undefined) {
    check(
      checks,
      "dataset_source",
      report.datasetSource === options.expectedDatasetSource,
      "dataset source matches proof input.",
    );
  }
  check(
    checks,
    "adapter_output_dir",
    report.adapterOutputDir.trim() !== "",
    "report records the adapter output directory.",
  );
  checkExpectedNumber(checks, "train_limit", report.trainLimit, options.expectedTrainLimit);
  checkExpectedNumber(checks, "eval_limit", report.evalLimit, options.expectedEvalLimit);
  checkExpectedNumber(checks, "batch_size", report.batchSize, options.expectedBatchSize);
  checkExpectedNumber(checks, "steps", report.steps, options.expectedSteps);
  checkExpectedNumber(
    checks,
    "max_sequence_length",
    report.maxSequenceLength,
    options.expectedMaxSequenceLength,
  );
  checkExpectedNumber(checks, "seed", report.seed, options.expectedSeed);
  check(checks, "stages.non_empty", report.stages.length > 0, "report includes proof stages.");

  const observedStages = new Set(report.stages.map((stage) => stage.stage));
  for (const stage of options.requiredStages ?? []) {
    check(
      checks,
      `required_stage.${stage}`,
      observedStages.has(stage),
      `report includes required ${stage} stage.`,
    );
  }
  check(
    checks,
    "stages.unique",
    observedStages.size === report.stages.length,
    "report does not contain duplicate stages.",
  );

  check(
    checks,
    "data_notes.dataset_source",
    report.dataNotes.includes(`dataset_source=${report.datasetSource}`),
    "data notes record the dataset source.",
  );
  if (report.datasetSource === "huggingface") {
    check(
      checks,
      "data_notes.supervision_dataset",
      report.dataNotes.some((note) => note.startsWith("supervision_dataset=")),
      "data notes record the supervision dataset.",
    );
    check(
      checks,
      "data_notes.preference_dataset",
      report.dataNotes.some((note) => note.startsWith("preference_dataset=")),
      "data notes record the preference dataset.",
    );
  }
}

export function verifyTrainingProofReport(
  report: TrainingProofReport,
  options: TrainingProofVerificationOptions = {},
): TrainingProofVerification {
  const checks: TrainingProofVerificationCheck[] = [];
  checkReportShape(checks, report, options);
  for (const stage of report.stages) {
    checkStageCommon(checks, stage, options, report.steps);
    checkStageSpecific(checks, stage, options);
  }
  return {
    passed: checks.every((entry) => entry.passed),
    checks,
  };
}

export function assertTrainingProofReport(
  report: TrainingProofReport,
  options: TrainingProofVerificationOptions = {},
): TrainingProofVerification {
  const verification = verifyTrainingProofReport(report, options);
  if (!verification.passed) {
    const failures = verification.checks
      .filter((entry) => !entry.passed)
      .map((entry) => `- ${entry.id}: ${entry.message}`)
      .join("\n");
    throw new Error(`Training proof verification failed:\n${failures}`);
  }
  return verification;
}

export function verificationOptionsFromArgs(
  args: TrainingProofArgs,
): TrainingProofVerificationOptions {
  return {
    expectedSource: args.source,
    expectedDatasetSource: args.datasetSource,
    expectedTrainLimit: args.trainLimit,
    expectedEvalLimit: args.evalLimit,
    expectedBatchSize: args.batchSize,
    expectedSteps: args.steps,
    expectedMaxSequenceLength: args.maxSequenceLength,
    expectedSeed: args.seed,
    requiredStages: args.stages,
    expectedDPOProfile: args.dpoProfile,
  };
}
