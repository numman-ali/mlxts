import type { TrainingProofDatasetSource } from "./args";
import type { MetricPair, StageReport, TrainingProofReport } from "./types";

type StageMetricKey =
  | "evalLoss"
  | "rewardAccuracy"
  | "rewardMargin"
  | "chosenReward"
  | "rejectedReward"
  | "chosenLogProb"
  | "rejectedLogProb"
  | "rawPreferenceAccuracy";

const STAGE_METRIC_KEYS: readonly StageMetricKey[] = [
  "evalLoss",
  "rewardAccuracy",
  "rewardMargin",
  "chosenReward",
  "rejectedReward",
  "chosenLogProb",
  "rejectedLogProb",
  "rawPreferenceAccuracy",
];

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function readString(value: unknown, context: string): string {
  if (typeof value !== "string") {
    throw new Error(`${context}: expected a string.`);
  }
  return value;
}

function readNumber(value: unknown, context: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${context}: expected a finite number.`);
  }
  return value;
}

function readPositiveInteger(value: unknown, context: string): number {
  const parsed = readNumber(value, context);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`${context}: expected a positive integer.`);
  }
  return parsed;
}

function readStringArray(value: unknown, context: string): string[] {
  if (!Array.isArray(value)) {
    throw new Error(`${context}: expected a string array.`);
  }
  return value.map((entry, index) => readString(entry, `${context}[${index}]`));
}

function readOptionalStringArray(value: unknown, context: string): string[] | undefined {
  return value === undefined ? undefined : readStringArray(value, context);
}

function readTrainingStepLosses(
  value: unknown,
  context: string,
): StageReport["trainingStepLosses"] | undefined {
  if (value === undefined) {
    return undefined;
  }
  if (!Array.isArray(value)) {
    throw new Error(`${context}: expected a step-loss array.`);
  }
  return value.map((entry, index) => {
    if (!isRecord(entry)) {
      throw new Error(`${context}[${index}]: expected a step-loss object.`);
    }
    return {
      step: readPositiveInteger(entry.step, `${context}[${index}].step`),
      loss: readNumber(entry.loss, `${context}[${index}].loss`),
    };
  });
}

function readNonNegativeInteger(value: unknown, context: string): number {
  const parsed = readNumber(value, context);
  if (!Number.isInteger(parsed) || parsed < 0) {
    throw new Error(`${context}: expected a non-negative integer.`);
  }
  return parsed;
}

function readDatasetSource(value: unknown, context: string): TrainingProofDatasetSource {
  if (value !== "tiny" && value !== "huggingface") {
    throw new Error(`${context}: expected tiny or huggingface.`);
  }
  return value;
}

function readMetricPair(value: unknown, context: string): MetricPair | undefined {
  if (value === undefined) {
    return undefined;
  }
  if (!isRecord(value)) {
    throw new Error(`${context}: expected a metric pair object.`);
  }
  return {
    before: readNumber(value.before, `${context}.before`),
    after: readNumber(value.after, `${context}.after`),
    delta: readNumber(value.delta, `${context}.delta`),
  };
}

function assignMetric(
  report: StageReport,
  key: StageMetricKey,
  metric: MetricPair | undefined,
): void {
  if (metric !== undefined) {
    report[key] = metric;
  }
}

function readParameterCounts(
  value: unknown,
  context: string,
): StageReport["parameterCounts"] | undefined {
  if (value === undefined) {
    return undefined;
  }
  if (!isRecord(value)) {
    throw new Error(`${context}: expected a parameter count object.`);
  }
  return {
    total: readPositiveInteger(value.total, `${context}.total`),
    trainable: readNonNegativeInteger(value.trainable, `${context}.trainable`),
  };
}

function readMemory(value: unknown, context: string): StageReport["memory"] | undefined {
  if (value === undefined) {
    return undefined;
  }
  if (!isRecord(value)) {
    throw new Error(`${context}: expected a memory object.`);
  }
  return {
    peakBytes: readPositiveInteger(value.peakBytes, `${context}.peakBytes`),
  };
}

function readAdapterCheck(
  value: unknown,
  context: string,
): StageReport["adapterCheck"] | undefined {
  if (value === undefined) {
    return undefined;
  }
  if (!isRecord(value)) {
    throw new Error(`${context}: expected an adapter check object.`);
  }
  return {
    directory: readString(value.directory, `${context}.directory`),
    reloadedMergeTargets: readStringArray(
      value.reloadedMergeTargets,
      `${context}.reloadedMergeTargets`,
    ),
    trainedSampleText: readString(value.trainedSampleText, `${context}.trainedSampleText`),
    reloadedSampleText: readString(value.reloadedSampleText, `${context}.reloadedSampleText`),
    reloadedMergedSampleText: readString(
      value.reloadedMergedSampleText,
      `${context}.reloadedMergedSampleText`,
    ),
  };
}

function readStageReport(value: unknown, context: string): StageReport {
  if (!isRecord(value)) {
    throw new Error(`${context}: expected a stage object.`);
  }

  const report: StageReport = {
    stage: readString(value.stage, `${context}.stage`),
    notes: readStringArray(value.notes, `${context}.notes`),
  };
  for (const key of STAGE_METRIC_KEYS) {
    assignMetric(report, key, readMetricPair(value[key], `${context}.${key}`));
  }
  if (value.averageTrainingLoss !== undefined) {
    report.averageTrainingLoss = readNumber(
      value.averageTrainingLoss,
      `${context}.averageTrainingLoss`,
    );
  }
  const trainingStepLosses = readTrainingStepLosses(
    value.trainingStepLosses,
    `${context}.trainingStepLosses`,
  );
  if (trainingStepLosses !== undefined) {
    report.trainingStepLosses = trainingStepLosses;
  }
  if (value.sampleText !== undefined) {
    report.sampleText = readString(value.sampleText, `${context}.sampleText`);
  }
  const targets = readOptionalStringArray(value.targets, `${context}.targets`);
  if (targets !== undefined) {
    report.targets = targets;
  }
  const parameterCounts = readParameterCounts(value.parameterCounts, `${context}.parameterCounts`);
  if (parameterCounts !== undefined) {
    report.parameterCounts = parameterCounts;
  }
  const memory = readMemory(value.memory, `${context}.memory`);
  if (memory !== undefined) {
    report.memory = memory;
  }
  const adapterCheck = readAdapterCheck(value.adapterCheck, `${context}.adapterCheck`);
  if (adapterCheck !== undefined) {
    report.adapterCheck = adapterCheck;
  }
  return report;
}

export function parseTrainingProofReport(value: unknown): TrainingProofReport {
  if (!isRecord(value)) {
    throw new Error("training proof report: expected a JSON object.");
  }
  if (!Array.isArray(value.stages)) {
    throw new Error("training proof report.stages: expected an array.");
  }

  return {
    source: readString(value.source, "training proof report.source"),
    quantizedOutputDir: readString(
      value.quantizedOutputDir,
      "training proof report.quantizedOutputDir",
    ),
    adapterOutputDir: readString(value.adapterOutputDir, "training proof report.adapterOutputDir"),
    datasetSource: readDatasetSource(value.datasetSource, "training proof report.datasetSource"),
    trainLimit: readPositiveInteger(value.trainLimit, "training proof report.trainLimit"),
    evalLimit: readPositiveInteger(value.evalLimit, "training proof report.evalLimit"),
    batchSize: readPositiveInteger(value.batchSize, "training proof report.batchSize"),
    steps: readPositiveInteger(value.steps, "training proof report.steps"),
    maxSequenceLength: readPositiveInteger(
      value.maxSequenceLength,
      "training proof report.maxSequenceLength",
    ),
    seed: readPositiveInteger(value.seed, "training proof report.seed"),
    dataNotes: readStringArray(value.dataNotes, "training proof report.dataNotes"),
    stages: value.stages.map((stage, index) =>
      readStageReport(stage, `training proof report.stages[${index}]`),
    ),
  };
}
