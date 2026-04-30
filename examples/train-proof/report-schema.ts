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
  if (value.sampleText !== undefined) {
    report.sampleText = readString(value.sampleText, `${context}.sampleText`);
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
