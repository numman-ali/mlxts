import type { CausalLMAdapterFormat, LoRATargetPreset } from "@mlxts/transformers";

import type { DatasetSource, FinetuneMode, FinetuneReport } from "./args";

export type FinetuneReportVerificationOptions = {
  expectedMode?: FinetuneMode;
  expectedAdapterFormat?: CausalLMAdapterFormat;
  requireLossNotWorse?: boolean;
};

export type FinetuneReportVerification = {
  checks: readonly string[];
};

function fail(message: string): never {
  throw new Error(`lora finetune report: ${message}`);
}

function readRecord(value: unknown, path: string): object {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    fail(`${path} expected a JSON object.`);
  }
  return value;
}

function field(record: object, name: string): unknown {
  return Reflect.get(record, name);
}

function readString(value: unknown, path: string): string {
  if (typeof value !== "string" || value.trim() === "") {
    fail(`${path} expected a non-empty string.`);
  }
  return value;
}

function readFiniteNumber(value: unknown, path: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    fail(`${path} expected a finite number.`);
  }
  return value;
}

function readPositiveInteger(value: unknown, path: string): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value <= 0) {
    fail(`${path} expected a positive integer.`);
  }
  return value;
}

function readNonNegativeInteger(value: unknown, path: string): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value < 0) {
    fail(`${path} expected a non-negative integer.`);
  }
  return value;
}

function readBoolean(value: unknown, path: string): boolean {
  if (typeof value !== "boolean") {
    fail(`${path} expected a boolean.`);
  }
  return value;
}

function readMode(value: unknown, path: string): FinetuneMode {
  if (value === "lora" || value === "qlora") {
    return value;
  }
  fail(`${path} expected "lora" or "qlora".`);
}

function readPreset(value: unknown, path: string): LoRATargetPreset {
  if (value === "attention" || value === "attention+mlp" || value === "all-linear") {
    return value;
  }
  fail(`${path} expected a supported LoRA target preset.`);
}

function readAdapterFormat(value: unknown, path: string): CausalLMAdapterFormat {
  if (value === "mlxts" || value === "peft") {
    return value;
  }
  fail(`${path} expected "mlxts" or "peft".`);
}

function readDatasetSource(value: unknown, path: string): DatasetSource {
  if (value === "tiny" || value === "huggingface" || value === "jsonl") {
    return value;
  }
  fail(`${path} expected "tiny", "huggingface", or "jsonl".`);
}

function readMessageRole(value: unknown, path: string): "system" | "user" | "assistant" {
  if (value === "system" || value === "user" || value === "assistant") {
    return value;
  }
  fail(`${path} expected a chat role.`);
}

function readSamplePrompt(value: unknown, path: string): FinetuneReport["samplePrompt"] {
  if (!Array.isArray(value) || value.length === 0) {
    fail(`${path} expected a non-empty message array.`);
  }
  return value.map((message, index) => {
    const record = readRecord(message, `${path}[${index}]`);
    return {
      role: readMessageRole(field(record, "role"), `${path}[${index}].role`),
      content: readString(field(record, "content"), `${path}[${index}].content`),
    };
  });
}

function readStringArray(value: unknown, path: string): readonly string[] {
  if (!Array.isArray(value) || value.length === 0) {
    fail(`${path} expected a non-empty string array.`);
  }
  return value.map((item, index) => readString(item, `${path}[${index}]`));
}

function readParameterCounts(value: unknown, path: string): FinetuneReport["parameterCounts"] {
  const record = readRecord(value, path);
  return {
    total: readPositiveInteger(field(record, "total"), `${path}.total`),
    trainable: readPositiveInteger(field(record, "trainable"), `${path}.trainable`),
  };
}

function readMemory(value: unknown, path: string): FinetuneReport["memory"] {
  const record = readRecord(value, path);
  return {
    peakBytes: readPositiveInteger(field(record, "peakBytes"), `${path}.peakBytes`),
  };
}

function readPreparationStats(value: unknown, path: string): FinetuneReport["dataStats"]["train"] {
  const record = readRecord(value, path);
  return {
    kept: readPositiveInteger(field(record, "kept"), `${path}.kept`),
    skippedMalformed: readNonNegativeInteger(
      field(record, "skippedMalformed"),
      `${path}.skippedMalformed`,
    ),
    skippedLong: readNonNegativeInteger(field(record, "skippedLong"), `${path}.skippedLong`),
  };
}

function readDataStats(value: unknown, path: string): FinetuneReport["dataStats"] {
  const record = readRecord(value, path);
  return {
    train: readPreparationStats(field(record, "train"), `${path}.train`),
    eval: readPreparationStats(field(record, "eval"), `${path}.eval`),
  };
}

function readMetrics(value: unknown, path: string): FinetuneReport["metrics"] {
  const record = readRecord(value, path);
  return {
    evalLossBefore: readFiniteNumber(field(record, "evalLossBefore"), `${path}.evalLossBefore`),
    evalLossAfter: readFiniteNumber(field(record, "evalLossAfter"), `${path}.evalLossAfter`),
    averageTrainingLoss: readFiniteNumber(
      field(record, "averageTrainingLoss"),
      `${path}.averageTrainingLoss`,
    ),
    targetCount: readPositiveInteger(field(record, "targetCount"), `${path}.targetCount`),
  };
}

function readSampleText(value: unknown, path: string): FinetuneReport["sampleText"] {
  const record = readRecord(value, path);
  return {
    trained: readString(field(record, "trained"), `${path}.trained`),
    reloaded: readString(field(record, "reloaded"), `${path}.reloaded`),
    merged: readString(field(record, "merged"), `${path}.merged`),
  };
}

function readAdapterCheck(value: unknown, path: string): FinetuneReport["adapterCheck"] {
  const record = readRecord(value, path);
  const qloraValue = field(record, "qloraQuantizedBasePreserved");
  if (qloraValue !== null && typeof qloraValue !== "boolean") {
    fail(`${path}.qloraQuantizedBasePreserved expected a boolean or null.`);
  }
  return {
    reloadedMatchesTrained: readBoolean(
      field(record, "reloadedMatchesTrained"),
      `${path}.reloadedMatchesTrained`,
    ),
    qloraQuantizedBasePreserved: qloraValue,
  };
}

/** Parse a LoRA fine-tuning report and reject malformed or partial artifacts. */
export function parseFinetuneReport(value: unknown): FinetuneReport {
  const record = readRecord(value, "report");
  return {
    source: readString(field(record, "source"), "report.source"),
    mode: readMode(field(record, "mode"), "report.mode"),
    preset: readPreset(field(record, "preset"), "report.preset"),
    adapterFormat: readAdapterFormat(field(record, "adapterFormat"), "report.adapterFormat"),
    datasetSource: readDatasetSource(field(record, "datasetSource"), "report.datasetSource"),
    trainLimit: readPositiveInteger(field(record, "trainLimit"), "report.trainLimit"),
    evalLimit: readPositiveInteger(field(record, "evalLimit"), "report.evalLimit"),
    batchSize: readPositiveInteger(field(record, "batchSize"), "report.batchSize"),
    steps: readPositiveInteger(field(record, "steps"), "report.steps"),
    maxSequenceLength: readPositiveInteger(
      field(record, "maxSequenceLength"),
      "report.maxSequenceLength",
    ),
    outputDir: readString(field(record, "outputDir"), "report.outputDir"),
    adapterDir: readString(field(record, "adapterDir"), "report.adapterDir"),
    metrics: readMetrics(field(record, "metrics"), "report.metrics"),
    targetPaths: readStringArray(field(record, "targetPaths"), "report.targetPaths"),
    parameterCounts: readParameterCounts(
      field(record, "parameterCounts"),
      "report.parameterCounts",
    ),
    memory: readMemory(field(record, "memory"), "report.memory"),
    dataStats: readDataStats(field(record, "dataStats"), "report.dataStats"),
    adapterCheck: readAdapterCheck(field(record, "adapterCheck"), "report.adapterCheck"),
    samplePrompt: readSamplePrompt(field(record, "samplePrompt"), "report.samplePrompt"),
    sampleText: readSampleText(field(record, "sampleText"), "report.sampleText"),
  };
}

/** Verify that a LoRA fine-tuning report proves reloadable adapter behavior. */
export function assertFinetuneReport(
  report: FinetuneReport,
  options: FinetuneReportVerificationOptions = {},
): FinetuneReportVerification {
  const checks: string[] = [];
  const check = (condition: boolean, message: string): void => {
    if (!condition) {
      fail(`verification failed: ${message}`);
    }
    checks.push(message);
  };

  check(report.metrics.targetCount > 0, "resolved at least one trainable LoRA target.");
  check(
    report.metrics.targetCount === report.targetPaths.length,
    "target count matches selected target paths.",
  );
  check(
    report.parameterCounts.trainable > 0 &&
      report.parameterCounts.trainable <= report.parameterCounts.total,
    "parameter counts include positive trainable parameters within total parameters.",
  );
  check(report.memory.peakBytes > 0, "peak MLX memory evidence is recorded.");
  check(
    report.dataStats.train.kept === report.trainLimit,
    "training data preparation kept requested rows.",
  );
  check(
    report.dataStats.eval.kept === report.evalLimit,
    "evaluation data preparation kept requested rows.",
  );
  check(report.metrics.evalLossBefore >= 0, "held-out loss before training is non-negative.");
  check(report.metrics.evalLossAfter >= 0, "held-out loss after training is non-negative.");
  check(report.metrics.averageTrainingLoss >= 0, "average training loss is non-negative.");
  check(report.sampleText.trained.length > 0, "trained sample text is non-empty.");
  check(report.sampleText.reloaded.length > 0, "reloaded sample text is non-empty.");
  check(report.sampleText.merged.length > 0, "merged sample text is non-empty.");
  check(
    report.sampleText.trained === report.sampleText.reloaded,
    "saved adapters reload to the same deterministic sample text.",
  );
  check(report.adapterCheck.reloadedMatchesTrained, "report records adapter reload equality.");
  if (report.mode === "qlora") {
    check(
      report.adapterCheck.qloraQuantizedBasePreserved === true,
      "QLoRA merge preserves quantized base layers.",
    );
  }

  if (options.expectedMode !== undefined) {
    check(report.mode === options.expectedMode, `mode is ${options.expectedMode}.`);
  }
  if (options.expectedAdapterFormat !== undefined) {
    check(
      report.adapterFormat === options.expectedAdapterFormat,
      `adapter format is ${options.expectedAdapterFormat}.`,
    );
  }
  if (options.requireLossNotWorse === true) {
    check(
      report.metrics.evalLossAfter <= report.metrics.evalLossBefore,
      "held-out loss does not increase.",
    );
  }

  return { checks };
}
