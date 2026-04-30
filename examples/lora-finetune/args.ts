import type { CausalLMAdapterFormat, LoRATargetPreset } from "@mlxts/transformers";
import { join } from "path";

const DEFAULT_SOURCE = "meta-llama/Llama-3.2-1B-Instruct";
const DEFAULT_DATASET_SOURCE = "huggingface";
const DEFAULT_MODE = "lora";
const DEFAULT_TRAIN_LIMIT = 64;
const DEFAULT_EVAL_LIMIT = 16;
const DEFAULT_BATCH_SIZE = 4;
const DEFAULT_STEPS = 8;
const DEFAULT_MAX_SEQUENCE_LENGTH = 1024;
const DEFAULT_SEED = 7;

/** Default Hugging Face dataset used by the LoRA example. */
export const ULTRACHAT_DATASET = "HuggingFaceH4/ultrachat_200k";

export type FinetuneMode = "lora" | "qlora";
export type DatasetSource = "tiny" | "huggingface" | "jsonl";

export type FinetuneCommand = { kind: "help" } | { kind: "run"; options: FinetuneArgs };

/** CLI arguments for the readable LoRA example surface. */
export type FinetuneArgs = {
  source: string;
  mode: FinetuneMode;
  preset: LoRATargetPreset;
  adapterFormat: CausalLMAdapterFormat;
  datasetSource: DatasetSource;
  datasetJsonlPath: string | null;
  trainLimit: number;
  evalLimit: number;
  batchSize: number;
  steps: number;
  maxSequenceLength: number;
  seed: number;
  outputDir: string;
  quantizedOutputDir: string;
  reportPath: string;
};

/** Report emitted by the readable LoRA example surface. */
export type FinetuneReport = {
  source: string;
  mode: FinetuneMode;
  preset: LoRATargetPreset;
  adapterFormat: CausalLMAdapterFormat;
  datasetSource: DatasetSource;
  trainLimit: number;
  evalLimit: number;
  batchSize: number;
  steps: number;
  maxSequenceLength: number;
  outputDir: string;
  adapterDir: string;
  metrics: {
    evalLossBefore: number;
    evalLossAfter: number;
    averageTrainingLoss: number;
    targetCount: number;
  };
  samplePrompt: readonly { role: "system" | "user" | "assistant"; content: string }[];
  sampleText: {
    trained: string;
    reloaded: string;
    merged: string;
  };
};

/** Usage error raised before any model runtime work starts. */
export class FinetuneUsageError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "FinetuneUsageError";
  }
}

function safeSource(source: string): string {
  return source.replace(/[^A-Za-z0-9._-]+/g, "-").replace(/-+/g, "-");
}

export function defaultOutputDir(source: string): string {
  return join(process.cwd(), ".tmp", "lora-finetune", safeSource(source));
}

export function defaultQuantizedOutputDir(source: string): string {
  return join(process.cwd(), ".tmp", "lora-finetune", `${safeSource(source)}-4bit`);
}

export function defaultReportPath(source: string): string {
  return join(process.cwd(), ".tmp", "lora-finetune", `${safeSource(source)}-report.json`);
}

function readValue(flag: string, value: string | undefined): string {
  if (value === undefined || value.trim() === "" || value.startsWith("--")) {
    throw new FinetuneUsageError(`lora-finetune: missing value for ${flag}.`);
  }
  return value;
}

function readPositiveInteger(flag: string, value: string | undefined): number {
  const parsed = Number(readValue(flag, value));
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new FinetuneUsageError(`lora-finetune: ${flag} expects a positive integer.`);
  }
  return parsed;
}

function readMode(value: string): FinetuneMode {
  if (value === "lora" || value === "qlora") {
    return value;
  }
  throw new FinetuneUsageError(`lora-finetune: unknown mode "${value}".`);
}

function readPreset(value: string): LoRATargetPreset {
  if (value === "attention" || value === "attention+mlp" || value === "all-linear") {
    return value;
  }
  throw new FinetuneUsageError(`lora-finetune: unknown preset "${value}".`);
}

function readDatasetSource(value: string): DatasetSource {
  if (value === "tiny" || value === "huggingface" || value === "jsonl") {
    return value;
  }
  throw new FinetuneUsageError(`lora-finetune: unknown dataset source "${value}".`);
}

function readAdapterFormat(value: string): CausalLMAdapterFormat {
  if (value === "mlxts" || value === "peft") {
    return value;
  }
  throw new FinetuneUsageError(`lora-finetune: unknown adapter format "${value}".`);
}

/** Parse the LoRA example command before model runtime work begins. */
export function parseFinetuneCommand(argv: readonly string[]): FinetuneCommand {
  if (argv.includes("--help") || argv.includes("-h")) {
    return { kind: "help" };
  }
  return { kind: "run", options: parseArgs(argv) };
}

/** Parse the LoRA example CLI arguments into one explicit configuration object. */
export function parseArgs(argv: readonly string[]): FinetuneArgs {
  let source = DEFAULT_SOURCE;
  let mode: FinetuneMode = DEFAULT_MODE;
  let preset: LoRATargetPreset | null = null;
  let adapterFormat: CausalLMAdapterFormat = "mlxts";
  let datasetSource: DatasetSource = DEFAULT_DATASET_SOURCE;
  let datasetJsonlPath: string | null = null;
  let trainLimit = DEFAULT_TRAIN_LIMIT;
  let evalLimit = DEFAULT_EVAL_LIMIT;
  let batchSize = DEFAULT_BATCH_SIZE;
  let steps = DEFAULT_STEPS;
  let maxSequenceLength = DEFAULT_MAX_SEQUENCE_LENGTH;
  let seed = DEFAULT_SEED;
  let outputDir = defaultOutputDir(source);
  let quantizedOutputDir = defaultQuantizedOutputDir(source);
  let reportPath = defaultReportPath(source);

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === undefined) {
      continue;
    }

    switch (arg) {
      case "--source":
        source = readValue(arg, argv[index + 1]);
        outputDir = defaultOutputDir(source);
        quantizedOutputDir = defaultQuantizedOutputDir(source);
        reportPath = defaultReportPath(source);
        index += 1;
        break;
      case "--mode":
        mode = readMode(readValue(arg, argv[index + 1]));
        index += 1;
        break;
      case "--preset":
        preset = readPreset(readValue(arg, argv[index + 1]));
        index += 1;
        break;
      case "--adapter-format":
        adapterFormat = readAdapterFormat(readValue(arg, argv[index + 1]));
        index += 1;
        break;
      case "--dataset-source":
        datasetSource = readDatasetSource(readValue(arg, argv[index + 1]));
        index += 1;
        break;
      case "--dataset-jsonl":
        datasetJsonlPath = readValue(arg, argv[index + 1]);
        index += 1;
        break;
      case "--train-limit":
        trainLimit = readPositiveInteger(arg, argv[index + 1]);
        index += 1;
        break;
      case "--eval-limit":
        evalLimit = readPositiveInteger(arg, argv[index + 1]);
        index += 1;
        break;
      case "--batch-size":
        batchSize = readPositiveInteger(arg, argv[index + 1]);
        index += 1;
        break;
      case "--steps":
        steps = readPositiveInteger(arg, argv[index + 1]);
        index += 1;
        break;
      case "--max-seq-len":
        maxSequenceLength = readPositiveInteger(arg, argv[index + 1]);
        index += 1;
        break;
      case "--seed":
        seed = readPositiveInteger(arg, argv[index + 1]);
        index += 1;
        break;
      case "--output-dir":
        outputDir = readValue(arg, argv[index + 1]);
        index += 1;
        break;
      case "--quantized-output":
        quantizedOutputDir = readValue(arg, argv[index + 1]);
        index += 1;
        break;
      case "--report":
        reportPath = readValue(arg, argv[index + 1]);
        index += 1;
        break;
      default:
        throw new FinetuneUsageError(`lora-finetune: unknown argument "${arg}".`);
    }
  }

  if (datasetSource === "jsonl" && datasetJsonlPath === null) {
    throw new FinetuneUsageError(
      "lora-finetune: --dataset-jsonl is required when --dataset-source jsonl.",
    );
  }

  return {
    source,
    mode,
    preset: preset ?? (mode === "qlora" ? "all-linear" : "attention"),
    adapterFormat,
    datasetSource,
    datasetJsonlPath,
    trainLimit,
    evalLimit,
    batchSize,
    steps,
    maxSequenceLength,
    seed,
    outputDir,
    quantizedOutputDir,
    reportPath,
  };
}
