import { join } from "path";

/** Default official dense model for the training proof surface. */
export const DEFAULT_PROOF_MODEL = "meta-llama/Llama-3.2-1B-Instruct";

/** Default training dataset source for the proof runner. */
export const DEFAULT_PROOF_DATASET_SOURCE = "huggingface";

/** Default number of training examples to keep per task. */
export const DEFAULT_PROOF_TRAIN_LIMIT = 128;

/** Default number of held-out evaluation examples to keep per task. */
export const DEFAULT_PROOF_EVAL_LIMIT = 32;

/** Default batch size for the proof stages. */
export const DEFAULT_PROOF_BATCH_SIZE = 4;

/** Default optimizer step count for each proof stage. */
export const DEFAULT_PROOF_STEPS = 16;

/** Default token-length cap for dataset examples used in the proof. */
export const DEFAULT_PROOF_MAX_SEQUENCE_LENGTH = 1024;

/** Default deterministic seed for proof batching. */
export const DEFAULT_PROOF_SEED = 7;

export type TrainingProofDatasetSource = "tiny" | "huggingface";
export type TrainingProofStageName = "lora" | "qlora" | "sft" | "dpo";
export type DPOProofProfile = "canonical" | "handbook";
export type TrainingProofCommand = { kind: "help" } | { kind: "run"; options: TrainingProofArgs };

export const DEFAULT_PROOF_STAGES: TrainingProofStageName[] = ["lora", "qlora", "sft", "dpo"];
export const DEFAULT_DPO_PROFILE: DPOProofProfile = "canonical";

export class TrainingProofUsageError extends Error {}

/** CLI options for the training proof runner. */
export type TrainingProofArgs = {
  source: string;
  quantizedOutputDir: string;
  adapterOutputDir: string;
  reportPath: string;
  datasetSource: TrainingProofDatasetSource;
  trainLimit: number;
  evalLimit: number;
  batchSize: number;
  steps: number;
  maxSequenceLength: number;
  seed: number;
  stages: TrainingProofStageName[];
  dpoProfile: DPOProofProfile;
};

/** Default local directory for the repo-generated 4-bit snapshot. */
export function defaultQuantizedOutputDir(source = DEFAULT_PROOF_MODEL): string {
  const safeSource = source.replace(/[^A-Za-z0-9._-]+/g, "-").replace(/-+/g, "-");
  return join(process.cwd(), ".tmp", "training-proof", `${safeSource}-4bit`);
}

/** Default local directory for repo-generated proof adapters. */
export function defaultAdapterOutputDir(source = DEFAULT_PROOF_MODEL): string {
  const safeSource = source.replace(/[^A-Za-z0-9._-]+/g, "-").replace(/-+/g, "-");
  return join(process.cwd(), ".tmp", "training-proof", `${safeSource}-adapters`);
}

/** Default JSON report path for the training proof run. */
export function defaultReportPath(source = DEFAULT_PROOF_MODEL): string {
  const safeSource = source.replace(/[^A-Za-z0-9._-]+/g, "-").replace(/-+/g, "-");
  return join(process.cwd(), ".tmp", "training-proof", `${safeSource}-report.json`);
}

function readValue(flag: string, value: string | undefined): string {
  if (value === undefined || value.trim() === "" || value.startsWith("--")) {
    throw new TrainingProofUsageError(`Missing value for ${flag}.`);
  }
  return value;
}

function readPositiveInteger(flag: string, value: string | undefined): number {
  const parsed = Number(readValue(flag, value));
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new TrainingProofUsageError(`${flag} expects a positive integer.`);
  }
  return parsed;
}

function readDatasetSource(value: string): TrainingProofDatasetSource {
  if (value === "tiny" || value === "huggingface") {
    return value;
  }
  throw new TrainingProofUsageError(`Unknown dataset source: ${value}`);
}

function readStages(value: string): TrainingProofStageName[] {
  const parsed = value
    .split(",")
    .map((entry) => entry.trim())
    .filter((entry) => entry !== "");
  if (parsed.length === 0) {
    throw new TrainingProofUsageError("--stages expects a comma-separated list of proof stages.");
  }
  const unique: TrainingProofStageName[] = [];
  for (const stage of parsed) {
    if (stage !== "lora" && stage !== "qlora" && stage !== "sft" && stage !== "dpo") {
      throw new TrainingProofUsageError(`Unknown proof stage: ${stage}`);
    }
    if (!unique.includes(stage)) {
      unique.push(stage);
    }
  }
  return unique;
}

function readDPOProfile(value: string): DPOProofProfile {
  if (value === "canonical" || value === "handbook") {
    return value;
  }
  throw new TrainingProofUsageError(`Unknown DPO proof profile: ${value}`);
}

/** Parse the training proof command before model runtime work begins. */
export function parseTrainingProofCommand(argv: readonly string[]): TrainingProofCommand {
  if (argv.some((arg) => arg === "--help" || arg === "-h")) {
    return { kind: "help" };
  }
  return { kind: "run", options: parseTrainingProofArgs(argv) };
}

/** Parse the training proof CLI arguments. */
export function parseTrainingProofArgs(argv: readonly string[]): TrainingProofArgs {
  let source = DEFAULT_PROOF_MODEL;
  let quantizedOutputDir = defaultQuantizedOutputDir(source);
  let adapterOutputDir = defaultAdapterOutputDir(source);
  let reportPath = defaultReportPath(source);
  let datasetSource: TrainingProofDatasetSource = DEFAULT_PROOF_DATASET_SOURCE;
  let trainLimit = DEFAULT_PROOF_TRAIN_LIMIT;
  let evalLimit = DEFAULT_PROOF_EVAL_LIMIT;
  let batchSize = DEFAULT_PROOF_BATCH_SIZE;
  let steps = DEFAULT_PROOF_STEPS;
  let maxSequenceLength = DEFAULT_PROOF_MAX_SEQUENCE_LENGTH;
  let seed = DEFAULT_PROOF_SEED;
  let stages = [...DEFAULT_PROOF_STAGES];
  let dpoProfile: DPOProofProfile = DEFAULT_DPO_PROFILE;

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === undefined) {
      continue;
    }

    switch (arg) {
      case "--source":
        source = readValue(arg, argv[index + 1]);
        quantizedOutputDir = defaultQuantizedOutputDir(source);
        adapterOutputDir = defaultAdapterOutputDir(source);
        reportPath = defaultReportPath(source);
        index += 1;
        break;
      case "--quantized-output":
        quantizedOutputDir = readValue(arg, argv[index + 1]);
        index += 1;
        break;
      case "--adapter-output":
        adapterOutputDir = readValue(arg, argv[index + 1]);
        index += 1;
        break;
      case "--report":
        reportPath = readValue(arg, argv[index + 1]);
        index += 1;
        break;
      case "--dataset-source":
        datasetSource = readDatasetSource(readValue(arg, argv[index + 1]));
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
      case "--stages":
        stages = readStages(readValue(arg, argv[index + 1]));
        index += 1;
        break;
      case "--dpo-profile":
        dpoProfile = readDPOProfile(readValue(arg, argv[index + 1]));
        index += 1;
        break;
      default:
        throw new TrainingProofUsageError(`Unknown argument: ${arg}`);
    }
  }

  return {
    source,
    quantizedOutputDir,
    adapterOutputDir,
    reportPath,
    datasetSource,
    trainLimit,
    evalLimit,
    batchSize,
    steps,
    maxSequenceLength,
    seed,
    stages,
    dpoProfile,
  };
}
