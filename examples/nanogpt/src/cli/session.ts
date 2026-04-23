import { random } from "@mlxts/core";
import { loadText } from "@mlxts/data";
import type { AdamW } from "@mlxts/optimizers";
import { CharTokenizer } from "@mlxts/tokenizers";

import {
  applyCheckpoint,
  type CheckpointKind,
  loadCheckpoint,
  restoreAdamWFromCheckpoint,
} from "../checkpoint";
import { GPT_SMALL, GPT_TINY, type ModelPreset, resolveConfig } from "../config";
import { GPT } from "../model/gpt";
import { initializeGPT } from "../model/init";
import { createDefaultAdamW } from "../optimizer-defaults";
import type { TrainConfig } from "../train";
import {
  getBooleanFlag,
  getFlag,
  getNonNegativeNumberFlag,
  getNullableNonNegativeIntegerFlag,
  getNullablePositiveNumberFlag,
  getNumberFlag,
  UserError,
} from "./shared";

export type TrainingSession = {
  model: GPT;
  optimizer: AdamW;
  tokenizer: CharTokenizer;
  config: ReturnType<typeof resolveConfig>;
  trainConfig: TrainConfig;
  presetName: string;
  text: string;
  resumeStep: number;
  checkpointSource?: string;
};

export type TrainDefaults = {
  batchSize: number;
  evalInterval: number;
  evalSteps: number;
  gradAccumSteps: number;
  logInterval: number;
  maxSteps: number;
  minLearningRate: number;
  resumeInterval: number;
  snapshotInterval: number;
  warmupSteps: number;
};

export type AutoTrainingPolicy = {
  patience: number | null;
  minDelta: number;
  bestValLoss: number | null;
  bestCheckpointStep: number | null;
  bestCheckpointPath?: string | undefined;
  consecutiveBadEvals: number;
  stopReason?: string | undefined;
};

const PRESETS: Record<string, ModelPreset> = {
  "gpt-tiny": GPT_TINY,
  "gpt-small": GPT_SMALL,
};

const TINY_TRAIN_DEFAULTS: TrainDefaults = {
  maxSteps: 500,
  batchSize: 4,
  gradAccumSteps: 1,
  evalInterval: 100,
  evalSteps: 10,
  logInterval: 10,
  snapshotInterval: 250,
  resumeInterval: 1000,
  warmupSteps: 100,
  minLearningRate: 3e-5,
};

const SMALL_TRAIN_DEFAULTS: TrainDefaults = {
  maxSteps: 500,
  batchSize: 1,
  gradAccumSteps: 8,
  evalInterval: 100,
  evalSteps: 10,
  logInterval: 10,
  snapshotInterval: 250,
  resumeInterval: 1000,
  warmupSteps: 100,
  minLearningRate: 3e-5,
};

function applyPresetOverrides(preset: ModelPreset, flags: Map<string, string>): ModelPreset {
  const gradientCheckpointing = getBooleanFlag(flags, "gradient-checkpointing");
  if (gradientCheckpointing === undefined) {
    return preset;
  }
  return {
    ...preset,
    gradientCheckpointing,
  };
}

function parsePreset(flags: Map<string, string>): { name: string; preset: ModelPreset } {
  const presetName = getFlag(flags, "preset", "gpt-tiny");
  const preset = presetName === undefined ? undefined : PRESETS[presetName];
  if (presetName === undefined || preset === undefined) {
    throw new UserError(
      `Unknown preset "${presetName ?? "<missing>"}". Available: ${Object.keys(PRESETS).join(", ")}`,
    );
  }
  return { name: presetName, preset: applyPresetOverrides(preset, flags) };
}

export function trainDefaultsForPresetName(presetName: string): TrainDefaults {
  return presetName === "gpt-small" ? SMALL_TRAIN_DEFAULTS : TINY_TRAIN_DEFAULTS;
}

export function inferPresetName(config: ReturnType<typeof resolveConfig>): string {
  if (
    config.nLayer === GPT_TINY.nLayer &&
    config.nHead === GPT_TINY.nHead &&
    config.nEmbd === GPT_TINY.nEmbd &&
    config.blockSize === GPT_TINY.blockSize &&
    config.dropout === GPT_TINY.dropout
  ) {
    return "gpt-tiny";
  }
  if (
    config.nLayer === GPT_SMALL.nLayer &&
    config.nHead === GPT_SMALL.nHead &&
    config.nEmbd === GPT_SMALL.nEmbd &&
    config.blockSize === GPT_SMALL.blockSize &&
    config.dropout === GPT_SMALL.dropout
  ) {
    return "gpt-small";
  }
  return "checkpoint";
}

export function trainDefaultsForConfig(config: ReturnType<typeof resolveConfig>): TrainDefaults {
  const presetName = inferPresetName(config);
  if (presetName === "gpt-tiny" || presetName === "gpt-small") {
    return trainDefaultsForPresetName(presetName);
  }
  return config.gradientCheckpointing ? SMALL_TRAIN_DEFAULTS : TINY_TRAIN_DEFAULTS;
}

export function checkpointPath(
  directory: string,
  presetName: string,
  kind: CheckpointKind,
  step: number,
): string {
  if (kind === "best") {
    return `${directory}/${presetName}-best`;
  }
  return `${directory}/${presetName}-${kind}-step-${step}`;
}

export function createAutoTrainingPolicy(
  flags: Map<string, string>,
  checkpointDir: string | undefined,
  presetName: string,
): AutoTrainingPolicy {
  return {
    patience: getNullableNonNegativeIntegerFlag(flags, "early-stop-patience", 8),
    minDelta: getNonNegativeNumberFlag(flags, "early-stop-min-delta", 0.02),
    bestValLoss: null,
    bestCheckpointStep: null,
    bestCheckpointPath:
      checkpointDir === undefined
        ? undefined
        : checkpointPath(checkpointDir, presetName, "best", 0),
    consecutiveBadEvals: 0,
  };
}

function trainConfigFromFlags(
  defaults: TrainDefaults,
  flags: Map<string, string>,
  startStep = 0,
): TrainConfig {
  const maxSteps = getNumberFlag(flags, "max-steps", defaults.maxSteps);
  const warmupFallback = Math.max(0, Math.min(defaults.warmupSteps, maxSteps - 1));
  return {
    startStep,
    maxSteps,
    batchSize: getNumberFlag(flags, "batch-size", defaults.batchSize),
    learningRate: getNumberFlag(flags, "lr", 3e-4),
    weightDecay: getNumberFlag(flags, "weight-decay", 0.1),
    warmupSteps: getNumberFlag(flags, "warmup-steps", warmupFallback),
    minLearningRate: getNumberFlag(flags, "min-lr", defaults.minLearningRate),
    gradAccumSteps: getNumberFlag(flags, "grad-accum", defaults.gradAccumSteps),
    evalInterval: getNumberFlag(flags, "eval-interval", defaults.evalInterval),
    evalSteps: getNumberFlag(flags, "eval-steps", defaults.evalSteps),
    logInterval: getNumberFlag(flags, "log-interval", defaults.logInterval),
    maxGradNorm: getNullablePositiveNumberFlag(flags, "max-grad-norm", 1),
    seed: getNumberFlag(flags, "seed", 42),
  };
}

async function createFreshTrainingSession(flags: Map<string, string>): Promise<TrainingSession> {
  const { name: presetName, preset } = parsePreset(flags);
  const trainConfig = trainConfigFromFlags(trainDefaultsForPresetName(presetName), flags);

  random.seed(trainConfig.seed);
  const dataPath = flags.get("data");
  const text = await loadText(dataPath !== undefined ? { path: dataPath } : {});
  const tokenizer = CharTokenizer.fromText(text);
  const config = resolveConfig(preset, tokenizer.vocabSize);
  const model = new GPT(config);
  const optimizer = createDefaultAdamW(trainConfig.learningRate, trainConfig.weightDecay);

  initializeGPT(model, config);
  return {
    model,
    optimizer,
    tokenizer,
    config,
    trainConfig,
    presetName,
    text,
    resumeStep: 0,
  };
}

async function createResumedTrainingSession(
  checkpointPath: string,
  flags: Map<string, string>,
): Promise<TrainingSession> {
  const checkpoint = loadCheckpoint(checkpointPath);
  if (checkpoint.kind !== "resume") {
    throw new UserError(
      `Checkpoint "${checkpointPath}" is a ${checkpoint.kind} checkpoint and cannot be resumed`,
    );
  }

  const optimizerData = checkpoint.optimizer;
  if (optimizerData === undefined) {
    throw new UserError(
      `Checkpoint "${checkpointPath}" does not contain optimizer state and cannot be resumed`,
    );
  }
  if (optimizerData.step !== checkpoint.step) {
    throw new UserError(
      `Checkpoint "${checkpointPath}" has mismatched optimizer step ${optimizerData.step} and model step ${checkpoint.step}`,
    );
  }

  const checkpointConfig = checkpoint.metadata.config;
  const presetName = inferPresetName(checkpointConfig);
  const trainConfig = trainConfigFromFlags(
    trainDefaultsForConfig(checkpointConfig),
    flags,
    checkpoint.step,
  );
  random.seed(trainConfig.seed);

  const dataPath = flags.get("data");
  const text = await loadText(dataPath !== undefined ? { path: dataPath } : {});
  const tokenizer = CharTokenizer.fromVocab(checkpoint.metadata.tokenizer.chars);
  tokenizer.encode(text);

  const model = new GPT(checkpointConfig);
  applyCheckpoint(model, checkpoint);
  const optimizer = restoreAdamWFromCheckpoint(optimizerData);

  return {
    model,
    optimizer,
    tokenizer,
    config: checkpointConfig,
    trainConfig,
    presetName,
    text,
    resumeStep: checkpoint.step,
    checkpointSource: checkpointPath,
  };
}

async function createWarmStartTrainingSession(
  checkpointPath: string,
  flags: Map<string, string>,
): Promise<TrainingSession> {
  const checkpoint = loadCheckpoint(checkpointPath);
  const checkpointConfig = checkpoint.metadata.config;
  const presetName = inferPresetName(checkpointConfig);
  const trainConfig = trainConfigFromFlags(trainDefaultsForConfig(checkpointConfig), flags);
  random.seed(trainConfig.seed);

  const dataPath = flags.get("data");
  const text = await loadText(dataPath !== undefined ? { path: dataPath } : {});
  const tokenizer = CharTokenizer.fromVocab(checkpoint.metadata.tokenizer.chars);
  tokenizer.encode(text);

  const model = new GPT(checkpointConfig);
  applyCheckpoint(model, checkpoint);
  const optimizer = createDefaultAdamW(trainConfig.learningRate, trainConfig.weightDecay);

  return {
    model,
    optimizer,
    tokenizer,
    config: checkpointConfig,
    trainConfig,
    presetName,
    text,
    resumeStep: 0,
    checkpointSource: checkpointPath,
  };
}

function validateTrainSourceFlags(flags: Map<string, string>): {
  resumePath?: string;
  warmStartPath?: string;
} {
  const resumePath = flags.get("resume");
  const warmStartPath = flags.get("warm-start");
  if (resumePath !== undefined && warmStartPath !== undefined) {
    throw new UserError("Flags --resume and --warm-start are mutually exclusive");
  }
  if (
    (resumePath !== undefined || warmStartPath !== undefined) &&
    (flags.has("preset") || flags.has("gradient-checkpointing"))
  ) {
    throw new UserError(
      "Flags --preset/--gradient-checkpointing cannot be combined with --resume/--warm-start; checkpoint config is authoritative",
    );
  }

  const resolved: { resumePath?: string; warmStartPath?: string } = {};
  if (resumePath !== undefined) {
    resolved.resumePath = resumePath;
  }
  if (warmStartPath !== undefined) {
    resolved.warmStartPath = warmStartPath;
  }
  return resolved;
}

export async function createTrainingSession(flags: Map<string, string>): Promise<TrainingSession> {
  const { resumePath, warmStartPath } = validateTrainSourceFlags(flags);
  return resumePath !== undefined
    ? await createResumedTrainingSession(resumePath, flags)
    : warmStartPath !== undefined
      ? await createWarmStartTrainingSession(warmStartPath, flags)
      : await createFreshTrainingSession(flags);
}
