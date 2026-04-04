#!/usr/bin/env bun
/**
 * nanogpt CLI — train GPT models and generate text.
 *
 * Usage:
 *   nanogpt train [options]
 *   nanogpt generate --checkpoint <path> [options]
 *
 * @module
 */

import { mkdirSync } from "fs";
import {
  type AdamW,
  clearMemoryCache,
  getMemoryStats,
  random,
  setCacheLimitBytes,
  setMemoryLimitBytes,
  setWiredLimitBytes,
} from "mlx-ts";
import { join } from "path";
import {
  applyCheckpoint,
  type CheckpointKind,
  loadCheckpoint,
  restoreAdamWFromCheckpoint,
  saveCheckpoint,
} from "./checkpoint";
import {
  estimateParameterCount,
  GPT_SMALL,
  GPT_TINY,
  type ModelPreset,
  resolveConfig,
} from "./config";
import { loadText, prepareData } from "./data";
import { type GenerateConfig, generate } from "./generate";
import { GPT } from "./model/gpt";
import { initializeGPT } from "./model/init";
import { createDefaultAdamW } from "./optimizer-defaults";
import { type RunControlCommand, readRunControl } from "./run/files";
import { saveModelSafetensors } from "./safetensors";
import { CharTokenizer } from "./tokenizer";
import { type TrainConfig, type TrainEvent, train } from "./train";

const USER_ERROR_EXIT_CODE = 1;
const SYSTEM_ERROR_EXIT_CODE = 2;

class UserError extends Error {}

const TRAIN_FLAG_ALLOWLIST = new Set([
  "preset",
  "gradient-checkpointing",
  "data",
  "max-steps",
  "batch-size",
  "grad-accum",
  "eval-interval",
  "eval-steps",
  "log-interval",
  "lr",
  "weight-decay",
  "max-grad-norm",
  "warmup-steps",
  "min-lr",
  "seed",
  "resume",
  "warm-start",
  "checkpoint-dir",
  "snapshot-interval",
  "resume-interval",
  "sample-interval",
  "sample-tokens",
  "early-stop-patience",
  "early-stop-min-delta",
  "memory-limit-mb",
  "cache-limit-mb",
  "wired-limit-mb",
  "run-dir",
  "json",
  "help",
]);

const GENERATE_FLAG_ALLOWLIST = new Set([
  "checkpoint",
  "prompt",
  "max-tokens",
  "temperature",
  "json",
  "help",
]);

const EXPORT_FLAG_ALLOWLIST = new Set(["checkpoint", "output", "help"]);

type JsonEnvelope = {
  timestamp: string;
  [key: string]: unknown;
};

type TrainingSession = {
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

function getBooleanFlag(flags: Map<string, string>, key: string): boolean | undefined {
  const raw = flags.get(key);
  if (raw === undefined) {
    return undefined;
  }
  if (raw === "true") {
    return true;
  }
  if (raw === "false") {
    return false;
  }
  throw new UserError(`Flag --${key} must be "true" or "false"`);
}

function parseArgs(argv: string[]): { command: string; flags: Map<string, string> } {
  const command = argv[2] ?? "help";
  const flags = new Map<string, string>();

  for (let index = 3; index < argv.length; index++) {
    const argument = argv[index];
    if (argument === undefined || !argument.startsWith("--")) {
      continue;
    }

    const key = argument.slice(2);
    const next = argv[index + 1];
    if (next !== undefined && !next.startsWith("--")) {
      flags.set(key, next);
      index += 1;
      continue;
    }

    flags.set(key, "true");
  }

  return { command, flags };
}

function validateKnownFlags(
  flags: Map<string, string>,
  allowed: ReadonlySet<string>,
  context: string,
): void {
  for (const key of flags.keys()) {
    if (!allowed.has(key)) {
      throw new UserError(`${context}: unknown flag --${key}`);
    }
  }
}

function getFlag(flags: Map<string, string>, key: string, fallback?: string): string | undefined {
  return flags.get(key) ?? fallback;
}

function getNumberFlag(flags: Map<string, string>, key: string, fallback: number): number {
  const raw = flags.get(key);
  if (raw === undefined) {
    return fallback;
  }

  const parsed = Number(raw);
  if (!Number.isFinite(parsed)) {
    throw new UserError(`Flag --${key} must be a finite number`);
  }
  return parsed;
}

function getNullablePositiveNumberFlag(
  flags: Map<string, string>,
  key: string,
  fallback: number | null,
): number | null {
  const raw = flags.get(key);
  if (raw === undefined) {
    return fallback;
  }
  if (raw === "none" || raw === "null") {
    return null;
  }

  const parsed = Number(raw);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new UserError(`Flag --${key} must be a positive number or "none"`);
  }
  return parsed;
}

function getNullableNonNegativeIntegerFlag(
  flags: Map<string, string>,
  key: string,
  fallback: number | null,
): number | null {
  const raw = flags.get(key);
  if (raw === undefined) {
    return fallback;
  }
  if (raw === "none" || raw === "null") {
    return null;
  }

  const parsed = Number(raw);
  if (!Number.isInteger(parsed) || parsed < 0) {
    throw new UserError(`Flag --${key} must be a non-negative integer or "none"`);
  }
  return parsed;
}

function getNonNegativeNumberFlag(
  flags: Map<string, string>,
  key: string,
  fallback: number,
): number {
  const raw = flags.get(key);
  if (raw === undefined) {
    return fallback;
  }

  const parsed = Number(raw);
  if (!Number.isFinite(parsed) || parsed < 0) {
    throw new UserError(`Flag --${key} must be a non-negative number`);
  }
  return parsed;
}

const PRESETS: Record<string, ModelPreset> = {
  "gpt-tiny": GPT_TINY,
  "gpt-small": GPT_SMALL,
};

type TrainDefaults = {
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

function trainDefaultsForPresetName(presetName: string): TrainDefaults {
  return presetName === "gpt-small" ? SMALL_TRAIN_DEFAULTS : TINY_TRAIN_DEFAULTS;
}

function trainDefaultsForConfig(config: ReturnType<typeof resolveConfig>): TrainDefaults {
  const presetName = inferPresetName(config);
  if (presetName === "gpt-tiny" || presetName === "gpt-small") {
    return trainDefaultsForPresetName(presetName);
  }
  return config.gradientCheckpointing ? SMALL_TRAIN_DEFAULTS : TINY_TRAIN_DEFAULTS;
}

function checkpointPath(
  directory: string,
  presetName: string,
  kind: CheckpointKind,
  step: number,
): string {
  if (kind === "best") {
    return join(directory, `${presetName}-best`);
  }
  return join(directory, `${presetName}-${kind}-step-${step}`);
}

function createAutoTrainingPolicy(
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

function samplePrompt(tokenizer: CharTokenizer): string {
  const vocab = tokenizer.vocab;
  return vocab.includes("\n") ? "\n" : (vocab[0] ?? "");
}

function printHelp(): void {
  process.stdout.write(`nanogpt — Train GPT models and generate text

Usage:
  nanogpt train [options]     Train a GPT model on text data
  nanogpt generate [options]  Generate text from a checkpoint
  nanogpt export [options]    Export model weights as safetensors

Train options:
  --preset <name>            Model preset: gpt-tiny (default), gpt-small
                             gpt-small enables gradient checkpointing by default
  --gradient-checkpointing <true|false>
                             Override the preset's gradient checkpointing setting
  --data <path>              Path to training text file (default: cached/downloaded Shakespeare)
  --max-steps <n>            Maximum training steps (default: preset-specific safe default)
  --batch-size <n>           Batch size (default: preset-specific safe default)
  --grad-accum <n>           Gradient accumulation steps (default: preset-specific safe default)
  --lr <n>                   Peak learning rate (default: 3e-4)
  --weight-decay <n>         Weight decay (default: 0.1)
  --seed <n>                 Training seed for MLX + batching (default: 42)
  --resume <path>            Resume training from a checkpoint directory
  --warm-start <path>        Initialize weights from a checkpoint and start with a fresh optimizer
  --checkpoint-dir <path>    Directory for periodic/final checkpoints (default: .nanogpt-checkpoints)
  --snapshot-interval <n>    Save snapshot checkpoints every N eval steps (default: 250)
  --resume-interval <n>      Save resumable checkpoints every N eval steps (default: 1000)
  --early-stop-patience <n|none>
                             Stop after N evals without meaningful val-loss improvement (default: 8)
  --early-stop-min-delta <n> Minimum val-loss improvement required to reset patience (default: 0.02)
  --memory-limit-mb <n>      Set the MLX allocator memory limit in MB
  --cache-limit-mb <n>       Set the MLX allocator cache limit in MB
  --wired-limit-mb <n>       Set the MLX wired-memory limit in MB
  --json                     Emit JSON events to stdout
  --help                     Show this help

Generate options:
  --checkpoint <path>        Path to checkpoint directory (required)
  --prompt <text>            Prompt text (default: newline)
  --max-tokens <n>           Tokens to generate (default: 500)
  --temperature <n>          Sampling temperature, 0=greedy (default: 0.8)
  --json                     Emit a JSON result object to stdout
  --help                     Show this help

Examples:
  nanogpt train --preset gpt-tiny
  nanogpt train --resume .nanogpt-checkpoints/gpt-small-resume-step-500
  nanogpt train --warm-start .nanogpt-checkpoints/gpt-small-snapshot-step-50 --max-steps 500
  nanogpt train --preset gpt-small --max-steps 5000 --grad-accum 8
  nanogpt generate --checkpoint .nanogpt-checkpoints/gpt-tiny-resume-step-500 --prompt "To be or"
  nanogpt export --checkpoint .nanogpt-checkpoints/gpt-small-resume-step-500 --output model.safetensors
`);
}

function printTrainHelp(): void {
  process.stdout.write(`nanogpt train — Train a GPT model on text data

Usage:
  nanogpt train [options]

Options:
  --preset <name>            Model preset: gpt-tiny (default), gpt-small
                             gpt-small enables gradient checkpointing by default
  --gradient-checkpointing <true|false>
                             Override the preset's gradient checkpointing setting
  --data <path>              Path to training text file (default: cached/downloaded Shakespeare)
  --max-steps <n>            Maximum training steps (default: preset-specific safe default)
  --batch-size <n>           Batch size (default: preset-specific safe default)
  --grad-accum <n>           Gradient accumulation steps (default: preset-specific safe default)
  --lr <n>                   Peak learning rate (default: 3e-4)
  --weight-decay <n>         Weight decay (default: 0.1)
  --max-grad-norm <n|none>   Global gradient clipping threshold (default: 1.0)
  --seed <n>                 Training seed for MLX + batching (default: 42)
  --resume <path>            Resume training from a checkpoint directory
  --warm-start <path>        Initialize weights from a checkpoint and start with a fresh optimizer
  --checkpoint-dir <path>    Directory for periodic/final checkpoints (default: .nanogpt-checkpoints)
  --snapshot-interval <n>    Save snapshot checkpoints every N eval steps (default: 250)
  --resume-interval <n>      Save resumable checkpoints every N eval steps (default: 1000)
  --sample-interval <n>      Emit a generated sample every N training steps (default: 0 for plain train, snapshot interval for supervised runs)
  --sample-tokens <n>        Number of tokens to generate per sample (default: 200)
  --early-stop-patience <n|none>
                             Stop after N evals without meaningful val-loss improvement (default: 8)
  --early-stop-min-delta <n> Minimum val-loss improvement required to reset patience (default: 0.02)
  --memory-limit-mb <n>      Set the MLX allocator memory limit in MB
  --cache-limit-mb <n>       Set the MLX allocator cache limit in MB
  --wired-limit-mb <n>       Set the MLX wired-memory limit in MB
  --json                     Emit JSON events to stdout
  --help                     Show this help
`);
}

function printGenerateHelp(): void {
  process.stdout.write(`nanogpt generate — Generate text from a checkpoint

Usage:
  nanogpt generate --checkpoint <path> [options]

Options:
  --checkpoint <path>        Path to checkpoint directory (required)
  --prompt <text>            Prompt text (default: newline)
  --max-tokens <n>           Tokens to generate (default: 500)
  --temperature <n>          Sampling temperature, 0=greedy (default: 0.8)
  --json                     Emit a JSON result object to stdout
  --help                     Show this help
`);
}

function printExportHelp(): void {
  process.stdout.write(`nanogpt export — Export model weights as safetensors

Usage:
  nanogpt export --checkpoint <path> --output <file>.safetensors

Options:
  --checkpoint <path>        Path to checkpoint directory (required)
  --output <path>            Output safetensors path (required)
  --help                     Show this help
`);
}

function emitJson(value: Record<string, unknown>): void {
  const envelope: JsonEnvelope = {
    timestamp: new Date().toISOString(),
    ...value,
  };
  process.stdout.write(`${JSON.stringify(envelope)}\n`);
}

function trainingTableHeader(): string {
  return (
    "  Step    Loss    Val     LR        Tokens/sec\n" +
    "  ──────────────────────────────────────────────\n"
  );
}

function formatStepEvent(event: Extract<TrainEvent, { type: "step" }>): string {
  return (
    `  ${String(event.step).padStart(5)}  ${event.loss.toFixed(4).padStart(7)}` +
    `                ${event.learningRate.toExponential(1).padStart(8)}  ${Math.round(
      event.tokensPerSec,
    )
      .toLocaleString()
      .padStart(10)}\n`
  );
}

function formatEvalEvent(event: Extract<TrainEvent, { type: "eval" }>): string {
  return `  ${String(event.step).padStart(5)}  ${event.trainLoss
    .toFixed(4)
    .padStart(7)}  ${event.valLoss.toFixed(4).padStart(7)}\n`;
}

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

  const presetName = inferPresetName(checkpoint.config);
  const trainConfig = trainConfigFromFlags(
    trainDefaultsForConfig(checkpoint.config),
    flags,
    checkpoint.step,
  );
  random.seed(trainConfig.seed);

  const dataPath = flags.get("data");
  const text = await loadText(dataPath !== undefined ? { path: dataPath } : {});
  const tokenizer = CharTokenizer.fromVocab(checkpoint.tokenizer.chars);
  tokenizer.encode(text);

  const model = new GPT(checkpoint.config);
  applyCheckpoint(model, checkpoint);
  const optimizer = restoreAdamWFromCheckpoint(optimizerData);

  return {
    model,
    optimizer,
    tokenizer,
    config: checkpoint.config,
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
  const presetName = inferPresetName(checkpoint.config);
  const trainConfig = trainConfigFromFlags(trainDefaultsForConfig(checkpoint.config), flags);
  random.seed(trainConfig.seed);

  const dataPath = flags.get("data");
  const text = await loadText(dataPath !== undefined ? { path: dataPath } : {});
  const tokenizer = CharTokenizer.fromVocab(checkpoint.tokenizer.chars);
  tokenizer.encode(text);

  const model = new GPT(checkpoint.config);
  applyCheckpoint(model, checkpoint);
  const optimizer = createDefaultAdamW(trainConfig.learningRate, trainConfig.weightDecay);

  return {
    model,
    optimizer,
    tokenizer,
    config: checkpoint.config,
    trainConfig,
    presetName,
    text,
    resumeStep: 0,
    checkpointSource: checkpointPath,
  };
}

function inferPresetName(config: ReturnType<typeof resolveConfig>): string {
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

type TelemetrySnapshot = ReturnType<typeof getMemoryStats>;
type CheckpointPolicy = {
  checkpointDir?: string | undefined;
  runDir?: string | undefined;
  snapshotEvery: number;
  resumeEvery: number;
};

type SamplingPolicy = {
  every: number;
  maxNewTokens: number;
};

type AutoTrainingPolicy = {
  patience: number | null;
  minDelta: number;
  bestValLoss: number | null;
  bestCheckpointStep: number | null;
  bestCheckpointPath?: string | undefined;
  consecutiveBadEvals: number;
  stopReason?: string | undefined;
};

function readTelemetry(): TelemetrySnapshot {
  return getMemoryStats();
}

function emitControlRequest(
  useJson: boolean,
  command: RunControlCommand,
  requestedAt: string,
): void {
  if (useJson) {
    emitJson({ type: "control", command, requestedAt });
    return;
  }
  if (command === "cancel") {
    process.stderr.write(
      "\nCancellation requested. Work since the latest resume checkpoint may be lost.\n",
    );
    return;
  }
  process.stderr.write("\nGraceful stop requested.\n");
}

function createStopController(
  useJson: boolean,
  runDir?: string,
): {
  cleanup: () => void;
  command: () => RunControlCommand | undefined;
  shouldStop: () => boolean;
} {
  let requested: RunControlCommand | undefined;
  let requestedAt: string | undefined;

  const rememberRequest = (command: RunControlCommand, at: string): void => {
    if (requested === command && requestedAt === at) {
      return;
    }
    if (requested !== undefined) {
      return;
    }
    requested = command;
    requestedAt = at;
    emitControlRequest(useJson, command, at);
  };

  const onSignal = (signal: NodeJS.Signals): void => {
    rememberRequest("stop", `${new Date().toISOString()} (${signal})`);
  };

  process.on("SIGINT", onSignal);
  process.on("SIGTERM", onSignal);

  return {
    command: () => requested,
    shouldStop: () => {
      if (runDir !== undefined) {
        const control = readRunControl(runDir);
        if (control !== undefined) {
          rememberRequest(control.command, control.requestedAt);
        }
      }
      return requested !== undefined;
    },
    cleanup: () => {
      process.off("SIGINT", onSignal);
      process.off("SIGTERM", onSignal);
    },
  };
}

function maybeClearAllocatorCache(): TelemetrySnapshot {
  const before = readTelemetry();
  if (before.cacheBytes <= Math.max(before.activeBytes * 2, 512 * 1024 * 1024)) {
    return before;
  }
  clearMemoryCache();
  return readTelemetry();
}

function emitCheckpoint(
  useJson: boolean,
  step: number,
  kind: CheckpointKind,
  path: string,
  telemetry: TelemetrySnapshot,
): void {
  if (useJson) {
    emitJson({
      type: "checkpoint",
      step,
      kind,
      path,
      activeMemoryBytes: telemetry.activeBytes,
      cacheMemoryBytes: telemetry.cacheBytes,
      peakMemoryBytes: telemetry.peakBytes,
      memoryLimitBytes: telemetry.limitBytes,
    });
    return;
  }
  process.stderr.write(`  ${kind} checkpoint saved: ${path}\n`);
}

function emitSample(
  session: TrainingSession,
  step: number,
  maxNewTokens: number,
  useJson: boolean,
): void {
  const { config, model, tokenizer } = session;
  const sample = generate(model, config, tokenizer, samplePrompt(tokenizer), {
    maxNewTokens,
    temperature: 0.8,
  });

  if (useJson) {
    emitJson({
      type: "sample",
      step,
      text: sample,
      maxNewTokens,
    });
    return;
  }

  process.stderr.write(`\n--- Sample @ step ${step} ---\n${sample}\n`);
}

function emitBestCheckpoint(
  useJson: boolean,
  step: number,
  valLoss: number,
  path: string | undefined,
): void {
  if (useJson) {
    emitJson({
      type: "best-checkpoint",
      step,
      valLoss,
      path,
    });
    return;
  }

  process.stderr.write(
    `  best checkpoint: val ${valLoss.toFixed(4)} at step ${step}${path === undefined ? "" : ` -> ${path}`}\n`,
  );
}

function emitEarlyStop(
  useJson: boolean,
  step: number,
  reason: string,
  autoTrainingPolicy: AutoTrainingPolicy,
): void {
  if (useJson) {
    emitJson({
      type: "early-stop",
      step,
      reason,
      bestValLoss: autoTrainingPolicy.bestValLoss,
      bestCheckpointStep: autoTrainingPolicy.bestCheckpointStep,
      bestCheckpointPath: autoTrainingPolicy.bestCheckpointPath,
      patience: autoTrainingPolicy.patience,
      minDelta: autoTrainingPolicy.minDelta,
      consecutiveBadEvals: autoTrainingPolicy.consecutiveBadEvals,
    });
    return;
  }

  process.stderr.write(`  early stop: ${reason}\n`);
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

async function createTrainingSession(flags: Map<string, string>): Promise<TrainingSession> {
  const { resumePath, warmStartPath } = validateTrainSourceFlags(flags);
  return resumePath !== undefined
    ? await createResumedTrainingSession(resumePath, flags)
    : warmStartPath !== undefined
      ? await createWarmStartTrainingSession(warmStartPath, flags)
      : await createFreshTrainingSession(flags);
}

function toBytesFromMb(flagKey: string, value: string | undefined): number | undefined {
  if (value === undefined) {
    return undefined;
  }
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new UserError(`Flag --${flagKey} must be a positive number`);
  }
  return Math.round(parsed * 1024 * 1024);
}

function applyRuntimeLimits(flags: Map<string, string>): void {
  const cacheLimitBytes = toBytesFromMb("cache-limit-mb", flags.get("cache-limit-mb"));
  const memoryLimitBytes = toBytesFromMb("memory-limit-mb", flags.get("memory-limit-mb"));
  const wiredLimitBytes = toBytesFromMb("wired-limit-mb", flags.get("wired-limit-mb"));

  if (cacheLimitBytes !== undefined) {
    setCacheLimitBytes(cacheLimitBytes);
  }
  if (memoryLimitBytes !== undefined) {
    setMemoryLimitBytes(memoryLimitBytes);
  }
  if (wiredLimitBytes !== undefined) {
    setWiredLimitBytes(wiredLimitBytes);
  }
}

function announceTrainingSession(
  session: TrainingSession,
  checkpointPolicy: CheckpointPolicy,
  samplingPolicy: SamplingPolicy,
  autoTrainingPolicy: AutoTrainingPolicy,
  useJson: boolean,
  parameterCount: number,
): void {
  const { config, presetName, text, tokenizer, trainConfig } = session;
  const telemetry = readTelemetry();
  if (useJson) {
    emitJson({
      type: "start",
      preset: presetName,
      config,
      params: parameterCount,
      vocabSize: tokenizer.vocabSize,
      maxSteps: trainConfig.maxSteps,
      batchSize: trainConfig.batchSize,
      gradAccumSteps: trainConfig.gradAccumSteps,
      maxGradNorm: trainConfig.maxGradNorm,
      warmupSteps: trainConfig.warmupSteps,
      startStep: trainConfig.startStep ?? 0,
      resumeFrom: session.checkpointSource,
      checkpointDir: checkpointPolicy.checkpointDir,
      snapshotEvery: checkpointPolicy.snapshotEvery,
      resumeEvery: checkpointPolicy.resumeEvery,
      sampleEvery: samplingPolicy.every,
      sampleTokens: samplingPolicy.maxNewTokens,
      earlyStopPatience: autoTrainingPolicy.patience,
      earlyStopMinDelta: autoTrainingPolicy.minDelta,
      pid: process.pid,
      activeMemoryBytes: telemetry.activeBytes,
      cacheMemoryBytes: telemetry.cacheBytes,
      peakMemoryBytes: telemetry.peakBytes,
      memoryLimitBytes: telemetry.limitBytes,
    });
    return;
  }

  process.stderr.write(
    `Training ${presetName} on ${text.length.toLocaleString()} chars (vocab: ${tokenizer.vocabSize}, params: ${parameterCount.toLocaleString()})\n\n`,
  );
  process.stderr.write(
    `Checkpoint policy: snapshot every ${checkpointPolicy.snapshotEvery} eval step(s), resume every ${checkpointPolicy.resumeEvery} eval step(s)\n`,
  );
  process.stderr.write(
    `Gradient clipping: ${trainConfig.maxGradNorm === null ? "disabled" : trainConfig.maxGradNorm}\n`,
  );
  process.stderr.write(
    `Sample output: ${samplingPolicy.every > 0 ? `every ${samplingPolicy.every} step(s), ${samplingPolicy.maxNewTokens} tokens` : "disabled"}\n\n`,
  );
  process.stderr.write(
    `Auto stop: ${
      autoTrainingPolicy.patience === null
        ? "disabled"
        : `patience ${autoTrainingPolicy.patience} eval(s), min delta ${autoTrainingPolicy.minDelta}`
    }\n\n`,
  );
  if (session.checkpointSource !== undefined) {
    const modeLabel = session.resumeStep > 0 ? "Resuming" : "Warm-starting";
    process.stderr.write(
      `${modeLabel} from ${session.checkpointSource} at step ${session.resumeStep}\n\n`,
    );
  }
  process.stderr.write(trainingTableHeader());
}

function createTrainEventHandler(options: {
  checkpointPolicy: CheckpointPolicy;
  samplingPolicy: SamplingPolicy;
  autoTrainingPolicy: AutoTrainingPolicy;
  requestAutoStop: (reason: string) => void;
  useJson: boolean;
  session: TrainingSession;
}): (event: TrainEvent) => void {
  const {
    checkpointPolicy,
    samplingPolicy,
    autoTrainingPolicy,
    requestAutoStop,
    session,
    useJson,
  } = options;
  const { config, model, optimizer, presetName, tokenizer } = session;

  function checkpointKindForStep(step: number): CheckpointKind | undefined {
    if (checkpointPolicy.resumeEvery > 0 && step % checkpointPolicy.resumeEvery === 0) {
      return "resume";
    }
    if (checkpointPolicy.snapshotEvery > 0 && step % checkpointPolicy.snapshotEvery === 0) {
      return "snapshot";
    }
    return undefined;
  }

  function emitEventOutput(event: TrainEvent): void {
    const telemetry = readTelemetry();
    if (useJson) {
      emitJson({
        ...event,
        activeMemoryBytes: telemetry.activeBytes,
        cacheMemoryBytes: telemetry.cacheBytes,
        peakMemoryBytes: telemetry.peakBytes,
        memoryLimitBytes: telemetry.limitBytes,
      });
      return;
    }

    if (event.type === "step") {
      process.stderr.write(formatStepEvent(event));
      return;
    }
    if (event.type === "progress") {
      return;
    }
    if (event.type === "eval") {
      process.stderr.write(formatEvalEvent(event));
      return;
    }
    if (event.type === "done") {
      process.stderr.write(`\nTraining complete. ${event.totalSteps} steps.\n`);
    }
  }

  function maybeSavePeriodicCheckpoint(event: TrainEvent): void {
    const checkpointDir = checkpointPolicy.checkpointDir;
    if (event.type !== "eval" || checkpointDir === undefined) {
      return;
    }

    const checkpointKind = checkpointKindForStep(event.step);
    if (checkpointKind === undefined) {
      return;
    }

    const path = checkpointPath(checkpointDir, presetName, checkpointKind, event.step);
    saveCheckpoint({
      model,
      optimizer: checkpointKind === "resume" ? optimizer : undefined,
      kind: checkpointKind,
      config,
      step: event.step,
      tokenizer,
      path,
    });
    emitCheckpoint(useJson, event.step, checkpointKind, path, maybeClearAllocatorCache());
  }

  function maybeEmitPeriodicSample(event: TrainEvent): void {
    if (
      samplingPolicy.every <= 0 ||
      event.type !== "step" ||
      event.step % samplingPolicy.every !== 0
    ) {
      return;
    }
    emitSample(session, event.step, samplingPolicy.maxNewTokens, useJson);
  }

  function maybeSaveBestCheckpoint(event: TrainEvent): void {
    if (event.type !== "eval") {
      return;
    }

    const bestValLoss = autoTrainingPolicy.bestValLoss;
    const improved =
      bestValLoss === null || event.valLoss <= bestValLoss - autoTrainingPolicy.minDelta;
    if (!improved) {
      return;
    }

    autoTrainingPolicy.bestValLoss = event.valLoss;
    autoTrainingPolicy.bestCheckpointStep = event.step;
    autoTrainingPolicy.consecutiveBadEvals = 0;
    autoTrainingPolicy.stopReason = undefined;

    if (autoTrainingPolicy.bestCheckpointPath !== undefined) {
      saveCheckpoint({
        model,
        kind: "best",
        config,
        step: event.step,
        tokenizer,
        path: autoTrainingPolicy.bestCheckpointPath,
      });
    }

    emitBestCheckpoint(useJson, event.step, event.valLoss, autoTrainingPolicy.bestCheckpointPath);
  }

  function maybeRequestEarlyStop(event: TrainEvent): void {
    if (event.type !== "eval" || autoTrainingPolicy.patience === null) {
      return;
    }
    if (autoTrainingPolicy.bestCheckpointStep === event.step) {
      return;
    }

    autoTrainingPolicy.consecutiveBadEvals += 1;
    if (autoTrainingPolicy.consecutiveBadEvals < autoTrainingPolicy.patience) {
      return;
    }

    const bestStep =
      autoTrainingPolicy.bestCheckpointStep === null
        ? "unknown"
        : String(autoTrainingPolicy.bestCheckpointStep);
    const bestVal =
      autoTrainingPolicy.bestValLoss === null
        ? "unknown"
        : autoTrainingPolicy.bestValLoss.toFixed(4);
    const reason =
      `validation loss did not improve by at least ${autoTrainingPolicy.minDelta} for ` +
      `${autoTrainingPolicy.consecutiveBadEvals} eval(s); best ${bestVal} at step ${bestStep}`;
    autoTrainingPolicy.stopReason = reason;
    emitEarlyStop(useJson, event.step, reason, autoTrainingPolicy);
    requestAutoStop(reason);
  }

  return (event: TrainEvent): void => {
    emitEventOutput(event);
    maybeSaveBestCheckpoint(event);
    maybeSavePeriodicCheckpoint(event);
    maybeEmitPeriodicSample(event);
    maybeRequestEarlyStop(event);
  };
}

function emitTrainingSample(
  session: TrainingSession,
  finalPath: string,
  summary: ReturnType<typeof train>,
  useJson: boolean,
): void {
  emitSample(session, summary.totalSteps, 200, useJson);
  if (useJson) {
    emitJson({
      type: "final-sample",
      preset: session.presetName,
      path: finalPath,
      summary,
    });
    return;
  }
}

function emitStoppedRun(
  finalPath: string,
  summary: ReturnType<typeof train>,
  autoTrainingPolicy: AutoTrainingPolicy,
  useJson: boolean,
): void {
  if (useJson) {
    emitJson({
      type: "stopped",
      path: finalPath,
      summary,
      reason: autoTrainingPolicy.stopReason,
      bestValLoss: autoTrainingPolicy.bestValLoss,
      bestCheckpointStep: autoTrainingPolicy.bestCheckpointStep,
      bestCheckpointPath: autoTrainingPolicy.bestCheckpointPath,
    });
    return;
  }

  process.stderr.write(
    `\nTraining stopped cleanly at step ${summary.totalSteps}. Final checkpoint: ${finalPath}${
      autoTrainingPolicy.stopReason === undefined
        ? ""
        : `\nReason: ${autoTrainingPolicy.stopReason}`
    }\n`,
  );
}

function emitCancelledRun(summary: ReturnType<typeof train>, useJson: boolean): void {
  if (useJson) {
    emitJson({
      type: "cancelled",
      summary,
    });
    return;
  }

  process.stderr.write(`\nTraining cancelled at step ${summary.totalSteps}.\n`);
}

async function runTrain(flags: Map<string, string>): Promise<void> {
  const checkpointDir = getFlag(flags, "checkpoint-dir", ".nanogpt-checkpoints");
  const runDir = flags.get("run-dir");
  const useJson = flags.has("json");
  const session = await createTrainingSession(flags);
  const { config, model, optimizer, presetName, text, tokenizer, trainConfig } = session;
  const defaults = trainDefaultsForConfig(config);
  const checkpointPolicy: CheckpointPolicy = {
    checkpointDir,
    runDir,
    snapshotEvery: getNumberFlag(flags, "snapshot-interval", defaults.snapshotInterval),
    resumeEvery: getNumberFlag(flags, "resume-interval", defaults.resumeInterval),
  };
  const samplingPolicy: SamplingPolicy = {
    every: flags.has("sample-interval")
      ? getNumberFlag(flags, "sample-interval", 0)
      : runDir !== undefined
        ? checkpointPolicy.snapshotEvery
        : 0,
    maxNewTokens: getNumberFlag(flags, "sample-tokens", 200),
  };
  const autoTrainingPolicy = createAutoTrainingPolicy(flags, checkpointDir, presetName);
  const parameterCount = estimateParameterCount(config);
  const { trainTokens, valTokens } = prepareData(tokenizer.encode(text), 0.9);
  const stopController = createStopController(useJson, runDir);
  let autoStopRequested = false;

  mkdirSync(checkpointDir ?? ".nanogpt-checkpoints", { recursive: true });
  applyRuntimeLimits(flags);
  announceTrainingSession(
    session,
    checkpointPolicy,
    samplingPolicy,
    autoTrainingPolicy,
    useJson,
    parameterCount,
  );
  const onEvent = createTrainEventHandler({
    useJson,
    checkpointPolicy,
    samplingPolicy,
    autoTrainingPolicy,
    requestAutoStop(reason) {
      if (autoStopRequested) {
        return;
      }
      autoStopRequested = true;
      autoTrainingPolicy.stopReason = reason;
    },
    session,
  });

  try {
    const summary = train({
      model,
      config,
      trainConfig,
      trainTokens,
      valTokens,
      optimizer,
      onEvent,
      shouldStop: () => stopController.shouldStop() || autoStopRequested,
    });

    const controlCommand = stopController.command();
    if (controlCommand === "cancel") {
      emitCancelledRun(summary, useJson);
      return;
    }

    const finalPath = checkpointPath(
      checkpointDir ?? ".nanogpt-checkpoints",
      presetName,
      "resume",
      summary.totalSteps,
    );
    saveCheckpoint({
      model,
      optimizer,
      kind: "resume",
      config,
      step: summary.totalSteps,
      tokenizer,
      path: finalPath,
    });
    emitCheckpoint(useJson, summary.totalSteps, "resume", finalPath, maybeClearAllocatorCache());
    if (summary.totalSteps < trainConfig.maxSteps) {
      emitStoppedRun(finalPath, summary, autoTrainingPolicy, useJson);
      return;
    }
    emitTrainingSample(session, finalPath, summary, useJson);
  } finally {
    stopController.cleanup();
    optimizer[Symbol.dispose]();
    model[Symbol.dispose]();
  }
}

function runGenerate(flags: Map<string, string>): void {
  const checkpoint = flags.get("checkpoint");
  if (checkpoint === undefined) {
    throw new UserError("Flag --checkpoint is required for generate");
  }

  const data = loadCheckpoint(checkpoint);
  const tokenizer = CharTokenizer.fromVocab(data.tokenizer.chars);
  const model = new GPT(data.config);

  try {
    applyCheckpoint(model, data);
    const prompt = getFlag(flags, "prompt", "\n") ?? "\n";
    const config: GenerateConfig = {
      maxNewTokens: getNumberFlag(flags, "max-tokens", 500),
      temperature: getNumberFlag(flags, "temperature", 0.8),
    };
    const text = generate(model, data.config, tokenizer, prompt, config);

    if (flags.has("json")) {
      emitJson({
        checkpoint,
        prompt,
        text,
        maxNewTokens: config.maxNewTokens,
        temperature: config.temperature,
      });
      return;
    }

    process.stdout.write(text);
    process.stdout.write("\n");
  } finally {
    model[Symbol.dispose]();
  }
}

async function runExport(flags: Map<string, string>): Promise<void> {
  const checkpointPath = flags.get("checkpoint");
  if (checkpointPath === undefined) {
    throw new UserError("Flag --checkpoint is required for export");
  }

  const outputPath = flags.get("output");
  if (outputPath === undefined) {
    throw new UserError("Flag --output is required for export");
  }

  const checkpoint = loadCheckpoint(checkpointPath);
  const model = new GPT(checkpoint.config);
  try {
    applyCheckpoint(model, checkpoint);
    await saveModelSafetensors(model, outputPath, {
      checkpoint: checkpointPath,
      step: String(checkpoint.step),
      kind: checkpoint.kind,
    });
  } finally {
    model[Symbol.dispose]();
  }

  process.stdout.write(`${outputPath}\n`);
}

function handleError(error: unknown): never {
  if (error instanceof UserError) {
    process.stderr.write(`${error.message}\n`);
    process.exit(USER_ERROR_EXIT_CODE);
  }

  const message = error instanceof Error ? (error.stack ?? error.message) : String(error);
  process.stderr.write(`${message}\n`);
  process.exit(SYSTEM_ERROR_EXIT_CODE);
}

async function main(): Promise<void> {
  const { command, flags } = parseArgs(process.argv);

  try {
    switch (command) {
      case "train":
        validateKnownFlags(flags, TRAIN_FLAG_ALLOWLIST, "train");
        if (flags.has("help")) {
          printTrainHelp();
          return;
        }
        await runTrain(flags);
        return;
      case "generate":
        validateKnownFlags(flags, GENERATE_FLAG_ALLOWLIST, "generate");
        if (flags.has("help")) {
          printGenerateHelp();
          return;
        }
        runGenerate(flags);
        return;
      case "export":
        validateKnownFlags(flags, EXPORT_FLAG_ALLOWLIST, "export");
        if (flags.has("help")) {
          printExportHelp();
          return;
        }
        await runExport(flags);
        return;
      default:
        printHelp();
    }
  } catch (error) {
    handleError(error);
  }
}

await main();
