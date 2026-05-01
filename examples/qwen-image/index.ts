#!/usr/bin/env bun

import type { DType } from "@mlxts/core";
import { random } from "@mlxts/core";
import {
  type DiffusionSnapshotResolveProgressEvent,
  FlowMatchEulerScheduler,
  generateQwenImage,
  loadDiffusionSchedulerFromSnapshot,
  loadDiffusionSnapshotManifest,
  loadQwenImageAutoencoderFromSnapshot,
  loadQwenImageTransformerFromSnapshot,
  type QwenImageGenerationOptions,
  resolveDiffusionSnapshot,
} from "@mlxts/diffusion";

import { acquireRuntimeCommandLock } from "../../scripts/runtime-command-lock";
import { loadQwenImagePromptConditionerFromSnapshot } from "./conditioning";
import {
  QWEN_IMAGE_DEFAULT_NEGATIVE_PROMPT,
  QWEN_IMAGE_DEFAULT_TRUE_CFG_SCALE,
  QWEN_IMAGE_MAX_SEQUENCE_LENGTH,
} from "./conditioning-runtime";
import type { QwenImagePromptConditioningOptions } from "./conditioning-types";
import { writeQwenImageBmp } from "./image-output";

type CliOptions = {
  source: string;
  revision?: string;
  cacheDir?: string;
  hfToken?: string;
  variant?: string;
  localFilesOnly: boolean;
  prompt: string;
  negativePrompt?: string;
  outputPath: string;
  steps: number;
  height: number;
  width: number;
  trueCfgScale: number;
  maxSequenceLength: number;
  seed: number;
  dtype: Extract<DType, "float16" | "float32" | "bfloat16">;
  json: boolean;
};

type CliCommand = { kind: "help" } | { kind: "run"; options: CliOptions };

type CliOptionsDraft = {
  revision?: string;
  cacheDir?: string;
  hfToken?: string;
  variant?: string;
  localFilesOnly: boolean;
  prompt?: string;
  negativePrompt?: string;
  outputPath: string;
  steps: number;
  height: number;
  width: number;
  trueCfgScale: number;
  maxSequenceLength: number;
  seed: number;
  dtype: CliOptions["dtype"];
  json: boolean;
};

type QwenImageExampleResult = {
  source: string;
  snapshotPath: string;
  requestedRevision?: string;
  resolvedRevision?: string;
  pipeline: "qwen-image";
  prompt: string;
  negativePrompt: string | null;
  outputPath: string;
  imageSize: { width: number; height: number };
  outputBytes: number;
  steps: number;
  trueCfgScale: number;
  maxSequenceLength: number;
  seed: number;
  dtype: string;
  promptTruncated: boolean;
  negativePromptTruncated: boolean;
  elapsedMs: number;
};

type RuntimeLock = {
  [Symbol.dispose](): void;
};

type QwenImageExampleRuntime = {
  stdout?: (text: string) => void;
  stderr?: (text: string) => void;
  acquireLock?: () => RuntimeLock;
  runExample?: (
    cli: CliOptions,
    progress: (line: string) => void,
  ) => Promise<QwenImageExampleResult>;
};

class QwenImageExampleUsageError extends Error {}

function quoteScalar(value: string | number | boolean | null): string {
  return typeof value === "string" ? JSON.stringify(value) : String(value);
}

export function formatUsage(): string {
  return [
    "description: Run one Qwen-Image text-to-image proof and write a BMP artifact",
    "usage[2]:",
    '  bun run examples/qwen-image/index.ts <snapshot-source> --prompt "a red apple"',
    '  bun run examples/qwen-image/index.ts Qwen/Qwen-Image-2512 --local-files-only --prompt "a quiet library" --negative-prompt " " --output .tmp/qwen-image/sample.bmp --steps 4 --true-cfg-scale 4',
    "arguments[1]{name,description}:",
    '  "snapshot-source","Local Diffusers snapshot directory or Hugging Face model id"',
    "options[16]{flag,description}:",
    '  "--revision <rev>","Hub revision; default main"',
    '  "--cache-dir <path>","Hub cache directory; default Hugging Face cache"',
    '  "--hf-token <token>","Hub access token; defaults to HF token environment or cache file"',
    '  "--variant <name>","Optional Hub weight filename variant, for example fp16"',
    '  "--local-files-only","Use only an already-cached Hub snapshot"',
    '  "--prompt <text>","Required positive prompt for Qwen2.5-VL"',
    '  "--negative-prompt <text>","Negative prompt for true CFG; default single space when true CFG is active"',
    '  "--output <path>","BMP output path; default .tmp/qwen-image/sample.bmp"',
    '  "--steps <n>","Inference steps; default 4"',
    '  "--height <n>","Output height; default 1024"',
    '  "--width <n>","Output width; default 1024"',
    '  "--true-cfg-scale <n>","Qwen true CFG scale; default 4, use 1 to disable negative conditioning"',
    '  "--max-sequence-length <n>","Qwen text length after template drop, 1..1024; default 1024"',
    '  "--seed <n>","RNG seed; default 0"',
    '  "--dtype <float16|float32|bfloat16>","Latent dtype; default bfloat16"',
    '  "--json","Emit final result as JSON"',
    '  "--help","Show this help"',
    "exit_codes[3]{code,meaning}:",
    '  0,"image proof passed or help"',
    '  1,"runtime or generation failure"',
    '  2,"usage error"',
  ].join("\n");
}

function readStringFlag(flag: string, value: string | undefined): string {
  if (value === undefined || value === "" || value.startsWith("--")) {
    throw new QwenImageExampleUsageError(`Missing value for ${flag}.`);
  }
  return value;
}

function readRequiredTextFlag(flag: string, value: string | undefined): string {
  const raw = readStringFlag(flag, value);
  if (raw.trim() === "") {
    throw new QwenImageExampleUsageError(`Missing value for ${flag}.`);
  }
  return raw;
}

function readNumberFlag(flag: string, value: string | undefined): number {
  const raw = readStringFlag(flag, value);
  const parsed = Number(raw);
  if (!Number.isFinite(parsed)) {
    throw new QwenImageExampleUsageError(`Expected ${flag} to be a finite number.`);
  }
  return parsed;
}

function readPositiveIntegerFlag(flag: string, value: string | undefined): number {
  const parsed = readNumberFlag(flag, value);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new QwenImageExampleUsageError(`Expected ${flag} to be a positive integer.`);
  }
  return parsed;
}

function readIntegerFlag(flag: string, value: string | undefined): number {
  const parsed = readNumberFlag(flag, value);
  if (!Number.isInteger(parsed)) {
    throw new QwenImageExampleUsageError(`Expected ${flag} to be an integer.`);
  }
  return parsed;
}

function readNonNegativeIntegerFlag(flag: string, value: string | undefined): number {
  const parsed = readIntegerFlag(flag, value);
  if (parsed < 0) {
    throw new QwenImageExampleUsageError(`Expected ${flag} to be non-negative.`);
  }
  return parsed;
}

function readDTypeFlag(value: string | undefined): CliOptions["dtype"] {
  const raw = readStringFlag("--dtype", value);
  if (raw === "float16" || raw === "float32" || raw === "bfloat16") {
    return raw;
  }
  throw new QwenImageExampleUsageError("Expected --dtype to be float16, float32, or bfloat16.");
}

function applyFlag(argv: readonly string[], index: number, draft: CliOptionsDraft): number {
  const arg = argv[index];
  switch (arg) {
    case "--prompt":
      draft.prompt = readRequiredTextFlag(arg, argv[index + 1]);
      return index + 1;
    case "--negative-prompt":
      draft.negativePrompt = readStringFlag(arg, argv[index + 1]);
      return index + 1;
    case "--revision":
      draft.revision = readStringFlag(arg, argv[index + 1]);
      return index + 1;
    case "--cache-dir":
      draft.cacheDir = readStringFlag(arg, argv[index + 1]);
      return index + 1;
    case "--hf-token":
      draft.hfToken = readStringFlag(arg, argv[index + 1]);
      return index + 1;
    case "--variant":
      draft.variant = readStringFlag(arg, argv[index + 1]);
      return index + 1;
    case "--local-files-only":
      draft.localFilesOnly = true;
      return index;
    case "--output":
      draft.outputPath = readStringFlag(arg, argv[index + 1]);
      return index + 1;
    case "--steps":
      draft.steps = readPositiveIntegerFlag(arg, argv[index + 1]);
      return index + 1;
    case "--height":
      draft.height = readPositiveIntegerFlag(arg, argv[index + 1]);
      return index + 1;
    case "--width":
      draft.width = readPositiveIntegerFlag(arg, argv[index + 1]);
      return index + 1;
    case "--true-cfg-scale":
      draft.trueCfgScale = readNumberFlag(arg, argv[index + 1]);
      return index + 1;
    case "--max-sequence-length":
      draft.maxSequenceLength = readPositiveIntegerFlag(arg, argv[index + 1]);
      return index + 1;
    case "--seed":
      draft.seed = readNonNegativeIntegerFlag(arg, argv[index + 1]);
      return index + 1;
    case "--dtype":
      draft.dtype = readDTypeFlag(argv[index + 1]);
      return index + 1;
    case "--json":
      draft.json = true;
      return index;
    default:
      throw new QwenImageExampleUsageError(
        arg === undefined ? "Missing argument." : `Unknown argument: ${arg}`,
      );
  }
}

export function resolveQwenImageNegativePrompt(
  trueCfgScale: number,
  negativePrompt: string | undefined,
): string | undefined {
  if (!Number.isFinite(trueCfgScale) || trueCfgScale <= 0) {
    throw new QwenImageExampleUsageError("--true-cfg-scale must be a positive finite number.");
  }
  if (trueCfgScale <= 1) {
    if (negativePrompt !== undefined) {
      throw new QwenImageExampleUsageError(
        "--negative-prompt requires --true-cfg-scale greater than 1.",
      );
    }
    return undefined;
  }
  return negativePrompt ?? QWEN_IMAGE_DEFAULT_NEGATIVE_PROMPT;
}

function validateCliOptions(draft: CliOptionsDraft): void {
  if (draft.prompt === undefined) {
    throw new QwenImageExampleUsageError("Missing required --prompt <text>.");
  }
  resolveQwenImageNegativePrompt(draft.trueCfgScale, draft.negativePrompt);
  if (draft.maxSequenceLength > QWEN_IMAGE_MAX_SEQUENCE_LENGTH) {
    throw new QwenImageExampleUsageError(
      `--max-sequence-length must be no greater than ${QWEN_IMAGE_MAX_SEQUENCE_LENGTH}.`,
    );
  }
  if (draft.height % 16 !== 0 || draft.width % 16 !== 0) {
    throw new QwenImageExampleUsageError("--height and --width must be divisible by 16.");
  }
  if (!draft.outputPath.toLowerCase().endsWith(".bmp")) {
    throw new QwenImageExampleUsageError("--output must end with .bmp.");
  }
}

export function parseCommand(argv: readonly string[]): CliCommand {
  if (argv.some((arg) => arg === "--help" || arg === "-h")) {
    return { kind: "help" };
  }

  const source = argv[0];
  if (source === undefined || source.trim() === "" || source.startsWith("--")) {
    throw new QwenImageExampleUsageError("Missing snapshot source.");
  }

  const draft: CliOptionsDraft = {
    localFilesOnly: false,
    outputPath: ".tmp/qwen-image/sample.bmp",
    steps: 4,
    height: 1024,
    width: 1024,
    trueCfgScale: QWEN_IMAGE_DEFAULT_TRUE_CFG_SCALE,
    maxSequenceLength: QWEN_IMAGE_MAX_SEQUENCE_LENGTH,
    seed: 0,
    dtype: "bfloat16",
    json: false,
  };

  for (let index = 1; index < argv.length; index += 1) {
    index = applyFlag(argv, index, draft);
  }
  validateCliOptions(draft);
  const prompt = draft.prompt;
  if (prompt === undefined) {
    throw new QwenImageExampleUsageError("Missing required --prompt <text>.");
  }
  const negativePrompt = resolveQwenImageNegativePrompt(draft.trueCfgScale, draft.negativePrompt);

  const options: CliOptions = {
    source,
    localFilesOnly: draft.localFilesOnly,
    prompt,
    ...(negativePrompt === undefined ? {} : { negativePrompt }),
    outputPath: draft.outputPath,
    steps: draft.steps,
    height: draft.height,
    width: draft.width,
    trueCfgScale: draft.trueCfgScale,
    maxSequenceLength: draft.maxSequenceLength,
    seed: draft.seed,
    dtype: draft.dtype,
    json: draft.json,
  };
  if (draft.revision !== undefined) {
    options.revision = draft.revision;
  }
  if (draft.cacheDir !== undefined) {
    options.cacheDir = draft.cacheDir;
  }
  if (draft.hfToken !== undefined) {
    options.hfToken = draft.hfToken;
  }
  if (draft.variant !== undefined) {
    options.variant = draft.variant;
  }
  return { kind: "run", options };
}

export function parseArgs(argv: readonly string[]): CliOptions {
  const command = parseCommand(argv);
  if (command.kind === "help") {
    throw new QwenImageExampleUsageError("Help is not an image proof command.");
  }
  return command.options;
}

function printRunIntro(cli: CliOptions, writeLine: (line: string) => void): void {
  writeLine(`Snapshot source: ${cli.source}`);
  if (cli.revision !== undefined) {
    writeLine(`Snapshot revision: ${cli.revision}`);
  }
  if (cli.localFilesOnly) {
    writeLine("Local files only: true");
  }
  if (cli.variant !== undefined) {
    writeLine(`Weight variant: ${cli.variant}`);
  }
  writeLine(`Prompt: ${cli.prompt}`);
  if (cli.negativePrompt !== undefined) {
    writeLine(`Negative prompt: ${cli.negativePrompt}`);
  }
  writeLine(`Output: ${cli.outputPath}`);
  writeLine(`Steps: ${cli.steps}`);
  writeLine(`True CFG scale: ${cli.trueCfgScale}`);
  writeLine(`Seed: ${cli.seed}`);
  writeLine(`Latent dtype: ${cli.dtype}`);
  writeLine("");
}

function formatSnapshotResolveProgress(event: DiffusionSnapshotResolveProgressEvent): string {
  if (event.stage === "resolve" && event.status === "start") {
    return `Resolving snapshot source: ${event.source}`;
  }
  if (event.stage === "resolve") {
    return `Resolved ${event.sourceKind} snapshot: ${event.directory} (${event.fileCount} files, ${event.totalBytes} bytes)`;
  }
  return `Snapshot ${event.status} ${event.index}/${event.totalFiles}: ${event.relativePath}`;
}

export async function runQwenImageExample(
  cli: CliOptions,
  progress: (line: string) => void,
): Promise<QwenImageExampleResult> {
  const startedAt = performance.now();
  printRunIntro(cli, progress);
  const snapshot = await resolveDiffusionSnapshot(cli.source, {
    ...(cli.revision === undefined ? {} : { revision: cli.revision }),
    ...(cli.cacheDir === undefined ? {} : { cacheDir: cli.cacheDir }),
    ...(cli.hfToken === undefined ? {} : { accessToken: cli.hfToken }),
    ...(cli.variant === undefined ? {} : { variant: cli.variant }),
    localFilesOnly: cli.localFilesOnly,
    onProgress: (event) => {
      progress(formatSnapshotResolveProgress(event));
    },
  });
  const snapshotPath = snapshot.directory;

  progress("Loading Qwen-Image snapshot manifest...");
  const manifest = await loadDiffusionSnapshotManifest(snapshotPath);
  if (manifest.modelIndex.kind !== "qwen-image") {
    throw new Error(`Qwen-Image proof does not support ${manifest.modelIndex.kind}.`);
  }

  progress("Loading Qwen-Image scheduler...");
  const loadedScheduler = await loadDiffusionSchedulerFromSnapshot(snapshotPath);
  if (!(loadedScheduler.scheduler instanceof FlowMatchEulerScheduler)) {
    throw new Error("Qwen-Image proof requires a FlowMatch Euler scheduler.");
  }
  const scheduler = loadedScheduler.scheduler;

  progress("Loading Qwen-Image transformer...");
  using transformer = await loadQwenImageTransformerFromSnapshot(manifest);

  progress("Loading Qwen-Image VAE...");
  using vae = await loadQwenImageAutoencoderFromSnapshot(manifest);

  progress("Loading Qwen-Image prompt conditioner...");
  using conditioner = await loadQwenImagePromptConditionerFromSnapshot(snapshotPath);
  const promptOptions: QwenImagePromptConditioningOptions = {
    prompt: cli.prompt,
    ...(cli.negativePrompt === undefined ? {} : { negativePrompt: cli.negativePrompt }),
    trueCfgScale: cli.trueCfgScale,
    maxSequenceLength: cli.maxSequenceLength,
  };

  progress("Encoding Qwen2.5-VL prompt conditioning...");
  using conditioning = conditioner.encodePrompt(promptOptions);
  using rngKey = random.key(cli.seed);

  progress(`Denoising ${cli.width}x${cli.height} image...`);
  const generationOptions: Omit<QwenImageGenerationOptions, "denoiser" | "vae" | "scheduler"> = {
    batchSize: conditioning.batchSize,
    height: cli.height,
    width: cli.width,
    conditioning: conditioning.conditioning,
    numInferenceSteps: cli.steps,
    dtype: cli.dtype,
    rngKey,
    onStep: (event) => {
      progress(`Denoise step ${event.stepIndex + 1}/${cli.steps}: sigma ${event.sigma}`);
    },
  };
  using image = generateQwenImage({
    ...generationOptions,
    denoiser: transformer,
    vae,
    scheduler,
  });

  const artifact = writeQwenImageBmp(image, cli.outputPath);
  return {
    source: cli.source,
    snapshotPath,
    ...(snapshot.requestedRevision === undefined
      ? {}
      : { requestedRevision: snapshot.requestedRevision }),
    ...(snapshot.resolvedRevision === undefined
      ? {}
      : { resolvedRevision: snapshot.resolvedRevision }),
    pipeline: "qwen-image",
    prompt: cli.prompt,
    negativePrompt: cli.negativePrompt ?? null,
    outputPath: artifact.path,
    imageSize: {
      width: artifact.width,
      height: artifact.height,
    },
    outputBytes: artifact.bytes,
    steps: cli.steps,
    trueCfgScale: cli.trueCfgScale,
    maxSequenceLength: cli.maxSequenceLength,
    seed: cli.seed,
    dtype: cli.dtype,
    promptTruncated: conditioning.promptTruncated,
    negativePromptTruncated: conditioning.negativePromptTruncated,
    elapsedMs: Math.round((performance.now() - startedAt) * 100) / 100,
  };
}

export function formatSuccess(report: QwenImageExampleResult): string {
  return [
    "qwen_image_example:",
    "  status: passed",
    `  source: ${quoteScalar(report.source)}`,
    `  snapshot_path: ${quoteScalar(report.snapshotPath)}`,
    ...(report.requestedRevision === undefined
      ? []
      : [`  requested_revision: ${quoteScalar(report.requestedRevision)}`]),
    ...(report.resolvedRevision === undefined
      ? []
      : [`  resolved_revision: ${quoteScalar(report.resolvedRevision)}`]),
    `  pipeline: ${quoteScalar(report.pipeline)}`,
    `  output_path: ${quoteScalar(report.outputPath)}`,
    `  image_size: ${quoteScalar(`${report.imageSize.width}x${report.imageSize.height}`)}`,
    `  output_bytes: ${report.outputBytes}`,
    `  steps: ${report.steps}`,
    `  true_cfg_scale: ${report.trueCfgScale}`,
    `  max_sequence_length: ${report.maxSequenceLength}`,
    `  seed: ${report.seed}`,
    `  dtype: ${quoteScalar(report.dtype)}`,
    `  prompt_truncated: ${report.promptTruncated}`,
    `  negative_prompt_truncated: ${report.negativePromptTruncated}`,
    `  elapsed_ms: ${report.elapsedMs}`,
    `  prompt: ${quoteScalar(report.prompt)}`,
    `  negative_prompt: ${quoteScalar(report.negativePrompt)}`,
  ].join("\n");
}

function formatError(message: string, code: "usage" | "runtime"): string {
  return [
    "error:",
    `  code: ${quoteScalar(code)}`,
    `  message: ${quoteScalar(message)}`,
    "help[1]:",
    '  "Run `bun run examples/qwen-image/index.ts --help` for options"',
  ].join("\n");
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export async function runQwenImageExampleCommand(
  argv: readonly string[],
  runtime: QwenImageExampleRuntime = {},
): Promise<number> {
  const stdout = runtime.stdout ?? console.log;
  const stderr = runtime.stderr ?? console.error;
  let command: CliCommand;
  try {
    command = parseCommand(argv);
  } catch (error) {
    stdout(formatError(errorMessage(error), "usage"));
    return error instanceof QwenImageExampleUsageError ? 2 : 1;
  }

  if (command.kind === "help") {
    stdout(formatUsage());
    return 0;
  }

  const acquireLock =
    runtime.acquireLock ?? (() => acquireRuntimeCommandLock("example:qwen-image"));
  const runExample = runtime.runExample ?? runQwenImageExample;
  let lock: RuntimeLock | undefined;
  try {
    lock = acquireLock();
    const report = await runExample(command.options, stderr);
    stdout(command.options.json ? JSON.stringify(report, null, 2) : formatSuccess(report));
    return 0;
  } catch (error) {
    stdout(formatError(errorMessage(error), "runtime"));
    if (error instanceof Error && error.stack !== undefined) {
      stderr(error.stack);
    }
    return 1;
  } finally {
    lock?.[Symbol.dispose]();
  }
}

if (import.meta.main) {
  const exitCode = await runQwenImageExampleCommand(Bun.argv.slice(2));
  process.exit(exitCode);
}
