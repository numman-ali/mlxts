#!/usr/bin/env bun

import type { DType } from "@mlxts/core";
import { random } from "@mlxts/core";
import {
  type DiffusionSnapshotResolveProgressEvent,
  FlowMatchEulerScheduler,
  generateZImage,
  loadDiffusionSchedulerFromSnapshot,
  loadDiffusionSnapshotManifest,
  loadZImageAutoencoderFromSnapshot,
  loadZImageTransformerFromSnapshot,
  resolveDiffusionSnapshot,
  type ZImageGenerationOptions,
} from "@mlxts/diffusion";

import { acquireRuntimeCommandLock } from "../../scripts/runtime-command-lock";
import { loadZImagePromptConditionerFromSnapshot } from "./conditioning";
import type { ZImagePromptConditioningOptions } from "./conditioning-types";
import { writeZImageBmp, type ZImageBmpWriteResult } from "./image-output";

type CliOptions = {
  source: string;
  revision?: string;
  cacheDir?: string;
  hfToken?: string;
  variant?: string;
  localFilesOnly: boolean;
  prompt: string;
  outputPath: string;
  steps: number;
  height: number;
  width: number;
  guidanceScale: number;
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
  outputPath: string;
  steps: number;
  height: number;
  width: number;
  guidanceScale: number;
  maxSequenceLength: number;
  seed: number;
  dtype: CliOptions["dtype"];
  json: boolean;
};

type ZImageExampleResult = {
  source: string;
  snapshotPath: string;
  requestedRevision?: string;
  resolvedRevision?: string;
  pipeline: "z-image";
  prompt: string;
  outputPath: string;
  imageSize: { width: number; height: number };
  outputBytes: number;
  artifact: ZImageBmpWriteResult;
  steps: number;
  guidanceScale: number;
  maxSequenceLength: number;
  seed: number;
  dtype: string;
  promptTruncated: boolean;
  elapsedMs: number;
};

type RuntimeLock = {
  [Symbol.dispose](): void;
};

type ZImageExampleRuntime = {
  stdout?: (text: string) => void;
  stderr?: (text: string) => void;
  acquireLock?: () => RuntimeLock;
  runExample?: (cli: CliOptions, progress: (line: string) => void) => Promise<ZImageExampleResult>;
};

class ZImageExampleUsageError extends Error {}

function quoteScalar(value: string | number | boolean | null): string {
  return typeof value === "string" ? JSON.stringify(value) : String(value);
}

export function formatUsage(): string {
  return [
    "description: Run one Z-Image text-to-image proof and write a BMP artifact",
    "usage[2]:",
    '  bun run examples/z-image/index.ts <snapshot-source> --prompt "a red apple"',
    '  bun run examples/z-image/index.ts Tongyi-MAI/Z-Image-Turbo --local-files-only --prompt "a quiet library" --output .tmp/z-image/sample.bmp --steps 9 --guidance-scale 0',
    "arguments[1]{name,description}:",
    '  "snapshot-source","Local Diffusers snapshot directory or Hugging Face model id"',
    "options[16]{flag,description}:",
    '  "--revision <rev>","Hub revision; default main"',
    '  "--cache-dir <path>","Hub cache directory; default Hugging Face cache"',
    '  "--hf-token <token>","Hub access token; defaults to HF token environment or cache file"',
    '  "--variant <name>","Optional Hub weight filename variant, for example fp16"',
    '  "--local-files-only","Use only an already-cached Hub snapshot"',
    '  "--prompt <text>","Required positive prompt for Qwen3"',
    '  "--output <path>","BMP output path; default .tmp/z-image/sample.bmp"',
    '  "--steps <n>","Inference steps; default 9"',
    '  "--height <n>","Output height; default 1024"',
    '  "--width <n>","Output width; default 1024"',
    '  "--guidance-scale <n>","Must be 0 for the current Turbo proof"',
    '  "--max-sequence-length <n>","Qwen3 token length, 1..512; default 512"',
    '  "--seed <n>","RNG seed; default 0"',
    '  "--dtype <float16|float32|bfloat16>","Latent dtype; default float32"',
    '  "--json","Emit final result as JSON"',
    '  "--help","Show this help"',
    "exit_codes[3]{code,meaning}:",
    '  0,"image proof passed or help"',
    '  1,"runtime or generation failure"',
    '  2,"usage error"',
  ].join("\n");
}

function readStringFlag(flag: string, value: string | undefined): string {
  if (value === undefined || value.trim() === "" || value.startsWith("--")) {
    throw new ZImageExampleUsageError(`Missing value for ${flag}.`);
  }
  return value;
}

function readNumberFlag(flag: string, value: string | undefined): number {
  const raw = readStringFlag(flag, value);
  const parsed = Number(raw);
  if (!Number.isFinite(parsed)) {
    throw new ZImageExampleUsageError(`Expected ${flag} to be a finite number.`);
  }
  return parsed;
}

function readPositiveIntegerFlag(flag: string, value: string | undefined): number {
  const parsed = readNumberFlag(flag, value);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new ZImageExampleUsageError(`Expected ${flag} to be a positive integer.`);
  }
  return parsed;
}

function readIntegerFlag(flag: string, value: string | undefined): number {
  const parsed = readNumberFlag(flag, value);
  if (!Number.isInteger(parsed)) {
    throw new ZImageExampleUsageError(`Expected ${flag} to be an integer.`);
  }
  return parsed;
}

function readNonNegativeIntegerFlag(flag: string, value: string | undefined): number {
  const parsed = readIntegerFlag(flag, value);
  if (parsed < 0) {
    throw new ZImageExampleUsageError(`Expected ${flag} to be non-negative.`);
  }
  return parsed;
}

function readDTypeFlag(value: string | undefined): CliOptions["dtype"] {
  const raw = readStringFlag("--dtype", value);
  if (raw === "float16" || raw === "float32" || raw === "bfloat16") {
    return raw;
  }
  throw new ZImageExampleUsageError("Expected --dtype to be float16, float32, or bfloat16.");
}

function applyFlag(argv: readonly string[], index: number, draft: CliOptionsDraft): number {
  const arg = argv[index];
  switch (arg) {
    case "--prompt":
      draft.prompt = readStringFlag(arg, argv[index + 1]);
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
    case "--guidance-scale":
      draft.guidanceScale = readNumberFlag(arg, argv[index + 1]);
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
      throw new ZImageExampleUsageError(
        arg === undefined ? "Missing argument." : `Unknown argument: ${arg}`,
      );
  }
}

function validateCliOptions(draft: CliOptionsDraft): void {
  if (draft.prompt === undefined) {
    throw new ZImageExampleUsageError("Missing required --prompt <text>.");
  }
  if (draft.guidanceScale !== 0) {
    throw new ZImageExampleUsageError("--guidance-scale must be 0 for the current Z-Image proof.");
  }
  if (draft.maxSequenceLength > 512) {
    throw new ZImageExampleUsageError("--max-sequence-length must be no greater than 512.");
  }
  if (draft.height % 16 !== 0 || draft.width % 16 !== 0) {
    throw new ZImageExampleUsageError("--height and --width must be divisible by 16.");
  }
  if (!draft.outputPath.toLowerCase().endsWith(".bmp")) {
    throw new ZImageExampleUsageError("--output must end with .bmp.");
  }
}

export function parseCommand(argv: readonly string[]): CliCommand {
  if (argv.some((arg) => arg === "--help" || arg === "-h")) {
    return { kind: "help" };
  }

  const source = argv[0];
  if (source === undefined || source.trim() === "" || source.startsWith("--")) {
    throw new ZImageExampleUsageError("Missing snapshot source.");
  }

  const draft: CliOptionsDraft = {
    localFilesOnly: false,
    outputPath: ".tmp/z-image/sample.bmp",
    steps: 9,
    height: 1024,
    width: 1024,
    guidanceScale: 0,
    maxSequenceLength: 512,
    seed: 0,
    dtype: "float32",
    json: false,
  };

  for (let index = 1; index < argv.length; index += 1) {
    index = applyFlag(argv, index, draft);
  }
  validateCliOptions(draft);
  const prompt = draft.prompt;
  if (prompt === undefined) {
    throw new ZImageExampleUsageError("Missing required --prompt <text>.");
  }

  const options: CliOptions = {
    source,
    localFilesOnly: draft.localFilesOnly,
    prompt,
    outputPath: draft.outputPath,
    steps: draft.steps,
    height: draft.height,
    width: draft.width,
    guidanceScale: draft.guidanceScale,
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
    throw new ZImageExampleUsageError("Help is not an image proof command.");
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
  writeLine(`Output: ${cli.outputPath}`);
  writeLine(`Steps: ${cli.steps}`);
  writeLine(`Guidance scale: ${cli.guidanceScale}`);
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

export function resolveZImageGuidanceScale(guidanceScale: number): number {
  if (guidanceScale !== 0) {
    throw new Error("Z-Image Turbo proof currently supports guidance scale 0 only.");
  }
  return guidanceScale;
}

export async function runZImageExample(
  cli: CliOptions,
  progress: (line: string) => void,
): Promise<ZImageExampleResult> {
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

  progress("Loading Z-Image snapshot manifest...");
  const manifest = await loadDiffusionSnapshotManifest(snapshotPath);
  if (manifest.modelIndex.kind !== "z-image") {
    throw new Error(`Z-Image proof does not support ${manifest.modelIndex.kind}.`);
  }

  progress("Loading Z-Image scheduler...");
  const loadedScheduler = await loadDiffusionSchedulerFromSnapshot(snapshotPath);
  if (!(loadedScheduler.scheduler instanceof FlowMatchEulerScheduler)) {
    throw new Error("Z-Image proof requires a FlowMatch Euler scheduler.");
  }
  const scheduler = loadedScheduler.scheduler;
  const guidanceScale = resolveZImageGuidanceScale(cli.guidanceScale);

  progress("Loading Z-Image transformer...");
  using transformer = await loadZImageTransformerFromSnapshot(manifest);

  progress("Loading Z-Image VAE...");
  using vae = await loadZImageAutoencoderFromSnapshot(manifest);

  progress("Loading Z-Image prompt conditioner...");
  using conditioner = await loadZImagePromptConditionerFromSnapshot(snapshotPath);
  const promptOptions: ZImagePromptConditioningOptions = {
    prompt: cli.prompt,
    maxSequenceLength: cli.maxSequenceLength,
  };

  progress("Encoding Qwen3 prompt conditioning...");
  using conditioning = conditioner.encodePrompt(promptOptions);
  using rngKey = random.key(cli.seed);

  progress(`Denoising ${cli.width}x${cli.height} image...`);
  const generationOptions: Omit<ZImageGenerationOptions, "denoiser" | "vae" | "scheduler"> = {
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
  using image = generateZImage({
    ...generationOptions,
    denoiser: transformer,
    vae,
    scheduler,
  });

  const artifact = writeZImageBmp(image, cli.outputPath);
  return {
    source: cli.source,
    snapshotPath,
    ...(snapshot.requestedRevision === undefined
      ? {}
      : { requestedRevision: snapshot.requestedRevision }),
    ...(snapshot.resolvedRevision === undefined
      ? {}
      : { resolvedRevision: snapshot.resolvedRevision }),
    pipeline: "z-image",
    prompt: cli.prompt,
    outputPath: artifact.path,
    imageSize: {
      width: artifact.width,
      height: artifact.height,
    },
    outputBytes: artifact.bytes,
    artifact,
    steps: cli.steps,
    guidanceScale,
    maxSequenceLength: cli.maxSequenceLength,
    seed: cli.seed,
    dtype: cli.dtype,
    promptTruncated: conditioning.promptTruncated,
    elapsedMs: Math.round((performance.now() - startedAt) * 100) / 100,
  };
}

export function formatSuccess(report: ZImageExampleResult): string {
  return [
    "z_image_example:",
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
    `  artifact_sha256: ${quoteScalar(report.artifact.sha256)}`,
    `  artifact_checks: ${quoteScalar(report.artifact.status)}`,
    `  artifact_unique_byte_values: ${report.artifact.tensor.uniqueByteValues}`,
    `  artifact_channel_stddev_max: ${report.artifact.tensor.maxChannelStddev}`,
    `  steps: ${report.steps}`,
    `  guidance_scale: ${report.guidanceScale}`,
    `  max_sequence_length: ${report.maxSequenceLength}`,
    `  seed: ${report.seed}`,
    `  dtype: ${quoteScalar(report.dtype)}`,
    `  prompt_truncated: ${report.promptTruncated}`,
    `  elapsed_ms: ${report.elapsedMs}`,
    `  prompt: ${quoteScalar(report.prompt)}`,
  ].join("\n");
}

function formatError(message: string, code: "usage" | "runtime"): string {
  return [
    "error:",
    `  code: ${quoteScalar(code)}`,
    `  message: ${quoteScalar(message)}`,
    "help[1]:",
    '  "Run `bun run examples/z-image/index.ts --help` for options"',
  ].join("\n");
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export async function runZImageExampleCommand(
  argv: readonly string[],
  runtime: ZImageExampleRuntime = {},
): Promise<number> {
  const stdout = runtime.stdout ?? console.log;
  const stderr = runtime.stderr ?? console.error;
  let command: CliCommand;
  try {
    command = parseCommand(argv);
  } catch (error) {
    stdout(formatError(errorMessage(error), "usage"));
    return error instanceof ZImageExampleUsageError ? 2 : 1;
  }

  if (command.kind === "help") {
    stdout(formatUsage());
    return 0;
  }

  const acquireLock = runtime.acquireLock ?? (() => acquireRuntimeCommandLock("example:z-image"));
  const runExample = runtime.runExample ?? runZImageExample;
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
  const exitCode = await runZImageExampleCommand(Bun.argv.slice(2));
  process.exit(exitCode);
}
