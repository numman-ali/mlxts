#!/usr/bin/env bun

import type { DType, MxArray } from "@mlxts/core";
import { random } from "@mlxts/core";
import {
  type DiffusionSnapshotResolveProgressEvent,
  FlowMatchEulerScheduler,
  generateStableDiffusion3Image,
  loadDiffusionSchedulerFromSnapshot,
  loadDiffusionSnapshotManifest,
  loadStableDiffusion3AutoencoderFromSnapshot,
  loadStableDiffusion3ComponentConfigs,
  loadStableDiffusion3TransformerFromSnapshot,
  type ResolvedDiffusionSnapshot,
  resolveDiffusionSnapshot,
  type StableDiffusion3ImageGenerationOptions,
} from "@mlxts/diffusion";

import { acquireRuntimeCommandLock } from "../../scripts/runtime-command-lock";
import { loadStableDiffusion3PromptConditionerFromSnapshot } from "./conditioning";
import { SD3_DEFAULT_MAX_SEQUENCE_LENGTH, SD3_MAX_SEQUENCE_LENGTH } from "./conditioning-runtime";
import type {
  StableDiffusion3PromptConditioning,
  StableDiffusion3PromptConditioningOptions,
} from "./conditioning-types";
import { type StableDiffusion3BmpWriteResult, writeStableDiffusion3Bmp } from "./image-output";

type CliOptions = {
  source: string;
  revision?: string;
  cacheDir?: string;
  hfToken?: string;
  variant?: string;
  localFilesOnly: boolean;
  prompt: string;
  prompt2?: string;
  prompt3?: string;
  negativePrompt?: string;
  negativePrompt2?: string;
  negativePrompt3?: string;
  outputPath: string;
  steps: number;
  height?: number;
  width?: number;
  guidanceScale: number;
  maxSequenceLength: number;
  clipSkip: number;
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
  prompt2?: string;
  prompt3?: string;
  negativePrompt?: string;
  negativePrompt2?: string;
  negativePrompt3?: string;
  outputPath: string;
  steps: number;
  height?: number;
  width?: number;
  guidanceScale: number;
  maxSequenceLength: number;
  clipSkip: number;
  seed: number;
  dtype: CliOptions["dtype"];
  json: boolean;
};

type StableDiffusion3ExampleResult = {
  source: string;
  snapshotPath: string;
  requestedRevision?: string;
  resolvedRevision?: string;
  pipeline: "stable-diffusion-3";
  prompt: string;
  negativePrompt: string | null;
  outputPath: string;
  imageSize: { width: number; height: number };
  outputBytes: number;
  artifact: StableDiffusion3BmpWriteResult;
  steps: number;
  guidanceScale: number;
  maxSequenceLength: number;
  clipSkip: number;
  seed: number;
  dtype: string;
  promptTruncated: boolean;
  prompt2Truncated: boolean;
  prompt3Truncated: boolean;
  negativePromptTruncated: boolean;
  negativePrompt2Truncated: boolean;
  negativePrompt3Truncated: boolean;
  elapsedMs: number;
};

type RuntimeLock = {
  [Symbol.dispose](): void;
};

type StableDiffusion3ExampleRuntime = {
  stdout?: (text: string) => void;
  stderr?: (text: string) => void;
  acquireLock?: () => RuntimeLock;
  runExample?: (
    cli: CliOptions,
    progress: (line: string) => void,
  ) => Promise<StableDiffusion3ExampleResult>;
};

class StableDiffusion3ExampleUsageError extends Error {}

function quoteScalar(value: string | number | boolean | null): string {
  return typeof value === "string" ? JSON.stringify(value) : String(value);
}

export function formatUsage(): string {
  return [
    "description: Run one Stable Diffusion 3 text-to-image proof and write a BMP artifact",
    "usage[2]:",
    '  bun run examples/stable-diffusion-3/index.ts <snapshot-source> --prompt "a red apple"',
    '  bun run examples/stable-diffusion-3/index.ts stabilityai/stable-diffusion-3.5-medium --local-files-only --prompt "a quiet library" --output .tmp/stable-diffusion-3/sample.bmp --steps 28 --guidance-scale 7',
    "arguments[1]{name,description}:",
    '  "snapshot-source","Local Diffusers snapshot directory or Hugging Face model id"',
    "options[22]{flag,description}:",
    '  "--revision <rev>","Hub revision; default main"',
    '  "--cache-dir <path>","Hub cache directory; default Hugging Face cache"',
    '  "--hf-token <token>","Hub access token; defaults to HF token environment or cache file"',
    '  "--variant <name>","Optional Hub weight filename variant, for example fp16"',
    '  "--local-files-only","Use only an already-cached Hub snapshot"',
    '  "--prompt <text>","Required positive prompt"',
    '  "--prompt-2 <text>","Optional second CLIP prompt; empty falls back to prompt"',
    '  "--prompt-3 <text>","Optional T5 prompt; empty falls back to prompt"',
    '  "--negative-prompt <text>","Negative prompt for CFG; default empty string when guidance is active"',
    '  "--negative-prompt-2 <text>","Optional second CLIP negative prompt; empty falls back to negative prompt"',
    '  "--negative-prompt-3 <text>","Optional T5 negative prompt; empty falls back to negative prompt"',
    '  "--output <path>","BMP output path; default .tmp/stable-diffusion-3/sample.bmp"',
    '  "--steps <n>","Inference steps; default 28"',
    '  "--height <n>","Output height; default transformer sample size times VAE scale"',
    '  "--width <n>","Output width; default transformer sample size times VAE scale"',
    '  "--guidance-scale <n>","Classifier-free guidance scale; default 7, use 1 to disable negative conditioning"',
    `  "--max-sequence-length <n>","T5 token length, 1..${SD3_MAX_SEQUENCE_LENGTH}; default ${SD3_DEFAULT_MAX_SEQUENCE_LENGTH}"`,
    '  "--clip-skip <n>","Positive CLIP hidden-state skip; default 0"',
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
  if (value === undefined || value.startsWith("--")) {
    throw new StableDiffusion3ExampleUsageError(`Missing value for ${flag}.`);
  }
  return value;
}

function readRequiredTextFlag(flag: string, value: string | undefined): string {
  const raw = readStringFlag(flag, value);
  if (raw.trim() === "") {
    throw new StableDiffusion3ExampleUsageError(`Missing value for ${flag}.`);
  }
  return raw;
}

function readNumberFlag(flag: string, value: string | undefined): number {
  const raw = readRequiredTextFlag(flag, value);
  const parsed = Number(raw);
  if (!Number.isFinite(parsed)) {
    throw new StableDiffusion3ExampleUsageError(`Expected ${flag} to be a finite number.`);
  }
  return parsed;
}

function readPositiveIntegerFlag(flag: string, value: string | undefined): number {
  const parsed = readNumberFlag(flag, value);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new StableDiffusion3ExampleUsageError(`Expected ${flag} to be a positive integer.`);
  }
  return parsed;
}

function readIntegerFlag(flag: string, value: string | undefined): number {
  const parsed = readNumberFlag(flag, value);
  if (!Number.isInteger(parsed)) {
    throw new StableDiffusion3ExampleUsageError(`Expected ${flag} to be an integer.`);
  }
  return parsed;
}

function readNonNegativeIntegerFlag(flag: string, value: string | undefined): number {
  const parsed = readIntegerFlag(flag, value);
  if (parsed < 0) {
    throw new StableDiffusion3ExampleUsageError(`Expected ${flag} to be non-negative.`);
  }
  return parsed;
}

function readDTypeFlag(value: string | undefined): CliOptions["dtype"] {
  const raw = readRequiredTextFlag("--dtype", value);
  if (raw === "float16" || raw === "float32" || raw === "bfloat16") {
    return raw;
  }
  throw new StableDiffusion3ExampleUsageError(
    "Expected --dtype to be float16, float32, or bfloat16.",
  );
}

function applyFlag(argv: readonly string[], index: number, draft: CliOptionsDraft): number {
  const arg = argv[index];
  switch (arg) {
    case "--prompt":
      draft.prompt = readRequiredTextFlag(arg, argv[index + 1]);
      return index + 1;
    case "--prompt-2":
      draft.prompt2 = readStringFlag(arg, argv[index + 1]);
      return index + 1;
    case "--prompt-3":
      draft.prompt3 = readStringFlag(arg, argv[index + 1]);
      return index + 1;
    case "--negative-prompt":
      draft.negativePrompt = readStringFlag(arg, argv[index + 1]);
      return index + 1;
    case "--negative-prompt-2":
      draft.negativePrompt2 = readStringFlag(arg, argv[index + 1]);
      return index + 1;
    case "--negative-prompt-3":
      draft.negativePrompt3 = readStringFlag(arg, argv[index + 1]);
      return index + 1;
    case "--revision":
      draft.revision = readRequiredTextFlag(arg, argv[index + 1]);
      return index + 1;
    case "--cache-dir":
      draft.cacheDir = readRequiredTextFlag(arg, argv[index + 1]);
      return index + 1;
    case "--hf-token":
      draft.hfToken = readRequiredTextFlag(arg, argv[index + 1]);
      return index + 1;
    case "--variant":
      draft.variant = readRequiredTextFlag(arg, argv[index + 1]);
      return index + 1;
    case "--local-files-only":
      draft.localFilesOnly = true;
      return index;
    case "--output":
      draft.outputPath = readRequiredTextFlag(arg, argv[index + 1]);
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
    case "--clip-skip":
      draft.clipSkip = readNonNegativeIntegerFlag(arg, argv[index + 1]);
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
      throw new StableDiffusion3ExampleUsageError(
        arg === undefined ? "Missing argument." : `Unknown argument: ${arg}`,
      );
  }
}

function hasNegativePromptOverride(draft: CliOptionsDraft): boolean {
  return (
    draft.negativePrompt !== undefined ||
    draft.negativePrompt2 !== undefined ||
    draft.negativePrompt3 !== undefined
  );
}

function validateCliOptions(draft: CliOptionsDraft): void {
  if (draft.prompt === undefined) {
    throw new StableDiffusion3ExampleUsageError("Missing required --prompt <text>.");
  }
  if (!Number.isFinite(draft.guidanceScale) || draft.guidanceScale < 0) {
    throw new StableDiffusion3ExampleUsageError(
      "--guidance-scale must be finite and non-negative.",
    );
  }
  if (draft.guidanceScale <= 1 && hasNegativePromptOverride(draft)) {
    throw new StableDiffusion3ExampleUsageError(
      "--negative-prompt options require --guidance-scale greater than 1.",
    );
  }
  if (draft.maxSequenceLength > SD3_MAX_SEQUENCE_LENGTH) {
    throw new StableDiffusion3ExampleUsageError(
      `--max-sequence-length must be no greater than ${SD3_MAX_SEQUENCE_LENGTH}.`,
    );
  }
  if (!draft.outputPath.toLowerCase().endsWith(".bmp")) {
    throw new StableDiffusion3ExampleUsageError("--output must end with .bmp.");
  }
}

function defaultCliOptionsDraft(): CliOptionsDraft {
  return {
    localFilesOnly: false,
    outputPath: ".tmp/stable-diffusion-3/sample.bmp",
    steps: 28,
    guidanceScale: 7,
    maxSequenceLength: SD3_DEFAULT_MAX_SEQUENCE_LENGTH,
    clipSkip: 0,
    seed: 0,
    dtype: "bfloat16",
    json: false,
  };
}

function buildCliOptions(source: string, draft: CliOptionsDraft): CliOptions {
  const prompt = draft.prompt;
  if (prompt === undefined) {
    throw new StableDiffusion3ExampleUsageError("Missing required --prompt <text>.");
  }
  return {
    source,
    localFilesOnly: draft.localFilesOnly,
    prompt,
    ...(draft.revision === undefined ? {} : { revision: draft.revision }),
    ...(draft.cacheDir === undefined ? {} : { cacheDir: draft.cacheDir }),
    ...(draft.hfToken === undefined ? {} : { hfToken: draft.hfToken }),
    ...(draft.variant === undefined ? {} : { variant: draft.variant }),
    ...(draft.prompt2 === undefined ? {} : { prompt2: draft.prompt2 }),
    ...(draft.prompt3 === undefined ? {} : { prompt3: draft.prompt3 }),
    ...(draft.negativePrompt === undefined ? {} : { negativePrompt: draft.negativePrompt }),
    ...(draft.negativePrompt2 === undefined ? {} : { negativePrompt2: draft.negativePrompt2 }),
    ...(draft.negativePrompt3 === undefined ? {} : { negativePrompt3: draft.negativePrompt3 }),
    outputPath: draft.outputPath,
    steps: draft.steps,
    ...(draft.height === undefined ? {} : { height: draft.height }),
    ...(draft.width === undefined ? {} : { width: draft.width }),
    guidanceScale: draft.guidanceScale,
    maxSequenceLength: draft.maxSequenceLength,
    clipSkip: draft.clipSkip,
    seed: draft.seed,
    dtype: draft.dtype,
    json: draft.json,
  };
}

export function parseCommand(argv: readonly string[]): CliCommand {
  if (argv.some((arg) => arg === "--help" || arg === "-h")) {
    return { kind: "help" };
  }

  const source = argv[0];
  if (source === undefined || source.trim() === "" || source.startsWith("--")) {
    throw new StableDiffusion3ExampleUsageError("Missing snapshot source.");
  }

  const draft = defaultCliOptionsDraft();
  for (let index = 1; index < argv.length; index += 1) {
    index = applyFlag(argv, index, draft);
  }
  validateCliOptions(draft);
  return { kind: "run", options: buildCliOptions(source, draft) };
}

export function parseArgs(argv: readonly string[]): CliOptions {
  const command = parseCommand(argv);
  if (command.kind === "help") {
    throw new StableDiffusion3ExampleUsageError("Help is not an image proof command.");
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
  writeLine(`Guidance scale: ${cli.guidanceScale}`);
  writeLine(`Max sequence length: ${cli.maxSequenceLength}`);
  writeLine(`CLIP skip: ${cli.clipSkip}`);
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

function defaultImageSize(
  sampleSize: number,
  vaeScaleFactor: number,
): {
  height: number;
  width: number;
} {
  const side = sampleSize * vaeScaleFactor;
  return { height: side, width: side };
}

function validateImageSize(height: number, width: number, divisor: number): void {
  if (height % divisor !== 0 || width % divisor !== 0) {
    throw new StableDiffusion3ExampleUsageError(
      `--height and --width must be divisible by ${divisor}.`,
    );
  }
}

async function resolveExampleSnapshot(
  cli: CliOptions,
  progress: (line: string) => void,
): Promise<ResolvedDiffusionSnapshot> {
  return await resolveDiffusionSnapshot(cli.source, {
    ...(cli.revision === undefined ? {} : { revision: cli.revision }),
    ...(cli.cacheDir === undefined ? {} : { cacheDir: cli.cacheDir }),
    ...(cli.hfToken === undefined ? {} : { accessToken: cli.hfToken }),
    ...(cli.variant === undefined ? {} : { variant: cli.variant }),
    localFilesOnly: cli.localFilesOnly,
    onProgress: (event) => {
      progress(formatSnapshotResolveProgress(event));
    },
  });
}

async function loadStableDiffusion3Scheduler(
  snapshotPath: string,
): Promise<FlowMatchEulerScheduler> {
  const loadedScheduler = await loadDiffusionSchedulerFromSnapshot(snapshotPath);
  if (!(loadedScheduler.scheduler instanceof FlowMatchEulerScheduler)) {
    throw new Error("Stable Diffusion 3 proof requires a FlowMatch Euler scheduler.");
  }
  return loadedScheduler.scheduler;
}

function buildPromptOptions(cli: CliOptions): StableDiffusion3PromptConditioningOptions {
  return {
    prompt: cli.prompt,
    ...(cli.prompt2 === undefined ? {} : { prompt2: cli.prompt2 }),
    ...(cli.prompt3 === undefined ? {} : { prompt3: cli.prompt3 }),
    ...(cli.negativePrompt === undefined ? {} : { negativePrompt: cli.negativePrompt }),
    ...(cli.negativePrompt2 === undefined ? {} : { negativePrompt2: cli.negativePrompt2 }),
    ...(cli.negativePrompt3 === undefined ? {} : { negativePrompt3: cli.negativePrompt3 }),
    guidanceScale: cli.guidanceScale,
    maxSequenceLength: cli.maxSequenceLength,
    clipSkip: cli.clipSkip,
  };
}

function buildGenerationOptions(
  cli: CliOptions,
  height: number,
  width: number,
  conditioning: StableDiffusion3PromptConditioning,
  rngKey: MxArray,
  progress: (line: string) => void,
): Omit<StableDiffusion3ImageGenerationOptions, "denoiser" | "vae" | "scheduler"> {
  return {
    batchSize: conditioning.batchSize,
    height,
    width,
    conditioning: conditioning.conditioning,
    ...(conditioning.negativeConditioning === undefined
      ? {}
      : { negativeConditioning: conditioning.negativeConditioning }),
    guidanceScale: cli.guidanceScale,
    numInferenceSteps: cli.steps,
    dtype: cli.dtype,
    rngKey,
    onStep: (event) => {
      progress(`Denoise step ${event.stepIndex + 1}/${cli.steps}: sigma ${event.sigma}`);
    },
  };
}

function buildExampleResult(
  cli: CliOptions,
  snapshot: ResolvedDiffusionSnapshot,
  artifact: StableDiffusion3BmpWriteResult,
  conditioning: StableDiffusion3PromptConditioning,
  startedAt: number,
): StableDiffusion3ExampleResult {
  return {
    source: cli.source,
    snapshotPath: snapshot.directory,
    ...(snapshot.requestedRevision === undefined
      ? {}
      : { requestedRevision: snapshot.requestedRevision }),
    ...(snapshot.resolvedRevision === undefined
      ? {}
      : { resolvedRevision: snapshot.resolvedRevision }),
    pipeline: "stable-diffusion-3",
    prompt: cli.prompt,
    negativePrompt: cli.guidanceScale > 1 ? (cli.negativePrompt ?? "") : null,
    outputPath: artifact.path,
    imageSize: {
      width: artifact.width,
      height: artifact.height,
    },
    outputBytes: artifact.bytes,
    artifact,
    steps: cli.steps,
    guidanceScale: cli.guidanceScale,
    maxSequenceLength: cli.maxSequenceLength,
    clipSkip: cli.clipSkip,
    seed: cli.seed,
    dtype: cli.dtype,
    promptTruncated: conditioning.promptTruncated,
    prompt2Truncated: conditioning.prompt2Truncated,
    prompt3Truncated: conditioning.prompt3Truncated,
    negativePromptTruncated: conditioning.negativePromptTruncated,
    negativePrompt2Truncated: conditioning.negativePrompt2Truncated,
    negativePrompt3Truncated: conditioning.negativePrompt3Truncated,
    elapsedMs: Math.round((performance.now() - startedAt) * 100) / 100,
  };
}

export async function runStableDiffusion3Example(
  cli: CliOptions,
  progress: (line: string) => void,
): Promise<StableDiffusion3ExampleResult> {
  const startedAt = performance.now();
  printRunIntro(cli, progress);
  const snapshot = await resolveExampleSnapshot(cli, progress);
  const snapshotPath = snapshot.directory;

  progress("Loading Stable Diffusion 3 snapshot manifest...");
  const manifest = await loadDiffusionSnapshotManifest(snapshotPath);
  if (manifest.modelIndex.kind !== "stable-diffusion-3") {
    throw new Error(`Stable Diffusion 3 proof does not support ${manifest.modelIndex.kind}.`);
  }
  const configs = await loadStableDiffusion3ComponentConfigs(manifest);
  const defaultSize = defaultImageSize(configs.transformer.sampleSize, configs.vae.vaeScaleFactor);
  const height = cli.height ?? defaultSize.height;
  const width = cli.width ?? defaultSize.width;
  validateImageSize(height, width, configs.vae.vaeScaleFactor * configs.transformer.patchSize);

  progress("Loading Stable Diffusion 3 scheduler...");
  const scheduler = await loadStableDiffusion3Scheduler(snapshotPath);

  progress("Loading Stable Diffusion 3 transformer...");
  using transformer = await loadStableDiffusion3TransformerFromSnapshot(manifest);

  progress("Loading Stable Diffusion 3 VAE...");
  using vae = await loadStableDiffusion3AutoencoderFromSnapshot(manifest);

  progress("Loading Stable Diffusion 3 prompt conditioner...");
  using conditioner = await loadStableDiffusion3PromptConditionerFromSnapshot(snapshotPath);
  const promptOptions = buildPromptOptions(cli);

  progress("Encoding CLIP/T5 prompt conditioning...");
  using conditioning = conditioner.encodePrompt(promptOptions);
  using rngKey = random.key(cli.seed);

  progress(`Denoising ${width}x${height} image...`);
  const generationOptions = buildGenerationOptions(
    cli,
    height,
    width,
    conditioning,
    rngKey,
    progress,
  );
  using image = generateStableDiffusion3Image({
    ...generationOptions,
    denoiser: transformer,
    vae,
    scheduler,
  });

  const artifact = writeStableDiffusion3Bmp(image, cli.outputPath);
  return buildExampleResult(cli, snapshot, artifact, conditioning, startedAt);
}

export function formatSuccess(report: StableDiffusion3ExampleResult): string {
  return [
    "stable_diffusion_3_example:",
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
    `  clip_skip: ${report.clipSkip}`,
    `  seed: ${report.seed}`,
    `  dtype: ${quoteScalar(report.dtype)}`,
    `  prompt_truncated: ${report.promptTruncated}`,
    `  prompt_2_truncated: ${report.prompt2Truncated}`,
    `  prompt_3_truncated: ${report.prompt3Truncated}`,
    `  negative_prompt_truncated: ${report.negativePromptTruncated}`,
    `  negative_prompt_2_truncated: ${report.negativePrompt2Truncated}`,
    `  negative_prompt_3_truncated: ${report.negativePrompt3Truncated}`,
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
    '  "Run `bun run examples/stable-diffusion-3/index.ts --help` for options"',
  ].join("\n");
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export async function runStableDiffusion3ExampleCommand(
  argv: readonly string[],
  runtime: StableDiffusion3ExampleRuntime = {},
): Promise<number> {
  const stdout = runtime.stdout ?? console.log;
  const stderr = runtime.stderr ?? console.error;
  let command: CliCommand;
  try {
    command = parseCommand(argv);
  } catch (error) {
    stdout(formatError(errorMessage(error), "usage"));
    return error instanceof StableDiffusion3ExampleUsageError ? 2 : 1;
  }

  if (command.kind === "help") {
    stdout(formatUsage());
    return 0;
  }

  const acquireLock =
    runtime.acquireLock ?? (() => acquireRuntimeCommandLock("example:stable-diffusion-3"));
  const runExample = runtime.runExample ?? runStableDiffusion3Example;
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
  const exitCode = await runStableDiffusion3ExampleCommand(Bun.argv.slice(2));
  process.exit(exitCode);
}
