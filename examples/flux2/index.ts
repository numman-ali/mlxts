#!/usr/bin/env bun

import type { DType } from "@mlxts/core";
import { random } from "@mlxts/core";
import {
  type DiffusionSnapshotResolveProgressEvent,
  FlowMatchEulerScheduler,
  type Flux2KleinImageGenerationOptions,
  generateFlux2KleinImage,
  loadDiffusionSchedulerFromSnapshot,
  loadDiffusionSnapshotManifest,
  loadFlux2KleinAutoencoderFromSnapshot,
  loadFlux2KleinComponentConfigs,
  loadFlux2KleinTransformerFromSnapshot,
  resolveDiffusionSnapshot,
} from "@mlxts/diffusion";

import { acquireRuntimeCommandLock } from "../../scripts/runtime-command-lock";
import { loadFlux2KleinPromptConditionerFromSnapshot } from "./conditioning";
import {
  FLUX2_KLEIN_DEFAULT_GUIDANCE_SCALE,
  FLUX2_KLEIN_MAX_SEQUENCE_LENGTH,
} from "./conditioning-runtime";
import type { Flux2KleinPromptConditioningOptions } from "./conditioning-types";
import { type Flux2KleinBmpWriteResult, writeFlux2KleinBmp } from "./image-output";

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
  negativePrompt?: string;
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

type Flux2KleinExampleResult = {
  source: string;
  snapshotPath: string;
  requestedRevision?: string;
  resolvedRevision?: string;
  pipeline: "flux2-klein";
  prompt: string;
  negativePrompt: string | null;
  outputPath: string;
  imageSize: { width: number; height: number };
  outputBytes: number;
  artifact: Flux2KleinBmpWriteResult;
  steps: number;
  guidanceScale: number;
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

type Flux2KleinExampleRuntime = {
  stdout?: (text: string) => void;
  stderr?: (text: string) => void;
  acquireLock?: () => RuntimeLock;
  runExample?: (
    cli: CliOptions,
    progress: (line: string) => void,
  ) => Promise<Flux2KleinExampleResult>;
};

class Flux2KleinExampleUsageError extends Error {}

function quoteScalar(value: string | number | boolean | null): string {
  return typeof value === "string" ? JSON.stringify(value) : String(value);
}

export function formatUsage(): string {
  return [
    "description: Run one FLUX.2 Klein text-to-image proof and write a BMP artifact",
    "usage[2]:",
    '  bun run examples/flux2/index.ts <snapshot-source> --prompt "a red apple"',
    '  bun run examples/flux2/index.ts black-forest-labs/FLUX.2-klein-4B --local-files-only --prompt "a quiet library" --output .tmp/flux2/sample.bmp --steps 4 --guidance-scale 1',
    "arguments[1]{name,description}:",
    '  "snapshot-source","Local Diffusers snapshot directory or Hugging Face model id"',
    "options[17]{flag,description}:",
    '  "--revision <rev>","Hub revision; default main"',
    '  "--cache-dir <path>","Hub cache directory; default Hugging Face cache"',
    '  "--hf-token <token>","Hub access token; defaults to HF token environment or cache file"',
    '  "--variant <name>","Optional Hub weight filename variant, for example fp16"',
    '  "--local-files-only","Use only an already-cached Hub snapshot"',
    '  "--prompt <text>","Required positive prompt for Qwen3"',
    '  "--negative-prompt <text>","Negative prompt for CFG; default empty string when CFG is active"',
    '  "--output <path>","BMP output path; default .tmp/flux2/sample.bmp"',
    '  "--steps <n>","Inference steps; default 50"',
    '  "--height <n>","Output height; default 1024"',
    '  "--width <n>","Output width; default 1024"',
    '  "--guidance-scale <n>","FLUX.2 guidance scale; default 4, use 1 to disable negative conditioning"',
    `  "--max-sequence-length <n>","Qwen3 token length, 1..${FLUX2_KLEIN_MAX_SEQUENCE_LENGTH}; default ${FLUX2_KLEIN_MAX_SEQUENCE_LENGTH}"`,
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

function readNonEmptyStringFlag(flag: string, value: string | undefined): string {
  if (value === undefined || value.trim() === "" || value.startsWith("--")) {
    throw new Flux2KleinExampleUsageError(`Missing value for ${flag}.`);
  }
  return value;
}

function readStringFlag(flag: string, value: string | undefined): string {
  if (value === undefined || value.startsWith("--")) {
    throw new Flux2KleinExampleUsageError(`Missing value for ${flag}.`);
  }
  return value;
}

function readNumberFlag(flag: string, value: string | undefined): number {
  const raw = readNonEmptyStringFlag(flag, value);
  const parsed = Number(raw);
  if (!Number.isFinite(parsed)) {
    throw new Flux2KleinExampleUsageError(`Expected ${flag} to be a finite number.`);
  }
  return parsed;
}

function readPositiveIntegerFlag(flag: string, value: string | undefined): number {
  const parsed = readNumberFlag(flag, value);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Flux2KleinExampleUsageError(`Expected ${flag} to be a positive integer.`);
  }
  return parsed;
}

function readIntegerFlag(flag: string, value: string | undefined): number {
  const parsed = readNumberFlag(flag, value);
  if (!Number.isInteger(parsed)) {
    throw new Flux2KleinExampleUsageError(`Expected ${flag} to be an integer.`);
  }
  return parsed;
}

function readNonNegativeIntegerFlag(flag: string, value: string | undefined): number {
  const parsed = readIntegerFlag(flag, value);
  if (parsed < 0) {
    throw new Flux2KleinExampleUsageError(`Expected ${flag} to be non-negative.`);
  }
  return parsed;
}

function readDTypeFlag(value: string | undefined): CliOptions["dtype"] {
  const raw = readNonEmptyStringFlag("--dtype", value);
  if (raw === "float16" || raw === "float32" || raw === "bfloat16") {
    return raw;
  }
  throw new Flux2KleinExampleUsageError("Expected --dtype to be float16, float32, or bfloat16.");
}

function applyFlag(argv: readonly string[], index: number, draft: CliOptionsDraft): number {
  const arg = argv[index];
  switch (arg) {
    case "--prompt":
      draft.prompt = readNonEmptyStringFlag(arg, argv[index + 1]);
      return index + 1;
    case "--negative-prompt":
      draft.negativePrompt = readStringFlag(arg, argv[index + 1]);
      return index + 1;
    case "--revision":
      draft.revision = readNonEmptyStringFlag(arg, argv[index + 1]);
      return index + 1;
    case "--cache-dir":
      draft.cacheDir = readNonEmptyStringFlag(arg, argv[index + 1]);
      return index + 1;
    case "--hf-token":
      draft.hfToken = readNonEmptyStringFlag(arg, argv[index + 1]);
      return index + 1;
    case "--variant":
      draft.variant = readNonEmptyStringFlag(arg, argv[index + 1]);
      return index + 1;
    case "--local-files-only":
      draft.localFilesOnly = true;
      return index;
    case "--output":
      draft.outputPath = readNonEmptyStringFlag(arg, argv[index + 1]);
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
      throw new Flux2KleinExampleUsageError(
        arg === undefined ? "Missing argument." : `Unknown argument: ${arg}`,
      );
  }
}

function validateCliOptions(draft: CliOptionsDraft): void {
  if (draft.prompt === undefined) {
    throw new Flux2KleinExampleUsageError("Missing required --prompt <text>.");
  }
  if (!Number.isFinite(draft.guidanceScale) || draft.guidanceScale <= 0) {
    throw new Flux2KleinExampleUsageError("--guidance-scale must be a positive finite number.");
  }
  if (draft.guidanceScale <= 1 && draft.negativePrompt !== undefined) {
    throw new Flux2KleinExampleUsageError(
      "--negative-prompt requires --guidance-scale greater than 1.",
    );
  }
  if (draft.maxSequenceLength > FLUX2_KLEIN_MAX_SEQUENCE_LENGTH) {
    throw new Flux2KleinExampleUsageError(
      `--max-sequence-length must be no greater than ${FLUX2_KLEIN_MAX_SEQUENCE_LENGTH}.`,
    );
  }
  if (draft.height % 16 !== 0 || draft.width % 16 !== 0) {
    throw new Flux2KleinExampleUsageError("--height and --width must be divisible by 16.");
  }
  if (!draft.outputPath.toLowerCase().endsWith(".bmp")) {
    throw new Flux2KleinExampleUsageError("--output must end with .bmp.");
  }
}

export function parseCommand(argv: readonly string[]): CliCommand {
  if (argv.some((arg) => arg === "--help" || arg === "-h")) {
    return { kind: "help" };
  }

  const source = argv[0];
  if (source === undefined || source.trim() === "" || source.startsWith("--")) {
    throw new Flux2KleinExampleUsageError("Missing snapshot source.");
  }

  const draft: CliOptionsDraft = {
    localFilesOnly: false,
    outputPath: ".tmp/flux2/sample.bmp",
    steps: 50,
    height: 1024,
    width: 1024,
    guidanceScale: FLUX2_KLEIN_DEFAULT_GUIDANCE_SCALE,
    maxSequenceLength: FLUX2_KLEIN_MAX_SEQUENCE_LENGTH,
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
    throw new Flux2KleinExampleUsageError("Missing required --prompt <text>.");
  }

  const options: CliOptions = {
    source,
    localFilesOnly: draft.localFilesOnly,
    prompt,
    ...(draft.negativePrompt === undefined ? {} : { negativePrompt: draft.negativePrompt }),
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
    throw new Flux2KleinExampleUsageError("Help is not an image proof command.");
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

export function resolveFlux2KleinNegativePrompt(
  guidanceScale: number,
  isDistilled: boolean,
  negativePrompt: string | undefined,
): string | undefined {
  if (!Number.isFinite(guidanceScale) || guidanceScale <= 0) {
    throw new Flux2KleinExampleUsageError("--guidance-scale must be a positive finite number.");
  }
  if (guidanceScale <= 1 || isDistilled) {
    if (negativePrompt !== undefined) {
      throw new Flux2KleinExampleUsageError(
        isDistilled
          ? "--negative-prompt is ignored for distilled FLUX.2 Klein snapshots."
          : "--negative-prompt requires --guidance-scale greater than 1.",
      );
    }
    return undefined;
  }
  return negativePrompt ?? "";
}

export async function runFlux2KleinExample(
  cli: CliOptions,
  progress: (line: string) => void,
): Promise<Flux2KleinExampleResult> {
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

  progress("Loading FLUX.2 Klein snapshot manifest...");
  const manifest = await loadDiffusionSnapshotManifest(snapshotPath);
  if (manifest.modelIndex.kind !== "flux2-klein") {
    throw new Error(`FLUX.2 Klein proof does not support ${manifest.modelIndex.kind}.`);
  }
  const configs = await loadFlux2KleinComponentConfigs(manifest);
  const negativePrompt = resolveFlux2KleinNegativePrompt(
    cli.guidanceScale,
    configs.isDistilled,
    cli.negativePrompt,
  );

  progress("Loading FLUX.2 Klein scheduler...");
  const loadedScheduler = await loadDiffusionSchedulerFromSnapshot(snapshotPath);
  if (!(loadedScheduler.scheduler instanceof FlowMatchEulerScheduler)) {
    throw new Error("FLUX.2 Klein proof requires a FlowMatch Euler scheduler.");
  }
  const scheduler = loadedScheduler.scheduler;

  progress("Loading FLUX.2 Klein transformer...");
  using transformer = await loadFlux2KleinTransformerFromSnapshot(manifest);

  progress("Loading FLUX.2 Klein VAE...");
  using vae = await loadFlux2KleinAutoencoderFromSnapshot(manifest);

  progress("Loading FLUX.2 Klein prompt conditioner...");
  using conditioner = await loadFlux2KleinPromptConditionerFromSnapshot(snapshotPath);
  const promptOptions: Flux2KleinPromptConditioningOptions = {
    prompt: cli.prompt,
    ...(negativePrompt === undefined ? {} : { negativePrompt }),
    guidanceScale: cli.guidanceScale,
    maxSequenceLength: cli.maxSequenceLength,
  };

  progress("Encoding Qwen3 prompt conditioning...");
  using conditioning = conditioner.encodePrompt(promptOptions);
  using rngKey = random.key(cli.seed);

  progress(`Denoising ${cli.width}x${cli.height} image...`);
  const generationOptions: Omit<
    Flux2KleinImageGenerationOptions,
    "denoiser" | "vae" | "scheduler"
  > = {
    batchSize: conditioning.batchSize,
    height: cli.height,
    width: cli.width,
    conditioning: conditioning.conditioning,
    numInferenceSteps: cli.steps,
    dtype: cli.dtype,
    rngKey,
    isDistilled: configs.isDistilled,
    onStep: (event) => {
      progress(`Denoise step ${event.stepIndex + 1}/${cli.steps}: sigma ${event.sigma}`);
    },
  };
  using image = generateFlux2KleinImage({
    ...generationOptions,
    denoiser: transformer,
    vae,
    scheduler,
  });

  const artifact = writeFlux2KleinBmp(image, cli.outputPath);
  return {
    source: cli.source,
    snapshotPath,
    ...(snapshot.requestedRevision === undefined
      ? {}
      : { requestedRevision: snapshot.requestedRevision }),
    ...(snapshot.resolvedRevision === undefined
      ? {}
      : { resolvedRevision: snapshot.resolvedRevision }),
    pipeline: "flux2-klein",
    prompt: cli.prompt,
    negativePrompt: negativePrompt ?? null,
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
    seed: cli.seed,
    dtype: cli.dtype,
    promptTruncated: conditioning.promptTruncated,
    negativePromptTruncated: conditioning.negativePromptTruncated,
    elapsedMs: Math.round((performance.now() - startedAt) * 100) / 100,
  };
}

export function formatSuccess(report: Flux2KleinExampleResult): string {
  return [
    "flux2_klein_example:",
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
    '  "Run `bun run examples/flux2/index.ts --help` for options"',
  ].join("\n");
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export async function runFlux2KleinExampleCommand(
  argv: readonly string[],
  runtime: Flux2KleinExampleRuntime = {},
): Promise<number> {
  const stdout = runtime.stdout ?? console.log;
  const stderr = runtime.stderr ?? console.error;
  let command: CliCommand;
  try {
    command = parseCommand(argv);
  } catch (error) {
    stdout(formatError(errorMessage(error), "usage"));
    return error instanceof Flux2KleinExampleUsageError ? 2 : 1;
  }

  if (command.kind === "help") {
    stdout(formatUsage());
    return 0;
  }

  const acquireLock = runtime.acquireLock ?? (() => acquireRuntimeCommandLock("example:flux2"));
  const runExample = runtime.runExample ?? runFlux2KleinExample;
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
  const exitCode = await runFlux2KleinExampleCommand(Bun.argv.slice(2));
  process.exit(exitCode);
}
