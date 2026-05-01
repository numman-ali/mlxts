#!/usr/bin/env bun

import type { DType } from "@mlxts/core";
import { random } from "@mlxts/core";
import {
  type DiffusionSnapshotResolveProgressEvent,
  FlowMatchEulerScheduler,
  type FluxImageGenerationOptions,
  generateFluxImage,
  loadDiffusionSchedulerFromSnapshot,
  loadDiffusionSnapshotManifest,
  loadFluxAutoencoderFromSnapshot,
  loadFluxTransformerFromSnapshot,
  resolveDiffusionSnapshot,
} from "@mlxts/diffusion";

import { acquireRuntimeCommandLock } from "../../scripts/runtime-command-lock";
import { loadFluxPromptConditionerFromSnapshot } from "./conditioning";
import type { FluxPromptConditioningOptions } from "./conditioning-types";
import { writeFluxBmp } from "./image-output";

type CliOptions = {
  source: string;
  revision?: string;
  cacheDir?: string;
  hfToken?: string;
  localFilesOnly: boolean;
  prompt: string;
  prompt2?: string;
  outputPath: string;
  steps: number;
  height: number;
  width: number;
  guidanceScale?: number;
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
  localFilesOnly: boolean;
  prompt?: string;
  prompt2?: string;
  outputPath: string;
  steps: number;
  height: number;
  width: number;
  guidanceScale?: number;
  maxSequenceLength: number;
  seed: number;
  dtype: CliOptions["dtype"];
  json: boolean;
};

type FluxExampleResult = {
  source: string;
  snapshotPath: string;
  requestedRevision?: string;
  resolvedRevision?: string;
  pipeline: "flux";
  prompt: string;
  prompt2: string;
  outputPath: string;
  imageSize: { width: number; height: number };
  outputBytes: number;
  steps: number;
  guidanceScale: number | null;
  maxSequenceLength: number;
  seed: number;
  dtype: string;
  promptTruncated: boolean;
  prompt2Truncated: boolean;
  elapsedMs: number;
};

type RuntimeLock = {
  [Symbol.dispose](): void;
};

type FluxExampleRuntime = {
  stdout?: (text: string) => void;
  stderr?: (text: string) => void;
  acquireLock?: () => RuntimeLock;
  runExample?: (cli: CliOptions, progress: (line: string) => void) => Promise<FluxExampleResult>;
};

class FluxExampleUsageError extends Error {}

function quoteScalar(value: string | number | boolean | null): string {
  return typeof value === "string" ? JSON.stringify(value) : String(value);
}

export function formatUsage(): string {
  return [
    "description: Run one FLUX text-to-image proof and write a BMP artifact",
    "usage[2]:",
    '  bun run examples/flux/index.ts <snapshot-source> --prompt "a red apple"',
    '  bun run examples/flux/index.ts black-forest-labs/FLUX.1-schnell --local-files-only --prompt "a quiet library" --output .tmp/flux.bmp --steps 4',
    "arguments[1]{name,description}:",
    '  "snapshot-source","Local Diffusers snapshot directory or Hugging Face model id"',
    "options[16]{flag,description}:",
    '  "--revision <rev>","Hub revision; default main"',
    '  "--cache-dir <path>","Hub cache directory; default Hugging Face cache"',
    '  "--hf-token <token>","Hub access token; defaults to HF token environment or cache file"',
    '  "--local-files-only","Use only an already-cached Hub snapshot"',
    '  "--prompt <text>","Required positive prompt for CLIP and T5"',
    '  "--prompt-2 <text>","Optional T5 prompt; defaults to --prompt"',
    '  "--output <path>","BMP output path; default .tmp/flux/sample.bmp"',
    '  "--steps <n>","Inference steps; default 4"',
    '  "--height <n>","Output height; default 1024"',
    '  "--width <n>","Output width; default 1024"',
    '  "--guidance-scale <n>","Guidance for FLUX dev-style checkpoints; auto 3.5 when required"',
    '  "--max-sequence-length <n>","T5 token length, 1..512; default 512"',
    '  "--seed <n>","RNG seed; default 0"',
    '  "--dtype <float16|float32|bfloat16>","Latent dtype; default float16"',
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
    throw new FluxExampleUsageError(`Missing value for ${flag}.`);
  }
  return value;
}

function readNumberFlag(flag: string, value: string | undefined): number {
  const raw = readStringFlag(flag, value);
  const parsed = Number(raw);
  if (!Number.isFinite(parsed)) {
    throw new FluxExampleUsageError(`Expected ${flag} to be a finite number.`);
  }
  return parsed;
}

function readPositiveIntegerFlag(flag: string, value: string | undefined): number {
  const parsed = readNumberFlag(flag, value);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new FluxExampleUsageError(`Expected ${flag} to be a positive integer.`);
  }
  return parsed;
}

function readIntegerFlag(flag: string, value: string | undefined): number {
  const parsed = readNumberFlag(flag, value);
  if (!Number.isInteger(parsed)) {
    throw new FluxExampleUsageError(`Expected ${flag} to be an integer.`);
  }
  return parsed;
}

function readNonNegativeIntegerFlag(flag: string, value: string | undefined): number {
  const parsed = readIntegerFlag(flag, value);
  if (parsed < 0) {
    throw new FluxExampleUsageError(`Expected ${flag} to be non-negative.`);
  }
  return parsed;
}

function readDTypeFlag(value: string | undefined): CliOptions["dtype"] {
  const raw = readStringFlag("--dtype", value);
  if (raw === "float16" || raw === "float32" || raw === "bfloat16") {
    return raw;
  }
  throw new FluxExampleUsageError("Expected --dtype to be float16, float32, or bfloat16.");
}

function applyFlag(argv: readonly string[], index: number, draft: CliOptionsDraft): number {
  const arg = argv[index];
  switch (arg) {
    case "--prompt":
      draft.prompt = readStringFlag(arg, argv[index + 1]);
      return index + 1;
    case "--prompt-2":
      draft.prompt2 = readStringFlag(arg, argv[index + 1]);
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
      throw new FluxExampleUsageError(
        arg === undefined ? "Missing argument." : `Unknown argument: ${arg}`,
      );
  }
}

function validateCliOptions(draft: CliOptionsDraft): void {
  if (draft.prompt === undefined) {
    throw new FluxExampleUsageError("Missing required --prompt <text>.");
  }
  if (draft.guidanceScale !== undefined && draft.guidanceScale < 0) {
    throw new FluxExampleUsageError("--guidance-scale must be finite and non-negative.");
  }
  if (draft.maxSequenceLength > 512) {
    throw new FluxExampleUsageError("--max-sequence-length must be no greater than 512.");
  }
  if (!draft.outputPath.toLowerCase().endsWith(".bmp")) {
    throw new FluxExampleUsageError("--output must end with .bmp.");
  }
}

export function parseCommand(argv: readonly string[]): CliCommand {
  if (argv.some((arg) => arg === "--help" || arg === "-h")) {
    return { kind: "help" };
  }

  const source = argv[0];
  if (source === undefined || source.trim() === "" || source.startsWith("--")) {
    throw new FluxExampleUsageError("Missing snapshot source.");
  }

  const draft: CliOptionsDraft = {
    localFilesOnly: false,
    outputPath: ".tmp/flux/sample.bmp",
    steps: 4,
    height: 1024,
    width: 1024,
    maxSequenceLength: 512,
    seed: 0,
    dtype: "float16",
    json: false,
  };

  for (let index = 1; index < argv.length; index += 1) {
    index = applyFlag(argv, index, draft);
  }
  validateCliOptions(draft);
  const prompt = draft.prompt;
  if (prompt === undefined) {
    throw new FluxExampleUsageError("Missing required --prompt <text>.");
  }

  const options: CliOptions = {
    source,
    localFilesOnly: draft.localFilesOnly,
    prompt,
    outputPath: draft.outputPath,
    steps: draft.steps,
    height: draft.height,
    width: draft.width,
    maxSequenceLength: draft.maxSequenceLength,
    seed: draft.seed,
    dtype: draft.dtype,
    json: draft.json,
  };
  if (draft.prompt2 !== undefined) {
    options.prompt2 = draft.prompt2;
  }
  if (draft.revision !== undefined) {
    options.revision = draft.revision;
  }
  if (draft.cacheDir !== undefined) {
    options.cacheDir = draft.cacheDir;
  }
  if (draft.hfToken !== undefined) {
    options.hfToken = draft.hfToken;
  }
  if (draft.guidanceScale !== undefined) {
    options.guidanceScale = draft.guidanceScale;
  }
  return { kind: "run", options };
}

export function parseArgs(argv: readonly string[]): CliOptions {
  const command = parseCommand(argv);
  if (command.kind === "help") {
    throw new FluxExampleUsageError("Help is not an image proof command.");
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
  writeLine(`Prompt: ${cli.prompt}`);
  writeLine(`Output: ${cli.outputPath}`);
  writeLine(`Steps: ${cli.steps}`);
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

export function resolveFluxGuidanceScale(
  guidanceScale: number | undefined,
  guidanceRequired: boolean,
): number | undefined {
  if (!guidanceRequired) {
    if (guidanceScale !== undefined) {
      throw new Error("This FLUX checkpoint does not support --guidance-scale.");
    }
    return undefined;
  }
  return guidanceScale ?? 3.5;
}

export async function runFluxExample(
  cli: CliOptions,
  progress: (line: string) => void,
): Promise<FluxExampleResult> {
  const startedAt = performance.now();
  printRunIntro(cli, progress);
  const snapshot = await resolveDiffusionSnapshot(cli.source, {
    ...(cli.revision === undefined ? {} : { revision: cli.revision }),
    ...(cli.cacheDir === undefined ? {} : { cacheDir: cli.cacheDir }),
    ...(cli.hfToken === undefined ? {} : { accessToken: cli.hfToken }),
    localFilesOnly: cli.localFilesOnly,
    onProgress: (event) => {
      progress(formatSnapshotResolveProgress(event));
    },
  });
  const snapshotPath = snapshot.directory;

  progress("Loading FLUX snapshot manifest...");
  const manifest = await loadDiffusionSnapshotManifest(snapshotPath);
  if (manifest.modelIndex.kind !== "flux") {
    throw new Error(`FLUX proof does not support ${manifest.modelIndex.kind}.`);
  }

  progress("Loading FLUX scheduler...");
  const loadedScheduler = await loadDiffusionSchedulerFromSnapshot(snapshotPath);
  if (!(loadedScheduler.scheduler instanceof FlowMatchEulerScheduler)) {
    throw new Error("FLUX proof requires a FlowMatch Euler scheduler.");
  }
  const scheduler = loadedScheduler.scheduler;

  progress("Loading FLUX transformer...");
  using transformer = await loadFluxTransformerFromSnapshot(manifest);
  const guidanceScale = resolveFluxGuidanceScale(
    cli.guidanceScale,
    transformer.guidanceEmbedding !== null,
  );

  progress("Loading FLUX VAE...");
  using vae = await loadFluxAutoencoderFromSnapshot(manifest);

  progress("Loading FLUX prompt conditioner...");
  using conditioner = await loadFluxPromptConditionerFromSnapshot(snapshotPath);
  const promptOptions: FluxPromptConditioningOptions = {
    prompt: cli.prompt,
    maxSequenceLength: cli.maxSequenceLength,
    guidanceDType: cli.dtype,
  };
  if (cli.prompt2 !== undefined) {
    promptOptions.prompt2 = cli.prompt2;
  }
  if (guidanceScale !== undefined) {
    promptOptions.guidanceScale = guidanceScale;
  }

  progress("Encoding prompt conditioning...");
  using conditioning = conditioner.encodePrompt(promptOptions);
  using rngKey = random.key(cli.seed);

  progress(`Denoising ${cli.width}x${cli.height} image...`);
  const generationOptions: Omit<FluxImageGenerationOptions, "denoiser" | "vae" | "scheduler"> = {
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
  using image = generateFluxImage({
    ...generationOptions,
    denoiser: transformer,
    vae,
    scheduler,
  });

  const artifact = writeFluxBmp(image, cli.outputPath);
  return {
    source: cli.source,
    snapshotPath,
    ...(snapshot.requestedRevision === undefined
      ? {}
      : { requestedRevision: snapshot.requestedRevision }),
    ...(snapshot.resolvedRevision === undefined
      ? {}
      : { resolvedRevision: snapshot.resolvedRevision }),
    pipeline: "flux",
    prompt: cli.prompt,
    prompt2: cli.prompt2 ?? cli.prompt,
    outputPath: artifact.path,
    imageSize: {
      width: artifact.width,
      height: artifact.height,
    },
    outputBytes: artifact.bytes,
    steps: cli.steps,
    guidanceScale: guidanceScale ?? null,
    maxSequenceLength: cli.maxSequenceLength,
    seed: cli.seed,
    dtype: cli.dtype,
    promptTruncated: conditioning.promptTruncated,
    prompt2Truncated: conditioning.prompt2Truncated,
    elapsedMs: Math.round((performance.now() - startedAt) * 100) / 100,
  };
}

export function formatSuccess(report: FluxExampleResult): string {
  return [
    "flux_example:",
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
    `  guidance_scale: ${quoteScalar(report.guidanceScale)}`,
    `  max_sequence_length: ${report.maxSequenceLength}`,
    `  seed: ${report.seed}`,
    `  dtype: ${quoteScalar(report.dtype)}`,
    `  prompt_truncated: ${report.promptTruncated}`,
    `  prompt_2_truncated: ${report.prompt2Truncated}`,
    `  elapsed_ms: ${report.elapsedMs}`,
    `  prompt: ${quoteScalar(report.prompt)}`,
    `  prompt_2: ${quoteScalar(report.prompt2)}`,
  ].join("\n");
}

function formatError(message: string, code: "usage" | "runtime"): string {
  return [
    "error:",
    `  code: ${quoteScalar(code)}`,
    `  message: ${quoteScalar(message)}`,
    "help[1]:",
    '  "Run `bun run examples/flux/index.ts --help` for options"',
  ].join("\n");
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export async function runFluxExampleCommand(
  argv: readonly string[],
  runtime: FluxExampleRuntime = {},
): Promise<number> {
  const stdout = runtime.stdout ?? console.log;
  const stderr = runtime.stderr ?? console.error;
  let command: CliCommand;
  try {
    command = parseCommand(argv);
  } catch (error) {
    stdout(formatError(errorMessage(error), "usage"));
    return error instanceof FluxExampleUsageError ? 2 : 1;
  }

  if (command.kind === "help") {
    stdout(formatUsage());
    return 0;
  }

  const acquireLock = runtime.acquireLock ?? (() => acquireRuntimeCommandLock("example:flux"));
  const runExample = runtime.runExample ?? runFluxExample;
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
  const exitCode = await runFluxExampleCommand(Bun.argv.slice(2));
  process.exit(exitCode);
}
