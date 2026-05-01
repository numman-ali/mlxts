#!/usr/bin/env bun

import type { DType } from "@mlxts/core";
import { random } from "@mlxts/core";
import {
  createLtxVideoInitialLatents,
  type DiffusionSnapshotResolveProgressEvent,
  decodeLtxVideoLatents,
  denoiseLtxVideoLatents,
  FlowMatchEulerScheduler,
  type LtxVideoDenoiseOptions,
  loadDiffusionSchedulerFromSnapshot,
  loadDiffusionSnapshotManifest,
  loadLtxComponentConfigs,
  loadLtxVideoAutoencoderFromSnapshot,
  loadLtxVideoTransformerFromSnapshot,
  ltxVideoLatentShape,
  resolveDiffusionSnapshot,
} from "@mlxts/diffusion";

import { acquireRuntimeCommandLock } from "../../scripts/runtime-command-lock";
import { loadLtxVideoPromptConditionerFromSnapshot } from "./conditioning";
import type { LtxVideoPromptConditioningOptions } from "./conditioning-types";
import { type LtxVideoPreviewBmpWriteResult, writeLtxVideoPreviewBmp } from "./video-output";

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
  frames: number;
  frameRate: number;
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
  frames: number;
  frameRate: number;
  guidanceScale: number;
  maxSequenceLength: number;
  seed: number;
  dtype: CliOptions["dtype"];
  json: boolean;
};

type LtxVideoExampleResult = {
  source: string;
  snapshotPath: string;
  requestedRevision?: string;
  resolvedRevision?: string;
  pipeline: "ltx-video";
  prompt: string;
  negativePrompt: string | null;
  outputPath: string;
  imageSize: { width: number; height: number };
  videoSize: { width: number; height: number; frames: number; channels: number };
  latentSize: { width: number; height: number; frames: number; channels: number };
  outputBytes: number;
  artifact: LtxVideoPreviewBmpWriteResult;
  steps: number;
  guidanceScale: number;
  maxSequenceLength: number;
  requestedFrames: number;
  decodedFrames: number;
  frameRate: number;
  seed: number;
  dtype: string;
  promptTruncated: boolean;
  negativePromptTruncated: boolean;
  elapsedMs: number;
};

type RuntimeLock = {
  [Symbol.dispose](): void;
};

type LtxVideoExampleRuntime = {
  stdout?: (text: string) => void;
  stderr?: (text: string) => void;
  acquireLock?: () => RuntimeLock;
  runExample?: (
    cli: CliOptions,
    progress: (line: string) => void,
  ) => Promise<LtxVideoExampleResult>;
};

class LtxVideoExampleUsageError extends Error {}

function quoteScalar(value: string | number | boolean | null): string {
  return typeof value === "string" ? JSON.stringify(value) : String(value);
}

export function formatUsage(): string {
  return [
    "description: Run one classic LTX-Video text-to-video proof and write a BMP preview artifact",
    "usage[2]:",
    '  bun run examples/ltx-video/index.ts <snapshot-source> --prompt "a red apple"',
    '  bun run examples/ltx-video/index.ts Lightricks/LTX-Video --local-files-only --prompt "a quiet library" --output .tmp/ltx-video/preview.bmp --steps 4',
    "arguments[1]{name,description}:",
    '  "snapshot-source","Local Diffusers snapshot directory or Hugging Face model id"',
    "options[18]{flag,description}:",
    '  "--revision <rev>","Hub revision; default main"',
    '  "--cache-dir <path>","Hub cache directory; default Hugging Face cache"',
    '  "--hf-token <token>","Hub access token; defaults to HF token environment or cache file"',
    '  "--variant <name>","Optional Hub weight filename variant, for example fp16"',
    '  "--local-files-only","Use only an already-cached Hub snapshot"',
    '  "--prompt <text>","Required positive prompt for T5"',
    '  "--negative-prompt <text>","Negative prompt for CFG; default empty string when guidance is active"',
    '  "--output <path>","BMP preview output path; default .tmp/ltx-video/preview.bmp"',
    '  "--steps <n>","Inference steps; default 4"',
    '  "--height <n>","Output frame height; default 128"',
    '  "--width <n>","Output frame width; default 128"',
    '  "--frames <n>","Output frame count; default 9"',
    '  "--frame-rate <n>","Frame rate used for LTX RoPE scaling; default 25"',
    '  "--guidance-scale <n>","Classifier-free guidance scale; default 3, use 1 to disable negative conditioning"',
    '  "--max-sequence-length <n>","T5 token length, 1..128; default 128"',
    '  "--seed <n>","RNG seed; default 0"',
    '  "--dtype <float16|float32|bfloat16>","Latent dtype; default float16"',
    '  "--json","Emit final result as JSON"',
    '  "--help","Show this help"',
    "exit_codes[3]{code,meaning}:",
    '  0,"video proof passed or help"',
    '  1,"runtime or generation failure"',
    '  2,"usage error"',
  ].join("\n");
}

function readStringFlag(flag: string, value: string | undefined): string {
  if (value === undefined || value === "" || value.startsWith("--")) {
    throw new LtxVideoExampleUsageError(`Missing value for ${flag}.`);
  }
  return value;
}

function readRequiredTextFlag(flag: string, value: string | undefined): string {
  const raw = readStringFlag(flag, value);
  if (raw.trim() === "") {
    throw new LtxVideoExampleUsageError(`Missing value for ${flag}.`);
  }
  return raw;
}

function readNumberFlag(flag: string, value: string | undefined): number {
  const raw = readStringFlag(flag, value);
  const parsed = Number(raw);
  if (!Number.isFinite(parsed)) {
    throw new LtxVideoExampleUsageError(`Expected ${flag} to be a finite number.`);
  }
  return parsed;
}

function readPositiveIntegerFlag(flag: string, value: string | undefined): number {
  const parsed = readNumberFlag(flag, value);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new LtxVideoExampleUsageError(`Expected ${flag} to be a positive integer.`);
  }
  return parsed;
}

function readIntegerFlag(flag: string, value: string | undefined): number {
  const parsed = readNumberFlag(flag, value);
  if (!Number.isInteger(parsed)) {
    throw new LtxVideoExampleUsageError(`Expected ${flag} to be an integer.`);
  }
  return parsed;
}

function readNonNegativeIntegerFlag(flag: string, value: string | undefined): number {
  const parsed = readIntegerFlag(flag, value);
  if (parsed < 0) {
    throw new LtxVideoExampleUsageError(`Expected ${flag} to be non-negative.`);
  }
  return parsed;
}

function readDTypeFlag(value: string | undefined): CliOptions["dtype"] {
  const raw = readStringFlag("--dtype", value);
  if (raw === "float16" || raw === "float32" || raw === "bfloat16") {
    return raw;
  }
  throw new LtxVideoExampleUsageError("Expected --dtype to be float16, float32, or bfloat16.");
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
    case "--frames":
      draft.frames = readPositiveIntegerFlag(arg, argv[index + 1]);
      return index + 1;
    case "--frame-rate":
      draft.frameRate = readNumberFlag(arg, argv[index + 1]);
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
      throw new LtxVideoExampleUsageError(
        arg === undefined ? "Missing argument." : `Unknown argument: ${arg}`,
      );
  }
}

function validateCliOptions(draft: CliOptionsDraft): void {
  if (draft.prompt === undefined) {
    throw new LtxVideoExampleUsageError("Missing required --prompt <text>.");
  }
  if (!draft.outputPath.toLowerCase().endsWith(".bmp")) {
    throw new LtxVideoExampleUsageError("--output must end with .bmp.");
  }
  if (draft.height % 32 !== 0 || draft.width % 32 !== 0) {
    throw new LtxVideoExampleUsageError("--height and --width must be divisible by 32.");
  }
  if (!Number.isFinite(draft.frameRate) || draft.frameRate <= 0) {
    throw new LtxVideoExampleUsageError("--frame-rate must be a positive finite number.");
  }
  if (!Number.isFinite(draft.guidanceScale) || draft.guidanceScale < 0) {
    throw new LtxVideoExampleUsageError("--guidance-scale must be finite and non-negative.");
  }
  if (draft.maxSequenceLength > 128) {
    throw new LtxVideoExampleUsageError("--max-sequence-length must be no greater than 128.");
  }
}

export function parseCommand(argv: readonly string[]): CliCommand {
  if (argv.some((arg) => arg === "--help" || arg === "-h")) {
    return { kind: "help" };
  }

  const source = argv[0];
  if (source === undefined || source.trim() === "" || source.startsWith("--")) {
    throw new LtxVideoExampleUsageError("Missing snapshot source.");
  }

  const draft: CliOptionsDraft = {
    localFilesOnly: false,
    outputPath: ".tmp/ltx-video/preview.bmp",
    steps: 4,
    height: 128,
    width: 128,
    frames: 9,
    frameRate: 25,
    guidanceScale: 3,
    maxSequenceLength: 128,
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
    throw new LtxVideoExampleUsageError("Missing required --prompt <text>.");
  }

  const options: CliOptions = {
    source,
    localFilesOnly: draft.localFilesOnly,
    prompt,
    outputPath: draft.outputPath,
    steps: draft.steps,
    height: draft.height,
    width: draft.width,
    frames: draft.frames,
    frameRate: draft.frameRate,
    guidanceScale: draft.guidanceScale,
    maxSequenceLength: draft.maxSequenceLength,
    seed: draft.seed,
    dtype: draft.dtype,
    json: draft.json,
  };
  if (draft.negativePrompt !== undefined) {
    options.negativePrompt = draft.negativePrompt;
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
  if (draft.variant !== undefined) {
    options.variant = draft.variant;
  }
  return { kind: "run", options };
}

export function parseArgs(argv: readonly string[]): CliOptions {
  const command = parseCommand(argv);
  if (command.kind === "help") {
    throw new LtxVideoExampleUsageError("Help is not a video proof command.");
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

function videoShape(video: { shape: readonly number[] }): {
  width: number;
  height: number;
  frames: number;
  channels: number;
} {
  const [, frames, height, width, channels] = video.shape;
  if (
    video.shape.length !== 5 ||
    frames === undefined ||
    height === undefined ||
    width === undefined ||
    channels === undefined
  ) {
    throw new Error(`LTX-Video decoded output must be BFHWC, got [${video.shape.join(",")}].`);
  }
  return { width, height, frames, channels };
}

function usesClassifierFreeGuidance(scale: number): boolean {
  return scale > 1;
}

export async function runLtxVideoExample(
  cli: CliOptions,
  progress: (line: string) => void,
): Promise<LtxVideoExampleResult> {
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

  progress("Loading LTX-Video snapshot manifest...");
  const manifest = await loadDiffusionSnapshotManifest(snapshotPath);
  if (manifest.modelIndex.kind !== "ltx-video") {
    throw new Error(`LTX-Video proof does not support ${manifest.modelIndex.kind}.`);
  }

  progress("Loading LTX-Video component configs...");
  const configs = await loadLtxComponentConfigs(manifest);
  if (configs.pipelineKind !== "ltx-video") {
    throw new Error(`LTX-Video proof does not support ${configs.pipelineKind}.`);
  }

  progress("Loading LTX-Video scheduler...");
  const loadedScheduler = await loadDiffusionSchedulerFromSnapshot(snapshotPath);
  if (!(loadedScheduler.scheduler instanceof FlowMatchEulerScheduler)) {
    throw new Error("LTX-Video proof requires a FlowMatch Euler scheduler.");
  }
  const scheduler = loadedScheduler.scheduler;

  progress("Loading LTX-Video transformer...");
  using transformer = await loadLtxVideoTransformerFromSnapshot(manifest);

  progress("Loading LTX-Video VAE...");
  using vae = await loadLtxVideoAutoencoderFromSnapshot(manifest);

  progress("Loading LTX-Video prompt conditioner...");
  using conditioner = await loadLtxVideoPromptConditionerFromSnapshot(snapshotPath);
  const includeNegativePrompt = usesClassifierFreeGuidance(cli.guidanceScale);
  const promptOptions: LtxVideoPromptConditioningOptions = {
    prompt: cli.prompt,
    includeNegativePrompt,
    maxSequenceLength: cli.maxSequenceLength,
  };
  if (includeNegativePrompt && cli.negativePrompt !== undefined) {
    promptOptions.negativePrompt = cli.negativePrompt;
  }

  progress("Encoding prompt conditioning...");
  using conditioning = conditioner.encodePrompt(promptOptions);
  using rngKey = random.key(cli.seed);

  const latentShape = ltxVideoLatentShape({
    batchSize: conditioning.batchSize,
    height: cli.height,
    width: cli.width,
    numFrames: cli.frames,
    latentChannels: configs.transformer.inChannels,
    vaeSpatialCompressionRatio: vae.spatialCompressionRatio,
    vaeTemporalCompressionRatio: vae.temporalCompressionRatio,
  });
  progress(`Denoising ${cli.frames} frames at ${cli.width}x${cli.height}...`);
  using initialLatents = createLtxVideoInitialLatents({
    scheduler,
    batchSize: conditioning.batchSize,
    height: cli.height,
    width: cli.width,
    numFrames: cli.frames,
    latentChannels: configs.transformer.inChannels,
    vaeSpatialCompressionRatio: vae.spatialCompressionRatio,
    vaeTemporalCompressionRatio: vae.temporalCompressionRatio,
    patchSize: configs.transformer.patchSize,
    patchSizeT: configs.transformer.patchSizeT,
    dtype: cli.dtype,
    rngKey,
  });
  const denoiseOptions: LtxVideoDenoiseOptions = {
    denoiser: transformer,
    scheduler,
    initialLatents,
    latentFrames: latentShape[2],
    latentHeight: latentShape[3],
    latentWidth: latentShape[4],
    conditioning: conditioning.conditioning,
    numInferenceSteps: cli.steps,
    patchSize: configs.transformer.patchSize,
    patchSizeT: configs.transformer.patchSizeT,
    guidanceScale: cli.guidanceScale,
    vaeSpatialCompressionRatio: vae.spatialCompressionRatio,
    vaeTemporalCompressionRatio: vae.temporalCompressionRatio,
    frameRate: cli.frameRate,
    onStep: (event) => {
      progress(`Denoise step ${event.stepIndex + 1}/${cli.steps}: sigma ${event.sigma}`);
    },
  };
  using denoisedLatents = denoiseLtxVideoLatents(denoiseOptions);

  progress("Decoding LTX-Video latents...");
  using video = decodeLtxVideoLatents(
    vae,
    denoisedLatents,
    latentShape[2],
    latentShape[3],
    latentShape[4],
    configs.transformer.patchSize,
    configs.transformer.patchSizeT,
  );

  const decodedVideo = videoShape(video);
  const artifact = writeLtxVideoPreviewBmp(video, cli.outputPath);
  return {
    source: cli.source,
    snapshotPath,
    ...(snapshot.requestedRevision === undefined
      ? {}
      : { requestedRevision: snapshot.requestedRevision }),
    ...(snapshot.resolvedRevision === undefined
      ? {}
      : { resolvedRevision: snapshot.resolvedRevision }),
    pipeline: "ltx-video",
    prompt: cli.prompt,
    negativePrompt: includeNegativePrompt ? (cli.negativePrompt ?? "") : null,
    outputPath: artifact.path,
    imageSize: {
      width: artifact.width,
      height: artifact.height,
    },
    videoSize: decodedVideo,
    latentSize: {
      width: latentShape[4],
      height: latentShape[3],
      frames: latentShape[2],
      channels: latentShape[1],
    },
    outputBytes: artifact.bytes,
    artifact,
    steps: cli.steps,
    guidanceScale: cli.guidanceScale,
    maxSequenceLength: cli.maxSequenceLength,
    requestedFrames: cli.frames,
    decodedFrames: decodedVideo.frames,
    frameRate: cli.frameRate,
    seed: cli.seed,
    dtype: cli.dtype,
    promptTruncated: conditioning.promptTruncated,
    negativePromptTruncated: conditioning.negativePromptTruncated,
    elapsedMs: Math.round((performance.now() - startedAt) * 100) / 100,
  };
}

export function formatSuccess(report: LtxVideoExampleResult): string {
  return [
    "ltx_video_example:",
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
    `  preview_size: ${quoteScalar(`${report.imageSize.width}x${report.imageSize.height}`)}`,
    `  video_size: ${quoteScalar(
      `${report.videoSize.width}x${report.videoSize.height}x${report.videoSize.frames}`,
    )}`,
    `  latent_size: ${quoteScalar(
      `${report.latentSize.width}x${report.latentSize.height}x${report.latentSize.frames}x${report.latentSize.channels}`,
    )}`,
    `  output_bytes: ${report.outputBytes}`,
    `  artifact_sha256: ${quoteScalar(report.artifact.sha256)}`,
    `  artifact_checks: ${quoteScalar(report.artifact.status)}`,
    `  artifact_unique_byte_values: ${report.artifact.tensor.uniqueByteValues}`,
    `  artifact_channel_stddev_max: ${report.artifact.tensor.maxChannelStddev}`,
    `  steps: ${report.steps}`,
    `  guidance_scale: ${report.guidanceScale}`,
    `  max_sequence_length: ${report.maxSequenceLength}`,
    `  requested_frames: ${report.requestedFrames}`,
    `  decoded_frames: ${report.decodedFrames}`,
    `  frame_rate: ${report.frameRate}`,
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
    '  "Run `bun run examples/ltx-video/index.ts --help` for options"',
  ].join("\n");
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export async function runLtxVideoExampleCommand(
  argv: readonly string[],
  runtime: LtxVideoExampleRuntime = {},
): Promise<number> {
  const stdout = runtime.stdout ?? console.log;
  const stderr = runtime.stderr ?? console.error;
  let command: CliCommand;
  try {
    command = parseCommand(argv);
  } catch (error) {
    stdout(formatError(errorMessage(error), "usage"));
    return error instanceof LtxVideoExampleUsageError ? 2 : 1;
  }

  if (command.kind === "help") {
    stdout(formatUsage());
    return 0;
  }

  const acquireLock = runtime.acquireLock ?? (() => acquireRuntimeCommandLock("example:ltx-video"));
  const runExample = runtime.runExample ?? runLtxVideoExample;
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
  const exitCode = await runLtxVideoExampleCommand(Bun.argv.slice(2));
  process.exit(exitCode);
}
