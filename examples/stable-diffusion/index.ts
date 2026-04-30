#!/usr/bin/env bun

import type { DType } from "@mlxts/core";
import { random } from "@mlxts/core";
import {
  loadStableDiffusionPipelineFromSnapshot,
  type StableDiffusionImageGenerationOptions,
} from "@mlxts/diffusion";

import { acquireRuntimeCommandLock } from "../../scripts/runtime-command-lock";
import {
  loadStableDiffusionPromptConditionerFromSnapshot,
  type StableDiffusionPromptConditioningOptions,
} from "./conditioning";
import { writeStableDiffusionBmp } from "./image-output";

type CliOptions = {
  snapshotPath: string;
  prompt: string;
  prompt2?: string;
  negativePrompt?: string;
  negativePrompt2?: string;
  outputPath: string;
  steps: number;
  height?: number;
  width?: number;
  guidanceScale: number;
  seed: number;
  dtype: Extract<DType, "float16" | "float32" | "bfloat16">;
  json: boolean;
};

type CliCommand = { kind: "help" } | { kind: "run"; options: CliOptions };

type CliOptionsDraft = {
  prompt?: string;
  prompt2?: string;
  negativePrompt?: string;
  negativePrompt2?: string;
  outputPath: string;
  steps: number;
  height?: number;
  width?: number;
  guidanceScale: number;
  seed: number;
  dtype: CliOptions["dtype"];
  json: boolean;
};

type StableDiffusionExampleResult = {
  snapshotPath: string;
  pipeline: "stable-diffusion" | "stable-diffusion-xl";
  prompt: string;
  negativePrompt: string;
  outputPath: string;
  imageSize: { width: number; height: number };
  outputBytes: number;
  steps: number;
  guidanceScale: number;
  seed: number;
  dtype: string;
  promptTruncated: boolean;
  negativePromptTruncated: boolean;
  elapsedMs: number;
};

type RuntimeLock = {
  [Symbol.dispose](): void;
};

type StableDiffusionExampleRuntime = {
  stdout?: (text: string) => void;
  stderr?: (text: string) => void;
  acquireLock?: () => RuntimeLock;
  runExample?: (
    cli: CliOptions,
    progress: (line: string) => void,
  ) => Promise<StableDiffusionExampleResult>;
};

class StableDiffusionExampleUsageError extends Error {}

function quoteScalar(value: string | number | boolean | null): string {
  return typeof value === "string" ? JSON.stringify(value) : String(value);
}

export function formatUsage(): string {
  return [
    "description: Run one local Stable Diffusion text-to-image proof and write a BMP artifact",
    "usage[2]:",
    '  bun run examples/stable-diffusion/index.ts <snapshot-path> --prompt "a red apple"',
    '  bun run examples/stable-diffusion/index.ts /models/sdxl --prompt "a quiet library" --output .tmp/sd.bmp --steps 20',
    "arguments[1]{name,description}:",
    '  "snapshot-path","Local Diffusers Stable Diffusion or SDXL snapshot directory"',
    "options[13]{flag,description}:",
    '  "--prompt <text>","Required positive prompt"',
    '  "--prompt-2 <text>","Optional SDXL second-encoder prompt"',
    '  "--negative-prompt <text>","Negative prompt; default empty when CFG is active"',
    '  "--negative-prompt-2 <text>","Optional SDXL second-encoder negative prompt"',
    '  "--output <path>","BMP output path; default .tmp/stable-diffusion/sample.bmp"',
    '  "--steps <n>","Inference steps; default 20"',
    '  "--height <n>","Output height; default 512 for SD, 1024 for SDXL"',
    '  "--width <n>","Output width; default 512 for SD, 1024 for SDXL"',
    '  "--guidance-scale <n>","Classifier-free guidance scale; default 7.5"',
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
    throw new StableDiffusionExampleUsageError(`Missing value for ${flag}.`);
  }
  return value;
}

function readNumberFlag(flag: string, value: string | undefined): number {
  const raw = readStringFlag(flag, value);
  const parsed = Number(raw);
  if (!Number.isFinite(parsed)) {
    throw new StableDiffusionExampleUsageError(`Expected ${flag} to be a finite number.`);
  }
  return parsed;
}

function readPositiveIntegerFlag(flag: string, value: string | undefined): number {
  const parsed = readNumberFlag(flag, value);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new StableDiffusionExampleUsageError(`Expected ${flag} to be a positive integer.`);
  }
  return parsed;
}

function readIntegerFlag(flag: string, value: string | undefined): number {
  const parsed = readNumberFlag(flag, value);
  if (!Number.isInteger(parsed)) {
    throw new StableDiffusionExampleUsageError(`Expected ${flag} to be an integer.`);
  }
  return parsed;
}

function readNonNegativeIntegerFlag(flag: string, value: string | undefined): number {
  const parsed = readIntegerFlag(flag, value);
  if (parsed < 0) {
    throw new StableDiffusionExampleUsageError(`Expected ${flag} to be non-negative.`);
  }
  return parsed;
}

function readDTypeFlag(value: string | undefined): CliOptions["dtype"] {
  const raw = readStringFlag("--dtype", value);
  if (raw === "float16" || raw === "float32" || raw === "bfloat16") {
    return raw;
  }
  throw new StableDiffusionExampleUsageError(
    `Expected --dtype to be float16, float32, or bfloat16.`,
  );
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
    case "--negative-prompt":
      draft.negativePrompt = readStringFlag(arg, argv[index + 1]);
      return index + 1;
    case "--negative-prompt-2":
      draft.negativePrompt2 = readStringFlag(arg, argv[index + 1]);
      return index + 1;
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
      throw new StableDiffusionExampleUsageError(
        arg === undefined ? "Missing argument." : `Unknown argument: ${arg}`,
      );
  }
}

export function parseCommand(argv: readonly string[]): CliCommand {
  if (argv.some((arg) => arg === "--help" || arg === "-h")) {
    return { kind: "help" };
  }

  const snapshotPath = argv[0];
  if (snapshotPath === undefined || snapshotPath.trim() === "" || snapshotPath.startsWith("--")) {
    throw new StableDiffusionExampleUsageError("Missing snapshot path.");
  }

  const draft: CliOptionsDraft = {
    outputPath: ".tmp/stable-diffusion/sample.bmp",
    steps: 20,
    guidanceScale: 7.5,
    seed: 0,
    dtype: "float16",
    json: false,
  };

  for (let index = 1; index < argv.length; index += 1) {
    index = applyFlag(argv, index, draft);
  }

  if (draft.prompt === undefined) {
    throw new StableDiffusionExampleUsageError("Missing required --prompt <text>.");
  }
  if (!Number.isFinite(draft.guidanceScale) || draft.guidanceScale < 0) {
    throw new StableDiffusionExampleUsageError("--guidance-scale must be finite and non-negative.");
  }
  if (!draft.outputPath.toLowerCase().endsWith(".bmp")) {
    throw new StableDiffusionExampleUsageError("--output must end with .bmp.");
  }

  const options: CliOptions = {
    snapshotPath,
    prompt: draft.prompt,
    outputPath: draft.outputPath,
    steps: draft.steps,
    guidanceScale: draft.guidanceScale,
    seed: draft.seed,
    dtype: draft.dtype,
    json: draft.json,
  };
  if (draft.prompt2 !== undefined) {
    options.prompt2 = draft.prompt2;
  }
  if (draft.negativePrompt !== undefined) {
    options.negativePrompt = draft.negativePrompt;
  }
  if (draft.negativePrompt2 !== undefined) {
    options.negativePrompt2 = draft.negativePrompt2;
  }
  if (draft.height !== undefined) {
    options.height = draft.height;
  }
  if (draft.width !== undefined) {
    options.width = draft.width;
  }
  return { kind: "run", options };
}

export function parseArgs(argv: readonly string[]): CliOptions {
  const command = parseCommand(argv);
  if (command.kind === "help") {
    throw new StableDiffusionExampleUsageError("Help is not an image proof command.");
  }
  return command.options;
}

function latentSampleSize(value: number | readonly [number, number] | undefined): {
  height: number;
  width: number;
} | null {
  if (value === undefined) {
    return null;
  }
  if (typeof value === "number") {
    return { height: value, width: value };
  }
  const [height, width] = value;
  return { height, width };
}

function defaultImageSize(
  pipeline: "stable-diffusion" | "stable-diffusion-xl",
  sampleSize: number | readonly [number, number] | undefined,
  vaeScaleFactor: number,
): { height: number; width: number } {
  const latentSize = latentSampleSize(sampleSize);
  if (latentSize !== null) {
    return {
      height: latentSize.height * vaeScaleFactor,
      width: latentSize.width * vaeScaleFactor,
    };
  }
  const fallback = pipeline === "stable-diffusion-xl" ? 1024 : 512;
  return { height: fallback, width: fallback };
}

function printRunIntro(cli: CliOptions, writeLine: (line: string) => void): void {
  writeLine(`Snapshot: ${cli.snapshotPath}`);
  writeLine(`Prompt: ${cli.prompt}`);
  writeLine(`Output: ${cli.outputPath}`);
  writeLine(`Steps: ${cli.steps}`);
  writeLine(`Guidance scale: ${cli.guidanceScale}`);
  writeLine(`Seed: ${cli.seed}`);
  writeLine(`Latent dtype: ${cli.dtype}`);
  writeLine("");
}

export async function runStableDiffusionExample(
  cli: CliOptions,
  progress: (line: string) => void,
): Promise<StableDiffusionExampleResult> {
  const startedAt = performance.now();
  printRunIntro(cli, progress);

  progress("Loading Stable Diffusion pipeline...");
  using pipeline = await loadStableDiffusionPipelineFromSnapshot(cli.snapshotPath);
  if (
    pipeline.manifest.modelIndex.kind !== "stable-diffusion" &&
    pipeline.manifest.modelIndex.kind !== "stable-diffusion-xl"
  ) {
    throw new Error(
      `Stable Diffusion proof does not support ${pipeline.manifest.modelIndex.kind}.`,
    );
  }
  const pipelineKind = pipeline.manifest.modelIndex.kind;
  const defaultSize = defaultImageSize(
    pipelineKind,
    pipeline.configs.unet.sampleSize,
    pipeline.vae.vaeScaleFactor,
  );
  const height = cli.height ?? defaultSize.height;
  const width = cli.width ?? defaultSize.width;

  progress("Loading prompt conditioner...");
  using conditioner = await loadStableDiffusionPromptConditionerFromSnapshot(cli.snapshotPath);
  const promptOptions: StableDiffusionPromptConditioningOptions = {
    prompt: cli.prompt,
    guidanceScale: cli.guidanceScale,
    targetSize: [height, width],
    originalSize: [height, width],
  };
  if (cli.prompt2 !== undefined) {
    promptOptions.prompt2 = cli.prompt2;
  }
  if (cli.negativePrompt !== undefined) {
    promptOptions.negativePrompt = cli.negativePrompt;
  }
  if (cli.negativePrompt2 !== undefined) {
    promptOptions.negativePrompt2 = cli.negativePrompt2;
  }

  progress("Encoding prompt conditioning...");
  using conditioning = conditioner.encodePrompt(promptOptions);
  using rngKey = random.key(cli.seed);

  progress(`Denoising ${width}x${height} image...`);
  const generationOptions: Omit<
    StableDiffusionImageGenerationOptions,
    "unet" | "vae" | "scheduler"
  > = {
    batchSize: conditioning.batchSize,
    height,
    width,
    conditioning: conditioning.conditioning,
    guidanceScale: cli.guidanceScale,
    numInferenceSteps: cli.steps,
    dtype: cli.dtype,
    rngKey,
    onStep: (event) => {
      progress(`Denoise step ${event.stepIndex + 1}/${cli.steps}: timestep ${event.timestep}`);
    },
  };
  if (conditioning.negativeConditioning !== undefined) {
    generationOptions.negativeConditioning = conditioning.negativeConditioning;
  }
  using image = pipeline.generateImage(generationOptions);

  const artifact = writeStableDiffusionBmp(image, cli.outputPath);
  return {
    snapshotPath: cli.snapshotPath,
    pipeline: pipelineKind,
    prompt: cli.prompt,
    negativePrompt: cli.negativePrompt ?? "",
    outputPath: artifact.path,
    imageSize: {
      width: artifact.width,
      height: artifact.height,
    },
    outputBytes: artifact.bytes,
    steps: cli.steps,
    guidanceScale: cli.guidanceScale,
    seed: cli.seed,
    dtype: cli.dtype,
    promptTruncated: conditioning.promptTruncated,
    negativePromptTruncated: conditioning.negativePromptTruncated,
    elapsedMs: Math.round((performance.now() - startedAt) * 100) / 100,
  };
}

export function formatSuccess(report: StableDiffusionExampleResult): string {
  return [
    "stable_diffusion_example:",
    "  status: passed",
    `  snapshot_path: ${quoteScalar(report.snapshotPath)}`,
    `  pipeline: ${quoteScalar(report.pipeline)}`,
    `  output_path: ${quoteScalar(report.outputPath)}`,
    `  image_size: ${quoteScalar(`${report.imageSize.width}x${report.imageSize.height}`)}`,
    `  output_bytes: ${report.outputBytes}`,
    `  steps: ${report.steps}`,
    `  guidance_scale: ${report.guidanceScale}`,
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
    '  "Run `bun run examples/stable-diffusion/index.ts --help` for options"',
  ].join("\n");
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export async function runStableDiffusionExampleCommand(
  argv: readonly string[],
  runtime: StableDiffusionExampleRuntime = {},
): Promise<number> {
  const stdout = runtime.stdout ?? console.log;
  const stderr = runtime.stderr ?? console.error;
  let command: CliCommand;
  try {
    command = parseCommand(argv);
  } catch (error) {
    stdout(formatError(errorMessage(error), "usage"));
    return error instanceof StableDiffusionExampleUsageError ? 2 : 1;
  }

  if (command.kind === "help") {
    stdout(formatUsage());
    return 0;
  }

  const acquireLock =
    runtime.acquireLock ?? (() => acquireRuntimeCommandLock("example:stable-diffusion"));
  const runExample = runtime.runExample ?? runStableDiffusionExample;
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
  const exitCode = await runStableDiffusionExampleCommand(Bun.argv.slice(2));
  process.exit(exitCode);
}
