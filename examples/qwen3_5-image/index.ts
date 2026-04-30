#!/usr/bin/env bun

import type { GenerationOptions } from "@mlxts/transformers";
import {
  createProgressReporter,
  generatePreparedTokens,
  loadInteractionProfile,
  loadPretrainedTokenizer,
  loadQwen3_5ForConditionalGeneration,
  loadQwen3_5VisionPreprocessor,
  prepareQwen3_5ImageBatch,
  prepareQwen3_5ImagePrompt,
  resolvePretrainedSource,
  smartResizeQwen3_5Image,
} from "@mlxts/transformers";

import { acquireRuntimeCommandLock } from "../../scripts/runtime-command-lock";
import { decodeResizedImage, readImageSize } from "./image-io";

type ThinkingMode = "disabled" | "enabled" | "template-default";

type CliOptions = {
  source: string;
  imagePath: string;
  prompt: string;
  maxTokens: number;
  systemPrompt?: string;
  localFilesOnly: boolean;
  thinking: ThinkingMode;
  json: boolean;
  overrides: Pick<GenerationOptions, "temperature" | "topK" | "topP">;
};

type CliCommand = { kind: "help" } | { kind: "run"; options: CliOptions };

type QwenImageExampleResult = {
  source: string;
  sourceMode: "cached-local-only" | "downloads-allowed";
  imagePath: string;
  prompt: string;
  thinking: ThinkingMode;
  originalSize: { width: number; height: number };
  resizedSize: { width: number; height: number };
  finishReason: string;
  generatedTokens: number;
  text: string;
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
    "description: Run one-shot Qwen 3.5/3.6 local image-conditioned generation",
    "usage[2]:",
    "  bun run examples/qwen3_5-image/index.ts <model-path-or-repo-id> --image <path>",
    "  bun run examples/qwen3_5-image/index.ts mlx-community/Qwen3.6-27B-4bit --image ./photo.jpg --greedy",
    "arguments[1]{name,description}:",
    '  "model-path-or-repo-id","Local snapshot path or Hugging Face repo id"',
    "options[14]{flag,description}:",
    '  "--image <path>","Required local image path"',
    '  "--prompt <text>","User prompt; default Describe this image."',
    '  "--system-prompt <text>","Optional system message for chat checkpoints"',
    '  "--max-tokens <n>","Maximum generated tokens; default 128"',
    '  "--temperature <n>","Sampling temperature"',
    '  "--top-k <n>","Top-k sampling limit"',
    '  "--top-p <n>","Top-p sampling threshold"',
    '  "--greedy","Set temperature to 0"',
    '  "--enable-thinking","Force Qwen thinking on"',
    '  "--disable-thinking","Force Qwen thinking off; default"',
    '  "--template-default-thinking","Use checkpoint chat-template default thinking behavior"',
    '  "--allow-download","Allow Hub downloads; default cached/local only"',
    '  "--json","Emit the final result as JSON"',
    '  "--help","Show this help"',
    "exit_codes[3]{code,meaning}:",
    '  0,"generation passed or help"',
    '  1,"runtime or generation failure"',
    '  2,"usage error"',
  ].join("\n");
}

function readNumberFlag(flag: string, value: string | undefined): number {
  if (value === undefined) {
    throw new QwenImageExampleUsageError(`Missing value for ${flag}.`);
  }
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    throw new QwenImageExampleUsageError(`Expected ${flag} to be a finite number, got "${value}".`);
  }
  return parsed;
}

function readPositiveIntegerFlag(flag: string, value: string | undefined): number {
  const parsed = readNumberFlag(flag, value);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new QwenImageExampleUsageError(
      `Expected ${flag} to be a positive integer, got "${value}".`,
    );
  }
  return parsed;
}

function readStringFlag(flag: string, value: string | undefined): string {
  if (value === undefined || value.trim() === "" || value.startsWith("--")) {
    throw new QwenImageExampleUsageError(`Missing value for ${flag}.`);
  }
  return value;
}

export function parseCommand(argv: readonly string[]): CliCommand {
  if (argv.some((arg) => arg === "--help" || arg === "-h")) {
    return { kind: "help" };
  }

  const source = argv[0];
  if (source === undefined || source.trim() === "") {
    throw new QwenImageExampleUsageError("Missing model source.");
  }

  const overrides: Pick<GenerationOptions, "temperature" | "topK" | "topP"> = {};
  let imagePath: string | undefined;
  let prompt = "Describe this image.";
  let maxTokens = 128;
  let systemPrompt: string | undefined;
  let localFilesOnly = true;
  let thinking: ThinkingMode = "disabled";
  let json = false;

  for (let index = 1; index < argv.length; index += 1) {
    const arg = argv[index];
    switch (arg) {
      case "--image":
        imagePath = readStringFlag(arg, argv[index + 1]);
        index += 1;
        break;
      case "--prompt":
        prompt = readStringFlag(arg, argv[index + 1]);
        index += 1;
        break;
      case "--temperature":
        overrides.temperature = readNumberFlag(arg, argv[index + 1]);
        index += 1;
        break;
      case "--top-k":
        overrides.topK = readNumberFlag(arg, argv[index + 1]);
        index += 1;
        break;
      case "--top-p":
        overrides.topP = readNumberFlag(arg, argv[index + 1]);
        index += 1;
        break;
      case "--max-tokens":
        maxTokens = readPositiveIntegerFlag(arg, argv[index + 1]);
        index += 1;
        break;
      case "--system-prompt":
        systemPrompt = readStringFlag(arg, argv[index + 1]);
        index += 1;
        break;
      case "--greedy":
        overrides.temperature = 0;
        break;
      case "--allow-download":
        localFilesOnly = false;
        break;
      case "--enable-thinking":
        thinking = "enabled";
        break;
      case "--disable-thinking":
        thinking = "disabled";
        break;
      case "--template-default-thinking":
        thinking = "template-default";
        break;
      case "--json":
        json = true;
        break;
      default:
        throw new QwenImageExampleUsageError(
          arg === undefined ? "Missing argument." : `Unknown argument: ${arg}`,
        );
    }
  }

  if (imagePath === undefined) {
    throw new QwenImageExampleUsageError("Missing required --image <path>.");
  }

  return {
    kind: "run",
    options: {
      source,
      imagePath,
      prompt,
      maxTokens,
      systemPrompt,
      localFilesOnly,
      thinking,
      json,
      overrides,
    },
  };
}

export function parseArgs(argv: readonly string[]): CliOptions {
  const command = parseCommand(argv);
  if (command.kind === "help") {
    throw new QwenImageExampleUsageError("Help is not a generation command.");
  }
  return command.options;
}

function printRunIntro(cli: CliOptions, writeLine: (line: string) => void): void {
  writeLine(`Source: ${cli.source}`);
  writeLine(`Source mode: ${cli.localFilesOnly ? "cached/local only" : "downloads allowed"}`);
  writeLine(`Image: ${cli.imagePath}`);
  writeLine(`Prompt: ${cli.prompt}`);
  writeLine(`Max tokens: ${cli.maxTokens}`);
  writeLine(`Thinking: ${cli.thinking}`);
  if (cli.systemPrompt !== undefined) {
    writeLine("System prompt: enabled");
  }
  writeLine("");
}

function buildPromptText(userPrompt: string): string {
  return `<|vision_start|><|image_pad|><|vision_end|>\n${userPrompt}`;
}

function enableThinkingOption(mode: ThinkingMode): boolean | undefined {
  if (mode === "template-default") {
    return undefined;
  }
  return mode === "enabled";
}

export async function runQwenImageExample(
  cli: CliOptions,
  progress: (line: string) => void,
): Promise<QwenImageExampleResult> {
  const startedAt = performance.now();
  printRunIntro(cli, progress);

  const reportProgress = createProgressReporter(progress);
  const loadOptions = {
    localFilesOnly: cli.localFilesOnly,
    onProgress: reportProgress,
  };
  const localSource = await resolvePretrainedSource(cli.source, {
    ...loadOptions,
  });
  using model = await loadQwen3_5ForConditionalGeneration(localSource, loadOptions);
  const tokenizer = await loadPretrainedTokenizer(localSource, loadOptions);
  const interactionProfile = await loadInteractionProfile(localSource, loadOptions);
  const preprocessor = await loadQwen3_5VisionPreprocessor(localSource, loadOptions);

  const originalSize = readImageSize(cli.imagePath);
  const resizedSize = smartResizeQwen3_5Image(
    originalSize.height,
    originalSize.width,
    preprocessor,
  );
  const image = await decodeResizedImage(cli.imagePath, resizedSize);
  progress(
    `Image resize: ${originalSize.width}x${originalSize.height} -> ${image.width}x${image.height}`,
  );

  const userContent = buildPromptText(cli.prompt);
  const enableThinking = enableThinkingOption(cli.thinking);
  const promptCompilation =
    interactionProfile.kind === "chat"
      ? interactionProfile.compileMessages(
          tokenizer,
          [
            ...(cli.systemPrompt === undefined
              ? []
              : [{ role: "system", content: cli.systemPrompt } as const]),
            { role: "user", content: userContent } as const,
          ],
          {
            addGenerationPrompt: true,
            ...(enableThinking === undefined ? {} : { enableThinking }),
          },
        )
      : interactionProfile.compileTextPrompt(tokenizer, userContent);

  const preparedImages = prepareQwen3_5ImageBatch(image, preprocessor);
  try {
    const preparedPrompt = prepareQwen3_5ImagePrompt(
      model,
      promptCompilation.tokenIds,
      preparedImages.pixelValues,
      preparedImages.imageGridThw,
    );
    try {
      const result = generatePreparedTokens(model, preparedPrompt, {
        maxTokens: cli.maxTokens,
        ...cli.overrides,
      });
      const text = tokenizer.decode(result.tokenIds, { skipSpecialTokens: true });
      const report: QwenImageExampleResult = {
        source: cli.source,
        sourceMode: cli.localFilesOnly ? "cached-local-only" : "downloads-allowed",
        imagePath: cli.imagePath,
        prompt: cli.prompt,
        thinking: cli.thinking,
        originalSize,
        resizedSize: {
          width: image.width,
          height: image.height,
        },
        finishReason: result.finishReason,
        generatedTokens: result.tokenIds.length,
        text,
        elapsedMs: Math.round((performance.now() - startedAt) * 100) / 100,
      };
      return report;
    } finally {
      preparedPrompt.inputEmbeddings?.free();
      preparedPrompt.positionIds?.free();
    }
  } finally {
    preparedImages.pixelValues.free();
    preparedImages.imageGridThw.free();
  }
}

function formatBlockField(name: string, value: string): string[] {
  const lines = value.split(/\r?\n/);
  return [`${name}: |`, ...lines.map((line) => `  ${line}`)];
}

export function formatSuccess(report: QwenImageExampleResult): string {
  return [
    "qwen_image_example:",
    "  status: passed",
    `  source: ${quoteScalar(report.source)}`,
    `  source_mode: ${quoteScalar(report.sourceMode)}`,
    `  image_path: ${quoteScalar(report.imagePath)}`,
    `  prompt: ${quoteScalar(report.prompt)}`,
    `  thinking: ${quoteScalar(report.thinking)}`,
    `  original_size: ${quoteScalar(`${report.originalSize.width}x${report.originalSize.height}`)}`,
    `  resized_size: ${quoteScalar(`${report.resizedSize.width}x${report.resizedSize.height}`)}`,
    `  finish_reason: ${quoteScalar(report.finishReason)}`,
    `  generated_tokens: ${report.generatedTokens}`,
    `  elapsed_ms: ${report.elapsedMs}`,
    ...formatBlockField("response", report.text === "" ? "[empty response]" : report.text),
  ].join("\n");
}

function formatError(message: string, code: "usage" | "runtime"): string {
  return [
    "error:",
    `  code: ${quoteScalar(code)}`,
    `  message: ${quoteScalar(message)}`,
    "help[1]:",
    '  "Run `bun run examples/qwen3_5-image/index.ts --help` for options"',
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
    runtime.acquireLock ?? (() => acquireRuntimeCommandLock("example:qwen3_5-image"));
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
