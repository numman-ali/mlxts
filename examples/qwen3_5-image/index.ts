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

function usageText(): string {
  return "Usage: bun run examples/qwen3_5-image/index.ts <model-path-or-repo-id> --image <path> [--prompt <text>] [--temperature <n>] [--top-k <n>] [--top-p <n>] [--max-tokens <n>] [--system-prompt <text>] [--greedy] [--enable-thinking|--disable-thinking|--template-default-thinking] [--allow-download] [--json]";
}

function readNumberFlag(flag: string, value: string | undefined): number {
  if (value === undefined) {
    throw new Error(`Missing value for ${flag}.`);
  }
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    throw new Error(`Expected ${flag} to be a finite number, got "${value}".`);
  }
  return parsed;
}

function readPositiveIntegerFlag(flag: string, value: string | undefined): number {
  const parsed = readNumberFlag(flag, value);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`Expected ${flag} to be a positive integer, got "${value}".`);
  }
  return parsed;
}

function readStringFlag(flag: string, value: string | undefined): string {
  if (value === undefined || value.trim() === "") {
    throw new Error(`Missing value for ${flag}.`);
  }
  return value;
}

export function parseArgs(argv: readonly string[]): CliOptions {
  const source = argv[0];
  if (source === undefined || source.trim() === "") {
    throw new Error(`Missing model source.\n${usageText()}`);
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
        throw new Error(`Unknown argument: ${arg}`);
    }
  }

  if (imagePath === undefined) {
    throw new Error("Missing required --image <path>.");
  }

  return {
    source,
    imagePath,
    prompt,
    maxTokens,
    systemPrompt,
    localFilesOnly,
    thinking,
    json,
    overrides,
  };
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

async function main(): Promise<void> {
  const cli = parseArgs(process.argv.slice(2));
  using _runtimeLock = acquireRuntimeCommandLock("example:qwen3_5-image");
  const startedAt = performance.now();
  const writeStatusLine = cli.json ? (line: string) => console.error(line) : console.log;
  printRunIntro(cli, writeStatusLine);

  const reportProgress = createProgressReporter(writeStatusLine);
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
  writeStatusLine(
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
      if (cli.json) {
        console.log(JSON.stringify(report, null, 2));
      } else {
        console.log("");
        console.log("Response:");
        console.log(text === "" ? "[empty response]" : text);
        console.log("");
        console.log(`Finish reason: ${result.finishReason}`);
        console.log(`Generated tokens: ${result.tokenIds.length}`);
      }
    } finally {
      preparedPrompt.inputEmbeddings?.free();
      preparedPrompt.positionIds?.free();
    }
  } finally {
    preparedImages.pixelValues.free();
    preparedImages.imageGridThw.free();
  }
}

if (import.meta.main) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exit(1);
  });
}
