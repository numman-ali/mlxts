#!/usr/bin/env bun

import type { GenerationOptions, PretrainedLoadProgressEvent } from "@mlxts/transformers";
import {
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

import { decodeResizedImage, readImageSize } from "./image-io";

type CliOptions = {
  source: string;
  imagePath: string;
  prompt: string;
  maxTokens: number;
  systemPrompt?: string;
  overrides: Pick<GenerationOptions, "temperature" | "topK" | "topP">;
};

function formatBytes(bytes: number): string {
  if (bytes >= 1e9) {
    return `${(bytes / 1e9).toFixed(1)} GB`;
  }
  if (bytes >= 1e6) {
    return `${(bytes / 1e6).toFixed(1)} MB`;
  }
  if (bytes >= 1e3) {
    return `${(bytes / 1e3).toFixed(1)} KB`;
  }
  return `${bytes} B`;
}

function logResolveEvent(event: Extract<PretrainedLoadProgressEvent, { stage: "resolve" }>): void {
  if (event.status === "start") {
    console.log(`[resolve] resolving ${event.source}`);
    return;
  }

  const revision =
    event.resolvedRevision === undefined ? "" : ` @ ${event.resolvedRevision.slice(0, 12)}`;
  const repo = event.repoId === undefined ? event.directory : `${event.repoId}${revision}`;
  console.log(
    `[resolve] ${repo} -> ${event.directory} (${event.fileCount} files, ${formatBytes(event.totalBytes)})`,
  );
}

function logDownloadEvent(
  event: Extract<PretrainedLoadProgressEvent, { stage: "download" }>,
): void {
  console.log(
    `[download] ${event.index}/${event.totalFiles} ${event.status} ${event.relativePath} (${formatBytes(event.size)}) ${formatBytes(event.completedBytes)} / ${formatBytes(event.totalBytes)}`,
  );
}

function logModelEvent(event: Extract<PretrainedLoadProgressEvent, { stage: "model" }>): void {
  console.log(
    event.status === "weights-start"
      ? `[model] loading ${event.shardCount} safetensor shard(s)`
      : `[model] finished loading ${event.shardCount} safetensor shard(s)`,
  );
}

function logTokenizerEvent(
  event: Extract<PretrainedLoadProgressEvent, { stage: "tokenizer" }>,
): void {
  console.log(
    event.status === "start" ? `[tokenizer] loading from ${event.directory}` : "[tokenizer] ready",
  );
}

function createProgressReporter(): (event: PretrainedLoadProgressEvent) => void {
  return (event) => {
    switch (event.stage) {
      case "resolve":
        logResolveEvent(event);
        return;
      case "download":
        logDownloadEvent(event);
        return;
      case "model":
        logModelEvent(event);
        return;
      case "tokenizer":
        logTokenizerEvent(event);
        return;
    }
  };
}

function usage(): never {
  console.error(
    "Usage: bun run examples/qwen3_5-image/index.ts <model-path-or-repo-id> --image <path> [--prompt <text>] [--temperature <n>] [--top-k <n>] [--top-p <n>] [--max-tokens <n>] [--system-prompt <text>] [--greedy]",
  );
  process.exit(1);
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

function readStringFlag(flag: string, value: string | undefined): string {
  if (value === undefined || value.trim() === "") {
    throw new Error(`Missing value for ${flag}.`);
  }
  return value;
}

function parseArgs(argv: readonly string[]): CliOptions {
  const source = argv[0];
  if (source === undefined || source.trim() === "") {
    usage();
  }

  const overrides: Pick<GenerationOptions, "temperature" | "topK" | "topP"> = {};
  let imagePath: string | undefined;
  let prompt = "Describe this image.";
  let maxTokens = 128;
  let systemPrompt: string | undefined;

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
        maxTokens = readNumberFlag(arg, argv[index + 1]);
        index += 1;
        break;
      case "--system-prompt":
        systemPrompt = readStringFlag(arg, argv[index + 1]);
        index += 1;
        break;
      case "--greedy":
        overrides.temperature = 0;
        break;
      default:
        throw new Error(`Unknown argument: ${arg}`);
    }
  }

  if (imagePath === undefined) {
    throw new Error("Missing required --image <path>.");
  }

  return { source, imagePath, prompt, maxTokens, systemPrompt, overrides };
}

function printRunIntro(cli: CliOptions): void {
  console.log(`Source: ${cli.source}`);
  console.log(`Image: ${cli.imagePath}`);
  console.log(`Prompt: ${cli.prompt}`);
  console.log(`Max tokens: ${cli.maxTokens}`);
  if (cli.systemPrompt !== undefined) {
    console.log("System prompt: enabled");
  }
  console.log("");
}

function buildPromptText(userPrompt: string): string {
  return `<|vision_start|><|image_pad|><|vision_end|>\n${userPrompt}`;
}

async function main(): Promise<void> {
  const cli = parseArgs(process.argv.slice(2));
  printRunIntro(cli);

  const reportProgress = createProgressReporter();
  const localSource = await resolvePretrainedSource(cli.source, { onProgress: reportProgress });
  using model = await loadQwen3_5ForConditionalGeneration(localSource, {
    onProgress: reportProgress,
  });
  const tokenizer = await loadPretrainedTokenizer(localSource, { onProgress: reportProgress });
  const interactionProfile = await loadInteractionProfile(localSource);
  const preprocessor = await loadQwen3_5VisionPreprocessor(localSource);

  const originalSize = readImageSize(cli.imagePath);
  const resizedSize = smartResizeQwen3_5Image(
    originalSize.height,
    originalSize.width,
    preprocessor,
  );
  const image = await decodeResizedImage(cli.imagePath, resizedSize);
  console.log(
    `Image resize: ${originalSize.width}x${originalSize.height} -> ${image.width}x${image.height}`,
  );

  const userContent = buildPromptText(cli.prompt);
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
          { addGenerationPrompt: true },
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
      console.log("");
      console.log("Response:");
      console.log(text === "" ? "[empty response]" : text);
      console.log("");
      console.log(`Finish reason: ${result.finishReason}`);
      console.log(`Generated tokens: ${result.tokenIds.length}`);
    } finally {
      preparedPrompt.inputEmbeddings?.free();
      preparedPrompt.positionIds?.free();
    }
  } finally {
    preparedImages.pixelValues.free();
    preparedImages.imageGridThw.free();
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
