#!/usr/bin/env bun

import type { GenerationOptions } from "@mlxts/transformers";
import {
  type ChatMessage,
  generateTextStream,
  loadCausalLM,
  loadInteractionProfile,
  loadPretrainedTokenizer,
  makePromptCache,
  type PretrainedLoadProgressEvent,
  resolvePretrainedSource,
} from "@mlxts/transformers";

type CliOptions = {
  source: string;
  maxTokens: number;
  systemPrompt?: string;
  overrides: Pick<GenerationOptions, "temperature" | "topK" | "topP">;
};

type PromptAction =
  | { kind: "quit" | "reset" | "help" | "skip" }
  | { kind: "prompt"; promptText: string };

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
    "Usage: bun run examples/chat/index.ts <model-path-or-repo-id> [--temperature <n>] [--top-k <n>] [--top-p <n>] [--max-tokens <n>] [--system-prompt <text>] [--greedy]",
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
  let maxTokens = 128;
  let systemPrompt: string | undefined;

  for (let index = 1; index < argv.length; index += 1) {
    const arg = argv[index];
    switch (arg) {
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

  return { source, maxTokens, systemPrompt, overrides };
}

function describeOverrides(
  overrides: Pick<GenerationOptions, "temperature" | "topK" | "topP">,
): string {
  const parts: string[] = [];
  if (overrides.temperature !== undefined) {
    parts.push(`temperature=${overrides.temperature}`);
  }
  if (overrides.topK !== undefined) {
    parts.push(`topK=${overrides.topK}`);
  }
  if (overrides.topP !== undefined) {
    parts.push(`topP=${overrides.topP}`);
  }
  return parts.length === 0 ? "none" : parts.join(", ");
}

function printHelp(): void {
  console.log("Commands:");
  console.log("  q or /exit  quit");
  console.log("  r or /reset reset the prompt cache");
  console.log("  h or /help  show this help");
}

function printSessionIntro(
  interactionProfile: Awaited<ReturnType<typeof loadInteractionProfile>>,
  defaultsEnabled: boolean,
  cli: CliOptions,
): void {
  console.log(
    interactionProfile.kind === "completion"
      ? "\nReady. No chat template was found, so each prompt will be appended as raw text."
      : "\nReady. Chat template detected and enabled.",
  );
  console.log(
    defaultsEnabled
      ? "Checkpoint generation defaults: enabled"
      : "Checkpoint generation defaults: none",
  );
  console.log(`CLI overrides: ${describeOverrides(cli.overrides)}`);
  console.log(`Max tokens per turn: ${cli.maxTokens}`);
  if (cli.systemPrompt !== undefined) {
    console.log("System prompt: enabled for the first turn after each reset");
  }
  printHelp();
  console.log("");
}

function interpretPromptInput(input: string | null): PromptAction {
  if (input === null) {
    return { kind: "quit" };
  }

  const trimmed = input.trim();
  if (trimmed === "") {
    return { kind: "skip" };
  }
  if (trimmed === "q" || trimmed === "/exit") {
    return { kind: "quit" };
  }
  if (trimmed === "r" || trimmed === "/reset") {
    return { kind: "reset" };
  }
  if (trimmed === "h" || trimmed === "/help") {
    return { kind: "help" };
  }
  return { kind: "prompt", promptText: input };
}

function buildTurnInput(
  tokenizer: Awaited<ReturnType<typeof loadPretrainedTokenizer>>,
  interactionProfile: Awaited<ReturnType<typeof loadInteractionProfile>>,
  systemPrompt: string | undefined,
  includeSystemPrompt: boolean,
  promptText: string,
): string {
  if (interactionProfile.kind === "completion") {
    if (includeSystemPrompt) {
      return systemPrompt === undefined ? promptText : `${systemPrompt}\n\n${promptText}`;
    }
    return promptText.startsWith("\n") ? promptText : `\n${promptText}`;
  }

  const messages: ChatMessage[] = [];
  if (includeSystemPrompt && systemPrompt !== undefined) {
    messages.push({ role: "system", content: systemPrompt });
  }
  messages.push({ role: "user", content: promptText });
  return interactionProfile.compileMessages(tokenizer, messages, { addGenerationPrompt: true })
    .text;
}

function runChatTurn(
  model: Awaited<ReturnType<typeof loadCausalLM>>,
  tokenizer: Awaited<ReturnType<typeof loadPretrainedTokenizer>>,
  promptCache: ReturnType<typeof makePromptCache>,
  interactionProfile: Awaited<ReturnType<typeof loadInteractionProfile>>,
  cli: CliOptions,
  promptText: string,
  includeSystemPrompt: boolean,
): string {
  const inputText = buildTurnInput(
    tokenizer,
    interactionProfile,
    cli.systemPrompt,
    includeSystemPrompt,
    promptText,
  );

  process.stdout.write("\nmodel> ");
  const response = generateTextStream(
    model,
    tokenizer,
    inputText,
    {
      maxTokens: cli.maxTokens,
      useCache: true,
      cache: promptCache,
      ...cli.overrides,
    },
    (chunk) => {
      process.stdout.write(chunk);
    },
  );
  process.stdout.write("\n\n");
  return response.text;
}

async function main(): Promise<void> {
  const cli = parseArgs(process.argv.slice(2));

  const reportProgress = createProgressReporter();
  const localSource = await resolvePretrainedSource(cli.source, { onProgress: reportProgress });
  using model = await loadCausalLM(localSource, { onProgress: reportProgress });
  const tokenizer = await loadPretrainedTokenizer(localSource, { onProgress: reportProgress });
  const interactionProfile = await loadInteractionProfile(localSource);
  let promptCache = makePromptCache(model);
  let includeSystemPrompt = true;

  try {
    printSessionIntro(interactionProfile, model.config.generationDefaults !== undefined, cli);

    while (true) {
      const action = interpretPromptInput(prompt("prompt> "));
      switch (action.kind) {
        case "quit":
          console.log(action.kind === "quit" ? "bye" : "");
          return;
        case "skip":
          continue;
        case "help":
          printHelp();
          console.log("");
          continue;
        case "reset":
          promptCache[Symbol.dispose]();
          promptCache = makePromptCache(model);
          includeSystemPrompt = true;
          console.log("chat reset\n");
          continue;
        case "prompt": {
          const response = runChatTurn(
            model,
            tokenizer,
            promptCache,
            interactionProfile,
            cli,
            action.promptText,
            includeSystemPrompt,
          );
          includeSystemPrompt = false;

          if (response === "") {
            console.log("[empty response]\n");
          }
          continue;
        }
      }
    }
  } finally {
    promptCache[Symbol.dispose]();
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
