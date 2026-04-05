#!/usr/bin/env bun

import {
  type ChatMessage,
  generateText,
  loadCausalLM,
  loadChatTemplate,
  loadPretrainedTokenizer,
  type PretrainedLoadProgressEvent,
  resolvePretrainedSource,
} from "@mlxts/transformers";

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
  console.error("Usage: bun run examples/llama-chat/index.ts <model-path-or-repo-id>");
  process.exit(1);
}

async function main(): Promise<void> {
  const source = process.argv[2];
  if (source === undefined || source.trim() === "") {
    usage();
  }

  const reportProgress = createProgressReporter();
  const localSource = await resolvePretrainedSource(source, { onProgress: reportProgress });
  using model = await loadCausalLM(localSource, { onProgress: reportProgress });
  const tokenizer = await loadPretrainedTokenizer(localSource, { onProgress: reportProgress });
  const chatTemplate = await loadChatTemplate(localSource);
  const messages: ChatMessage[] = [];

  console.log(
    chatTemplate === null
      ? "\nReady. No chat template was found, so prompts will be sent as raw text. Type /exit or press Ctrl-D to quit.\n"
      : "\nReady. Chat template detected and enabled. Type /exit or press Ctrl-D to quit.\n",
  );

  while (true) {
    const promptText = prompt("prompt> ");
    if (promptText === null) {
      console.log("\nbye");
      return;
    }

    const trimmed = promptText.trim();
    if (trimmed === "" || trimmed === "/exit") {
      console.log("bye");
      return;
    }

    messages.push({ role: "user", content: promptText });
    const inputText =
      chatTemplate === null
        ? promptText
        : chatTemplate.format(messages, {
            addGenerationPrompt: true,
          });
    const response = generateText(model, tokenizer, inputText, {
      maxTokens: 128,
      temperature: 0.8,
      topP: 0.95,
      useCache: true,
    });
    messages.push({ role: "assistant", content: response });
    console.log(`\nmodel> ${response}\n`);
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
