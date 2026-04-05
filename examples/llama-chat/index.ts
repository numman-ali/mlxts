#!/usr/bin/env bun

import { generateText, loadCausalLM, loadPretrainedTokenizer } from "@mlxts/transformers";

function usage(): never {
  console.error("Usage: bun run examples/llama-chat/index.ts <model-path-or-repo-id>");
  process.exit(1);
}

async function main(): Promise<void> {
  const source = process.argv[2];
  if (source === undefined || source.trim() === "") {
    usage();
  }

  console.log(`Loading model from ${source}...`);
  using model = await loadCausalLM(source);
  const tokenizer = await loadPretrainedTokenizer(source);

  console.log("Ready. No chat template is applied. Type /exit or press Ctrl-D to quit.\n");

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

    const response = generateText(model, tokenizer, promptText, {
      maxTokens: 128,
      temperature: 0.8,
      topP: 0.95,
      useCache: true,
    });
    console.log(`\nmodel> ${response}\n`);
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
