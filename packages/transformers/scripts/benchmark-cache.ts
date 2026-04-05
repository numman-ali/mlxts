#!/usr/bin/env bun

import { generateTokens } from "../src/generation";
import { loadCausalLM, loadPretrainedTokenizer } from "../src/load";

function usage(): never {
  console.error(
    "Usage: bun run packages/transformers/scripts/benchmark-cache.ts <model-path-or-repo-id> [prompt-text] [max-tokens]",
  );
  process.exit(1);
}

async function measure(
  modelSource: string,
  promptText: string,
  maxTokens: number,
  useCache: boolean,
): Promise<{ durationMs: number; tokenCount: number }> {
  using model = await loadCausalLM(modelSource);
  const tokenizer = await loadPretrainedTokenizer(modelSource);
  const promptTokenIds = tokenizer.encode(promptText, { addSpecialTokens: true });

  const started = performance.now();
  const result = generateTokens(model, promptTokenIds, {
    maxTokens,
    temperature: 0,
    useCache,
  });
  const durationMs = performance.now() - started;

  return {
    durationMs,
    tokenCount: result.tokenIds.length,
  };
}

async function main(): Promise<void> {
  const modelSource = Bun.argv[2];
  if (modelSource === undefined) {
    usage();
  }

  const promptText = Bun.argv[3] ?? "Once upon a time";
  const maxTokens = Number.parseInt(Bun.argv[4] ?? "64", 10);
  if (!Number.isInteger(maxTokens) || maxTokens <= 0) {
    throw new Error("benchmark-cache: maxTokens must be a positive integer.");
  }

  const uncached = await measure(modelSource, promptText, maxTokens, false);
  const cached = await measure(modelSource, promptText, maxTokens, true);

  console.log(
    JSON.stringify(
      {
        modelSource,
        promptText,
        maxTokens,
        uncached,
        cached,
        speedup: uncached.durationMs / cached.durationMs,
      },
      null,
      2,
    ),
  );
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
