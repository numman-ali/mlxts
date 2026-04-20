/**
 * Explicit generation helpers built on top of decoder model forward passes.
 * @module
 */

import type { MxArray } from "@mlxts/core";
import type { Tokenizer } from "@mlxts/tokenizers";
import { resolveGenerationOptions } from "./infrastructure/generation/defaults";
import {
  generateTokensInternal,
  predictNextTokenWithState,
} from "./infrastructure/generation/runtime";
import { SamplerState } from "./infrastructure/sampling";
import type {
  CausalLM,
  GenerationOptions,
  GenerationResult,
  SamplerOptions,
  TextGenerationResult,
  TransformerCache,
} from "./types";

/** Create the default prompt cache for a decoder model. */
export function makePromptCache(model: CausalLM): TransformerCache {
  return model.createCache();
}

function commonPrefixLength(left: string, right: string): number {
  const limit = Math.min(left.length, right.length);
  let index = 0;
  while (index < limit && left[index] === right[index]) {
    index += 1;
  }
  return index;
}

/** Run one generation step and sample a token from the final logits. */
export function generateStep(
  model: CausalLM,
  tokenIds: readonly number[] | MxArray,
  cache: TransformerCache | undefined,
  history: readonly number[],
  options: SamplerOptions = {},
): number {
  using samplerState = new SamplerState(history, options);
  using nextToken = predictNextTokenWithState(model, tokenIds, cache, samplerState, options);
  return nextToken.item();
}

/** Generate continuation token IDs from a prompt. */
export function generateTokens(
  model: CausalLM,
  promptTokenIds: readonly number[],
  options: GenerationOptions,
): GenerationResult {
  return generateTokensInternal(model, promptTokenIds, options, makePromptCache);
}

/** Tokenize a prompt, stream decoded continuation chunks, and return the final text. */
export function generateTextStream(
  model: CausalLM,
  tokenizer: Tokenizer,
  prompt: string,
  options: GenerationOptions,
  onText: (chunk: string) => void,
): TextGenerationResult {
  const resolvedOptions = resolveGenerationOptions(model, tokenizer, options);
  const promptTokenIds = tokenizer.encode(prompt, {
    addSpecialTokens: resolvedOptions.addSpecialTokens ?? true,
  });
  let text = "";
  const result = generateTokensInternal(
    model,
    promptTokenIds,
    resolvedOptions,
    makePromptCache,
    (_tokenId, tokenIds) => {
      const decoded = tokenizer.decode([...tokenIds], { skipSpecialTokens: true });
      const delta = decoded.slice(commonPrefixLength(text, decoded));
      text = decoded;
      if (delta !== "") {
        onText(delta);
      }
    },
  );

  return { ...result, text };
}

/** Tokenize a prompt, generate new tokens, and decode the continuation text. */
export function generateText(
  model: CausalLM,
  tokenizer: Tokenizer,
  prompt: string,
  options: GenerationOptions,
): string {
  return generateTextStream(model, tokenizer, prompt, options, () => undefined).text;
}
