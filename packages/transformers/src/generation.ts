/**
 * Explicit generation helpers built on top of decoder model forward passes.
 * @module
 */

import type { MxArray } from "@mlxts/core";
import type { Tokenizer } from "@mlxts/tokenizers";
import { resolveGenerationOptions } from "./infrastructure/generation/defaults";
import {
  generatePreparedTokensInternal,
  generateTokensInternal,
  predictNextTokenWithState,
} from "./infrastructure/generation/runtime";
import { SamplerState } from "./infrastructure/sampling";
import type {
  CausalLM,
  GenerationOptions,
  GenerationResult,
  PreparedPrompt,
  SamplerOptions,
  TextGenerationResult,
  TransformerCache,
} from "./types";

/** Create the default prompt cache for a decoder model. */
export function makePromptCache(model: CausalLM): TransformerCache {
  return model.createCache();
}

const STREAM_DECODE_INTERVAL = 32;

function commonPrefixLength(left: string, right: string): number {
  const limit = Math.min(left.length, right.length);
  let index = 0;
  while (index < limit && left[index] === right[index]) {
    index += 1;
  }
  return index;
}

function flushDecodedText(
  tokenizer: Tokenizer,
  tokenIds: readonly number[],
  text: string,
  onText: (chunk: string) => void,
): string {
  const decoded = tokenizer.decode([...tokenIds], { skipSpecialTokens: true });
  const delta = decoded.slice(commonPrefixLength(text, decoded));
  if (delta !== "") {
    onText(delta);
  }
  return decoded;
}

/** Run one generation step and sample a token from the final logits. */
export function generateStep(
  model: CausalLM,
  tokenIds: readonly number[] | MxArray,
  cache: TransformerCache | undefined,
  history: readonly number[],
  options: SamplerOptions = {},
  inputEmbeddings?: MxArray,
  positionIds?: MxArray,
): number {
  using samplerState = new SamplerState(history, options);
  using nextToken = predictNextTokenWithState(
    model,
    tokenIds,
    cache,
    samplerState,
    options,
    inputEmbeddings,
    positionIds,
  );
  return nextToken.item();
}

/** Generate continuation token IDs from a prompt. */
export function generateTokens(
  model: CausalLM,
  promptTokenIds: readonly number[],
  options: GenerationOptions,
  onToken?: (tokenId: number, generatedTokenIds: readonly number[]) => void,
): GenerationResult {
  return generateTokensInternal(model, promptTokenIds, options, makePromptCache, onToken);
}

/** Generate continuation token IDs from a prepared prompt with embeddings and/or position ids. */
export function generatePreparedTokens(
  model: CausalLM,
  prompt: PreparedPrompt,
  options: GenerationOptions,
  onToken?: (tokenId: number, generatedTokenIds: readonly number[]) => void,
): GenerationResult {
  return generatePreparedTokensInternal(model, prompt, options, makePromptCache, onToken);
}

/** Tokenize a prompt, stream decoded continuation chunks, and return the final text. */
export function generateTextStream(
  model: CausalLM,
  tokenizer: Tokenizer,
  prompt: string,
  options: GenerationOptions,
  onText: (chunk: string) => void,
  onToken?: (tokenId: number, generatedTokenIds: readonly number[]) => void,
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
      onToken?.(_tokenId, tokenIds);
      if (tokenIds.length % STREAM_DECODE_INTERVAL === 0) {
        text = flushDecodedText(tokenizer, tokenIds, text, onText);
      }
    },
  );
  text = flushDecodedText(tokenizer, result.tokenIds, text, onText);

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
