/**
 * Explicit generation helpers built on top of decoder model forward passes.
 * @module
 */

import {
  clearMemoryCache,
  createStream,
  getRecommendedWorkingSetBytes,
  MxArray,
  mxAsyncEval,
  mxEval,
  setWiredLimitBytes,
  synchronize,
  withDefaultStream,
} from "@mlxts/core";
import type { Tokenizer } from "@mlxts/tokenizers";
import { resolveGenerationOptions } from "./infrastructure/generation-defaults";
import {
  inputTensor,
  prefillPromptCache,
  takeLastLogits,
  validatePrefillStepSize,
} from "./infrastructure/generation-helpers";
import { SamplerState } from "./infrastructure/sampling";
import type {
  CausalLM,
  GenerationOptions,
  GenerationResult,
  SamplerOptions,
  TransformerCache,
} from "./types";

const PERIODIC_CACHE_CLEAR_INTERVAL = 256;

function predictNextTokenWithState(
  model: CausalLM,
  tokenIds: readonly number[] | MxArray,
  cache: TransformerCache | undefined,
  samplerState: SamplerState,
  options: SamplerOptions,
): MxArray {
  const ids = inputTensor(tokenIds);
  const ownsInput = !(tokenIds instanceof MxArray);

  try {
    using logits = model.forward(ids, cache === undefined ? undefined : { cache });
    using lastLogits = takeLastLogits(logits, "generateStep");
    return samplerState.sampleTokenTensor(lastLogits, options);
  } finally {
    if (ownsInput) {
      ids.free();
    }
  }
}

function runGenerationScope<T>(fn: () => T): T {
  using generationStream = createStream();
  const previousWiredLimit = setWiredLimitBytes(getRecommendedWorkingSetBytes());

  try {
    return withDefaultStream(generationStream, fn);
  } finally {
    synchronize(generationStream);
    setWiredLimitBytes(previousWiredLimit);
  }
}

function finishIfEos(
  eosTokenIds: ReadonlySet<number>,
  tokenId: number,
): GenerationResult["finishReason"] | null {
  return eosTokenIds.has(tokenId) ? "eos" : null;
}

function maybeClearGenerationCache(index: number): void {
  if ((index + 1) % PERIODIC_CACHE_CLEAR_INTERVAL === 0) {
    clearMemoryCache();
  }
}

function scheduleAsyncCachedToken(
  model: CausalLM,
  currentToken: MxArray,
  cache: TransformerCache,
  samplerState: SamplerState,
  options: SamplerOptions,
): MxArray {
  samplerState.appendToken(currentToken);
  const nextToken = predictNextTokenWithState(model, currentToken, cache, samplerState, options);
  mxAsyncEval(nextToken);
  return nextToken;
}

function takeScheduledAsyncToken(nextToken: MxArray | null): MxArray {
  if (nextToken === null) {
    throw new Error("generateTokens: async decode did not schedule the next token.");
  }
  return nextToken;
}

function generateWithCache(
  model: CausalLM,
  promptTokenIds: readonly number[],
  options: GenerationOptions,
  eosTokenIds: ReadonlySet<number>,
  prefillStepSize: number,
  generated: number[],
): GenerationResult {
  using samplerState = new SamplerState(promptTokenIds, options);
  using cache = makePromptCache(model);
  const initialPrompt =
    promptTokenIds.length > 1
      ? prefillPromptCache(model, promptTokenIds, cache, prefillStepSize)
      : [...promptTokenIds];
  const useAsyncDecode = true;
  let finishReason: GenerationResult["finishReason"] = "length";
  let currentToken = predictNextTokenWithState(model, initialPrompt, cache, samplerState, options);
  let nextToken: MxArray | null = null;

  try {
    if (useAsyncDecode) {
      mxAsyncEval(currentToken);
    }

    for (let index = 0; index < options.maxTokens; index += 1) {
      if (useAsyncDecode && index + 1 < options.maxTokens) {
        nextToken = scheduleAsyncCachedToken(model, currentToken, cache, samplerState, options);
      }

      mxEval(currentToken);
      const tokenId = currentToken.item();
      generated.push(tokenId);

      const eosFinishReason = finishIfEos(eosTokenIds, tokenId);
      if (eosFinishReason !== null) {
        finishReason = eosFinishReason;
        nextToken?.free();
        break;
      }

      maybeClearGenerationCache(index);
      if (index + 1 >= options.maxTokens) {
        break;
      }

      if (useAsyncDecode) {
        currentToken.free();
        const scheduledToken = takeScheduledAsyncToken(nextToken);
        nextToken = null;
        currentToken = scheduledToken;
        continue;
      }

      samplerState.appendToken(currentToken);
      currentToken.free();
      currentToken = predictNextTokenWithState(model, [tokenId], cache, samplerState, options);
    }
  } finally {
    currentToken.free();
    nextToken?.free();
  }

  return { tokenIds: generated, finishReason };
}

function generateWithoutCache(
  model: CausalLM,
  promptTokenIds: readonly number[],
  options: GenerationOptions,
  eosTokenIds: ReadonlySet<number>,
  generated: number[],
): GenerationResult {
  using samplerState = new SamplerState(promptTokenIds, options);
  const runningPrompt = [...promptTokenIds];
  let finishReason: GenerationResult["finishReason"] = "length";
  let currentToken = predictNextTokenWithState(
    model,
    runningPrompt,
    undefined,
    samplerState,
    options,
  );

  try {
    for (let index = 0; index < options.maxTokens; index += 1) {
      currentToken.eval();
      const tokenId = currentToken.item();
      generated.push(tokenId);
      runningPrompt.push(tokenId);

      const eosFinishReason = finishIfEos(eosTokenIds, tokenId);
      if (eosFinishReason !== null) {
        finishReason = eosFinishReason;
        break;
      }

      maybeClearGenerationCache(index);
      if (index + 1 >= options.maxTokens) {
        break;
      }

      samplerState.appendToken(currentToken);
      const nextToken = predictNextTokenWithState(
        model,
        runningPrompt,
        undefined,
        samplerState,
        options,
      );
      currentToken.free();
      currentToken = nextToken;
    }
  } finally {
    currentToken.free();
  }

  return { tokenIds: generated, finishReason };
}

/** Create the default prompt cache for a decoder model. */
export function makePromptCache(model: CausalLM): TransformerCache {
  return model.createCache();
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
  nextToken.eval();
  return nextToken.item();
}

/** Generate continuation token IDs from a prompt. */
export function generateTokens(
  model: CausalLM,
  promptTokenIds: readonly number[],
  options: GenerationOptions,
): GenerationResult {
  if (promptTokenIds.length === 0) {
    throw new Error("generateTokens: promptTokenIds must contain at least one token.");
  }
  if (options.maxTokens < 0) {
    throw new Error(`generateTokens: maxTokens must be >= 0, got ${options.maxTokens}.`);
  }
  if (options.maxTokens === 0) {
    return { tokenIds: [], finishReason: "length" };
  }

  const resolvedOptions = resolveGenerationOptions(model, undefined, options);
  const generated: number[] = [];
  const eosTokenIds = new Set(resolvedOptions.eosTokenIds ?? []);
  const useCache = resolvedOptions.useCache ?? true;
  const prefillStepSize = resolvedOptions.prefillStepSize ?? 2048;

  validatePrefillStepSize(prefillStepSize, "generateTokens");
  return runGenerationScope(() =>
    useCache
      ? generateWithCache(
          model,
          promptTokenIds,
          resolvedOptions,
          eosTokenIds,
          prefillStepSize,
          generated,
        )
      : generateWithoutCache(model, promptTokenIds, resolvedOptions, eosTokenIds, generated),
  );
}

/** Tokenize a prompt, generate new tokens, and decode the continuation text. */
export function generateText(
  model: CausalLM,
  tokenizer: Tokenizer,
  prompt: string,
  options: GenerationOptions,
): string {
  const resolvedOptions = resolveGenerationOptions(model, tokenizer, options);
  const promptTokenIds = tokenizer.encode(prompt, {
    addSpecialTokens: resolvedOptions.addSpecialTokens ?? true,
  });
  const result = generateTokens(model, promptTokenIds, resolvedOptions);
  return tokenizer.decode(result.tokenIds, { skipSpecialTokens: true });
}
