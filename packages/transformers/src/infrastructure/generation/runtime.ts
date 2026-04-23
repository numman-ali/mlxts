/**
 * Internal generation runtime helpers.
 * @module
 */

import {
  clearMemoryCache,
  createStream,
  getRecommendedWorkingSetBytes,
  MxArray,
  mxAsyncEval,
  setWiredLimitBytes,
  synchronize,
  withDefaultStream,
} from "@mlxts/core";
import type {
  CausalLM,
  ForwardOptions,
  GenerationOptions,
  GenerationResult,
  PreparedPrompt,
  SamplerOptions,
  TransformerCache,
} from "../../types";
import { retainPromptInputEmbeddings, retainPromptPositionIds } from "../input-embeddings";
import { SamplerState } from "../sampling";
import { resolveGenerationOptions } from "./defaults";
import {
  inputTensor,
  prefillPromptCache,
  takeLastLogits,
  validatePrefillStepSize,
} from "./helpers";

const PERIODIC_CACHE_CLEAR_INTERVAL = 256;

function createForwardOptions(
  cache: TransformerCache | undefined,
  inputEmbeddings: MxArray | undefined,
  positionIds: MxArray | undefined,
): ForwardOptions | undefined {
  if (cache === undefined && inputEmbeddings === undefined && positionIds === undefined) {
    return undefined;
  }

  const options: ForwardOptions = {};
  if (cache !== undefined) {
    options.cache = cache;
  }
  if (inputEmbeddings !== undefined) {
    options.inputEmbeddings = inputEmbeddings;
  }
  if (positionIds !== undefined) {
    options.positionIds = positionIds;
  }
  return options;
}

export function predictNextTokenWithState(
  model: CausalLM,
  tokenIds: readonly number[] | MxArray,
  cache: TransformerCache | undefined,
  samplerState: SamplerState,
  options: SamplerOptions,
  inputEmbeddings?: MxArray,
  positionIds?: MxArray,
): MxArray {
  const ids = inputTensor(tokenIds);
  const ownsInput = !(tokenIds instanceof MxArray);
  const forwardOptions = createForwardOptions(cache, inputEmbeddings, positionIds);

  try {
    using logits = model.forward(ids, forwardOptions);
    using lastLogits = takeLastLogits(logits, "generateStep");
    return samplerState.sampleTokenTensor(lastLogits, options);
  } finally {
    if (ownsInput) {
      ids.free();
    }
  }
}

export function runGenerationScope<T>(fn: () => T): T {
  using generationStream = createStream();
  const previousWiredLimit = setWiredLimitBytes(getRecommendedWorkingSetBytes());

  try {
    return withDefaultStream(generationStream, fn);
  } finally {
    synchronize(generationStream);
    clearMemoryCache();
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

function takeCurrentToken(currentToken: MxArray | null): MxArray {
  if (currentToken === null) {
    throw new Error("generateTokens: current token was not initialized.");
  }
  return currentToken;
}

function takeScheduledAsyncToken(nextToken: MxArray | null): MxArray {
  if (nextToken === null) {
    throw new Error("generateTokens: async decode did not schedule the next token.");
  }
  return nextToken;
}

export function generateWithCache(
  model: CausalLM,
  promptTokenIds: readonly number[],
  promptInputEmbeddings: MxArray | undefined,
  promptPositionIds: MxArray | undefined,
  options: GenerationOptions,
  eosTokenIds: ReadonlySet<number>,
  prefillStepSize: number,
  generated: number[],
  cache: TransformerCache,
  onToken?: (tokenId: number, generatedTokenIds: readonly number[]) => void,
): GenerationResult {
  using samplerState = new SamplerState(promptTokenIds, options);
  const initialPrompt =
    promptTokenIds.length > 1
      ? prefillPromptCache(
          model,
          promptTokenIds,
          cache,
          prefillStepSize,
          promptInputEmbeddings,
          promptPositionIds,
        )
      : {
          tokenIds: [...promptTokenIds],
          inputEmbeddings: retainPromptInputEmbeddings(
            promptTokenIds,
            promptInputEmbeddings,
            "generateTokens",
          ),
          positionIds: retainPromptPositionIds(promptTokenIds, promptPositionIds, "generateTokens"),
        };
  let finishReason: GenerationResult["finishReason"] = "length";
  let currentToken: MxArray | null = null;
  let nextToken: MxArray | null = null;

  try {
    currentToken = predictNextTokenWithState(
      model,
      initialPrompt.tokenIds,
      cache,
      samplerState,
      options,
      initialPrompt.inputEmbeddings ?? undefined,
      initialPrompt.positionIds ?? undefined,
    );
    mxAsyncEval(currentToken);

    for (let index = 0; index < options.maxTokens; index += 1) {
      const activeToken = takeCurrentToken(currentToken);
      if (index + 1 < options.maxTokens) {
        nextToken = scheduleAsyncCachedToken(model, activeToken, cache, samplerState, options);
      }

      const tokenId = activeToken.item();
      generated.push(tokenId);
      onToken?.(tokenId, generated);

      const eosFinishReason = finishIfEos(eosTokenIds, tokenId);
      if (eosFinishReason !== null) {
        finishReason = eosFinishReason;
        nextToken?.free();
        nextToken = null;
        break;
      }

      maybeClearGenerationCache(index);
      if (index + 1 >= options.maxTokens) {
        break;
      }

      activeToken.free();
      const scheduledToken = takeScheduledAsyncToken(nextToken);
      nextToken = null;
      currentToken = scheduledToken;
    }
  } finally {
    initialPrompt.inputEmbeddings?.free();
    initialPrompt.positionIds?.free();
    currentToken?.free();
    nextToken?.free();
  }

  return { tokenIds: generated, finishReason };
}

export function generateWithoutCache(
  model: CausalLM,
  promptTokenIds: readonly number[],
  promptInputEmbeddings: MxArray | undefined,
  promptPositionIds: MxArray | undefined,
  options: GenerationOptions,
  eosTokenIds: ReadonlySet<number>,
  generated: number[],
  onToken?: (tokenId: number, generatedTokenIds: readonly number[]) => void,
): GenerationResult {
  using samplerState = new SamplerState(promptTokenIds, options);
  const runningPrompt = [...promptTokenIds];
  const initialInputEmbeddings = retainPromptInputEmbeddings(
    promptTokenIds,
    promptInputEmbeddings,
    "generateTokens",
  );
  const initialPositionIds = retainPromptPositionIds(
    promptTokenIds,
    promptPositionIds,
    "generateTokens",
  );
  let finishReason: GenerationResult["finishReason"] = "length";
  let currentToken: MxArray | null = null;

  try {
    currentToken = predictNextTokenWithState(
      model,
      runningPrompt,
      undefined,
      samplerState,
      options,
      initialInputEmbeddings ?? undefined,
      initialPositionIds ?? undefined,
    );
    for (let index = 0; index < options.maxTokens; index += 1) {
      const activeToken = takeCurrentToken(currentToken);
      const tokenId = activeToken.item();
      generated.push(tokenId);
      onToken?.(tokenId, generated);
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

      samplerState.appendToken(activeToken);
      const nextToken = predictNextTokenWithState(
        model,
        runningPrompt,
        undefined,
        samplerState,
        options,
      );
      activeToken.free();
      currentToken = nextToken;
    }
  } finally {
    initialInputEmbeddings?.free();
    initialPositionIds?.free();
    currentToken?.free();
  }

  return { tokenIds: generated, finishReason };
}

export function validateCacheOptions(options: GenerationOptions): void {
  if (options.cache !== undefined && options.useCache === false) {
    throw new Error("generation: cache cannot be provided when useCache is false.");
  }
}

export function generateTokensInternal(
  model: CausalLM,
  promptTokenIds: readonly number[],
  options: GenerationOptions,
  makePromptCache: (model: CausalLM) => TransformerCache,
  onToken?: (tokenId: number, generatedTokenIds: readonly number[]) => void,
  promptInputEmbeddings?: MxArray,
  promptPositionIds?: MxArray,
): GenerationResult {
  validateCacheOptions(options);

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

  return runGenerationScope(() => {
    if (!useCache) {
      return generateWithoutCache(
        model,
        promptTokenIds,
        promptInputEmbeddings,
        promptPositionIds,
        resolvedOptions,
        eosTokenIds,
        generated,
        onToken,
      );
    }

    const externalCache = resolvedOptions.cache;
    if (externalCache !== undefined) {
      return generateWithCache(
        model,
        promptTokenIds,
        promptInputEmbeddings,
        promptPositionIds,
        resolvedOptions,
        eosTokenIds,
        prefillStepSize,
        generated,
        externalCache,
        onToken,
      );
    }

    using cache = makePromptCache(model);
    return generateWithCache(
      model,
      promptTokenIds,
      promptInputEmbeddings,
      promptPositionIds,
      resolvedOptions,
      eosTokenIds,
      prefillStepSize,
      generated,
      cache,
      onToken,
    );
  });
}

export function generatePreparedTokensInternal(
  model: CausalLM,
  prompt: PreparedPrompt,
  options: GenerationOptions,
  makePromptCache: (model: CausalLM) => TransformerCache,
  onToken?: (tokenId: number, generatedTokenIds: readonly number[]) => void,
): GenerationResult {
  return generateTokensInternal(
    model,
    prompt.tokenIds,
    options,
    makePromptCache,
    onToken,
    prompt.inputEmbeddings,
    prompt.positionIds,
  );
}
