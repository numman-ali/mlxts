/**
 * Async token-event generation helpers.
 * @module
 */

import {
  clearMemoryCache,
  createStream,
  getRecommendedWorkingSetBytes,
  type MxArray,
  mxAsyncEval,
  setWiredLimitBytes,
  synchronize,
  withDefaultStream,
} from "@mlxts/core";
import type {
  CausalLM,
  GenerationOptions,
  GenerationResult,
  PreparedPrompt,
  TokenGenerationEvent,
  TransformerCache,
} from "../../types";
import { retainPromptInputEmbeddings, retainPromptPositionIds } from "../input-embeddings";
import { SamplerState } from "../sampling";
import { throwIfGenerationAborted } from "./cancellation";
import { resolveGenerationOptions } from "./defaults";
import { type PrefilledPrompt, prefillPromptCache, validatePrefillStepSize } from "./helpers";
import {
  finishIfEos,
  maybeClearGenerationCache,
  predictNextTokenWithState,
  scheduleAsyncCachedToken,
  takeCurrentToken,
  takeScheduledAsyncToken,
  validateCacheOptions,
} from "./runtime";

function createGenerationScope(): {
  stream: ReturnType<typeof createStream>;
  previousWiredLimit: number;
} {
  return {
    stream: createStream(),
    previousWiredLimit: setWiredLimitBytes(getRecommendedWorkingSetBytes()),
  };
}

function closeGenerationScope(scope: {
  stream: ReturnType<typeof createStream>;
  previousWiredLimit: number;
}): void {
  try {
    synchronize(scope.stream);
    clearMemoryCache();
  } finally {
    setWiredLimitBytes(scope.previousWiredLimit);
    scope.stream[Symbol.dispose]();
  }
}

async function* streamWithCache(
  model: CausalLM,
  promptTokenIds: readonly number[],
  promptInputEmbeddings: MxArray | undefined,
  promptPositionIds: MxArray | undefined,
  options: GenerationOptions,
  eosTokenIds: ReadonlySet<number>,
  prefillStepSize: number,
  cache: TransformerCache,
): AsyncIterable<TokenGenerationEvent> {
  const scope = createGenerationScope();
  const samplerState = new SamplerState(promptTokenIds, options);
  const generated: number[] = [];
  let finishReason: GenerationResult["finishReason"] = "length";
  let initialPrompt: PrefilledPrompt | null = null;
  let currentToken: MxArray | null = null;
  let nextToken: MxArray | null = null;

  try {
    initialPrompt = withDefaultStream(scope.stream, () =>
      promptTokenIds.length > 1
        ? prefillPromptCache(
            model,
            promptTokenIds,
            cache,
            prefillStepSize,
            promptInputEmbeddings,
            promptPositionIds,
            options.onPrefillProgress,
            options.abortSignal,
          )
        : {
            tokenIds: [...promptTokenIds],
            inputEmbeddings: retainPromptInputEmbeddings(
              promptTokenIds,
              promptInputEmbeddings,
              "generateTokens",
            ),
            positionIds: retainPromptPositionIds(
              promptTokenIds,
              promptPositionIds,
              "generateTokens",
            ),
          },
    );
    const activePrompt = initialPrompt;
    throwIfGenerationAborted(options.abortSignal, "generateTokens");
    currentToken = withDefaultStream(scope.stream, () =>
      predictNextTokenWithState(
        model,
        activePrompt.tokenIds,
        cache,
        samplerState,
        options,
        activePrompt.inputEmbeddings ?? undefined,
        activePrompt.positionIds ?? undefined,
      ),
    );
    const initialToken = takeCurrentToken(currentToken);
    withDefaultStream(scope.stream, () => {
      mxAsyncEval(initialToken);
    });

    for (let index = 0; index < options.maxTokens; index += 1) {
      throwIfGenerationAborted(options.abortSignal, "generateTokens");
      const activeToken = takeCurrentToken(currentToken);
      if (index + 1 < options.maxTokens) {
        nextToken = withDefaultStream(scope.stream, () =>
          scheduleAsyncCachedToken(model, activeToken, cache, samplerState, options),
        );
      }

      const tokenId = withDefaultStream(scope.stream, () => activeToken.item());
      generated.push(tokenId);
      yield { type: "token", tokenId, completionTokens: generated.length };
      throwIfGenerationAborted(options.abortSignal, "generateTokens");

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

    yield { type: "done", tokenIds: [...generated], finishReason };
  } finally {
    samplerState[Symbol.dispose]();
    initialPrompt?.inputEmbeddings?.free();
    initialPrompt?.positionIds?.free();
    currentToken?.free();
    nextToken?.free();
    closeGenerationScope(scope);
  }
}

async function* streamWithoutCache(
  model: CausalLM,
  promptTokenIds: readonly number[],
  promptInputEmbeddings: MxArray | undefined,
  promptPositionIds: MxArray | undefined,
  options: GenerationOptions,
  eosTokenIds: ReadonlySet<number>,
): AsyncIterable<TokenGenerationEvent> {
  const scope = createGenerationScope();
  const samplerState = new SamplerState(promptTokenIds, options);
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
  const generated: number[] = [];
  let finishReason: GenerationResult["finishReason"] = "length";
  let currentToken: MxArray | null = null;

  try {
    throwIfGenerationAborted(options.abortSignal, "generateTokens");
    currentToken = withDefaultStream(scope.stream, () =>
      predictNextTokenWithState(
        model,
        runningPrompt,
        undefined,
        samplerState,
        options,
        initialInputEmbeddings ?? undefined,
        initialPositionIds ?? undefined,
      ),
    );
    for (let index = 0; index < options.maxTokens; index += 1) {
      throwIfGenerationAborted(options.abortSignal, "generateTokens");
      const activeToken = takeCurrentToken(currentToken);
      const tokenId = withDefaultStream(scope.stream, () => activeToken.item());
      generated.push(tokenId);
      yield { type: "token", tokenId, completionTokens: generated.length };
      throwIfGenerationAborted(options.abortSignal, "generateTokens");
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

      currentToken = withDefaultStream(scope.stream, () => {
        samplerState.appendToken(activeToken);
        return predictNextTokenWithState(model, runningPrompt, undefined, samplerState, options);
      });
      activeToken.free();
    }

    yield { type: "done", tokenIds: [...generated], finishReason };
  } finally {
    samplerState[Symbol.dispose]();
    initialInputEmbeddings?.free();
    initialPositionIds?.free();
    currentToken?.free();
    closeGenerationScope(scope);
  }
}

export function generateTokenEventsInternal(
  model: CausalLM,
  promptTokenIds: readonly number[],
  options: GenerationOptions,
  makePromptCache: (model: CausalLM) => TransformerCache,
  promptInputEmbeddings?: MxArray,
  promptPositionIds?: MxArray,
): AsyncIterable<TokenGenerationEvent> {
  validateCacheOptions(options);

  if (promptTokenIds.length === 0) {
    throw new Error("generateTokens: promptTokenIds must contain at least one token.");
  }
  if (options.maxTokens < 0) {
    throw new Error(`generateTokens: maxTokens must be >= 0, got ${options.maxTokens}.`);
  }
  if (options.maxTokens === 0) {
    return (async function* (): AsyncIterable<TokenGenerationEvent> {
      yield { type: "done", tokenIds: [], finishReason: "length" };
    })();
  }

  const resolvedOptions = resolveGenerationOptions(model, undefined, options);
  const eosTokenIds = new Set(resolvedOptions.eosTokenIds ?? []);
  const useCache = resolvedOptions.useCache ?? true;
  const prefillStepSize = resolvedOptions.prefillStepSize ?? 2048;

  validatePrefillStepSize(prefillStepSize, "generateTokens");

  if (!useCache) {
    return streamWithoutCache(
      model,
      promptTokenIds,
      promptInputEmbeddings,
      promptPositionIds,
      resolvedOptions,
      eosTokenIds,
    );
  }

  const externalCache = resolvedOptions.cache;
  if (externalCache !== undefined) {
    return streamWithCache(
      model,
      promptTokenIds,
      promptInputEmbeddings,
      promptPositionIds,
      resolvedOptions,
      eosTokenIds,
      prefillStepSize,
      externalCache,
    );
  }

  return (async function* (): AsyncIterable<TokenGenerationEvent> {
    using cache = makePromptCache(model);
    yield* streamWithCache(
      model,
      promptTokenIds,
      promptInputEmbeddings,
      promptPositionIds,
      resolvedOptions,
      eosTokenIds,
      prefillStepSize,
      cache,
    );
  })();
}

export function generatePreparedTokenEventsInternal(
  model: CausalLM,
  prompt: PreparedPrompt,
  options: GenerationOptions,
  makePromptCache: (model: CausalLM) => TransformerCache,
): AsyncIterable<TokenGenerationEvent> {
  return generateTokenEventsInternal(
    model,
    prompt.tokenIds,
    options,
    makePromptCache,
    prompt.inputEmbeddings,
    prompt.positionIds,
  );
}
