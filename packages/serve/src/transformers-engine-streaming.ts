/**
 * Shared text-delta handling for transformer-backed streaming generation.
 * @module
 */

import type { Tokenizer } from "@mlxts/tokenizers";
import type { GenerationResult, TokenGenerationEvent } from "@mlxts/transformers";
import type { TransformersGenerationEngineOptions } from "./transformers-engine";
import {
  emitGenerationProgress,
  promptHasOpenThinking,
  THINK_OPEN,
} from "./transformers-engine-shared";
import type {
  GenerationStreamEvent,
  NormalizedFinishReason,
  NormalizedGenerationRequest,
} from "./types";

export const DEFAULT_STREAM_DECODE_INTERVAL = 1;

type PromptText = {
  text: string;
};

export type StreamingDecodeState = {
  generatedTokenIds: number[];
  decodedText: string;
  rawPrefix: string;
};

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
): { text: string; delta: string } {
  const decoded = tokenizer.decode([...tokenIds], { skipSpecialTokens: true });
  const delta = decoded.slice(commonPrefixLength(text, decoded));
  return { text: decoded, delta };
}

function shouldEmitProgress(
  request: NormalizedGenerationRequest,
  completionTokens: number,
): boolean {
  return completionTokens % 64 === 0 || completionTokens === request.sampling.maxTokens;
}

function streamUsage(promptTokens: number, completionTokens: number) {
  return {
    promptTokens,
    completionTokens,
    totalTokens: promptTokens + completionTokens,
  };
}

function flushStreamDelta(
  tokenizer: Tokenizer,
  generatedTokenIds: readonly number[],
  decodedText: string,
  rawPrefix: string,
): { decodedText: string; rawPrefix: string; text?: string } {
  const flushed = flushDecodedText(tokenizer, generatedTokenIds, decodedText);
  const text = rawPrefix === "" ? flushed.delta : `${rawPrefix}${flushed.delta}`;
  return {
    decodedText: flushed.text,
    rawPrefix: "",
    ...(text === "" ? {} : { text }),
  };
}

function finishReason(reason: GenerationResult["finishReason"]): NormalizedFinishReason {
  return reason === "eos" ? "eos" : "length";
}

function configuredStreamDecodeInterval(options: TransformersGenerationEngineOptions): number {
  const interval = options.streamDecodeInterval ?? DEFAULT_STREAM_DECODE_INTERVAL;
  if (!Number.isInteger(interval) || interval <= 0) {
    throw new Error("streamDecodeInterval must be a positive integer.");
  }
  return interval;
}

export function streamDecodeInterval(
  options: TransformersGenerationEngineOptions,
  stop: readonly string[] | undefined,
): number {
  return stop === undefined || stop.length === 0 ? configuredStreamDecodeInterval(options) : 1;
}

export function createStreamingDecodeState(prompt: PromptText | null): StreamingDecodeState {
  return {
    generatedTokenIds: [],
    decodedText: "",
    rawPrefix: promptHasOpenThinking(prompt) ? THINK_OPEN : "",
  };
}

export function handleStreamingTokenDelta(
  request: NormalizedGenerationRequest,
  options: TransformersGenerationEngineOptions,
  promptTokens: number,
  decodeInterval: number,
  state: StreamingDecodeState,
  tokenId: number,
  completionTokens: number,
): string | undefined {
  state.generatedTokenIds.push(tokenId);
  if (shouldEmitProgress(request, completionTokens)) {
    emitGenerationProgress(options, request, promptTokens, completionTokens);
  }
  if (completionTokens % decodeInterval !== 0) {
    return undefined;
  }

  const flushed = flushStreamDelta(
    options.tokenizer,
    state.generatedTokenIds,
    state.decodedText,
    state.rawPrefix,
  );
  state.decodedText = flushed.decodedText;
  state.rawPrefix = flushed.rawPrefix;
  return flushed.text;
}

export function handleStreamingTokenEvent(
  request: NormalizedGenerationRequest,
  options: TransformersGenerationEngineOptions,
  promptTokens: number,
  decodeInterval: number,
  state: StreamingDecodeState,
  event: Extract<TokenGenerationEvent, { type: "token" }>,
): string | undefined {
  return handleStreamingTokenDelta(
    request,
    options,
    promptTokens,
    decodeInterval,
    state,
    event.tokenId,
    event.completionTokens,
  );
}

export function handleStreamingDone(
  options: TransformersGenerationEngineOptions,
  promptTokens: number,
  state: StreamingDecodeState,
  tokenIds: readonly number[],
  reason: GenerationResult["finishReason"],
): { text?: string; done: Extract<GenerationStreamEvent, { type: "done" }> } {
  state.generatedTokenIds = [...tokenIds];
  const flushed = flushStreamDelta(
    options.tokenizer,
    state.generatedTokenIds,
    state.decodedText,
    state.rawPrefix,
  );
  state.decodedText = flushed.decodedText;
  state.rawPrefix = flushed.rawPrefix;
  return {
    ...(flushed.text === undefined ? {} : { text: flushed.text }),
    done: {
      type: "done",
      finishReason: finishReason(reason),
      usage: streamUsage(promptTokens, state.generatedTokenIds.length),
    },
  };
}

export function handleStreamingDoneEvent(
  options: TransformersGenerationEngineOptions,
  promptTokens: number,
  state: StreamingDecodeState,
  event: Extract<TokenGenerationEvent, { type: "done" }>,
): { text?: string; done: Extract<GenerationStreamEvent, { type: "done" }> } {
  return handleStreamingDone(options, promptTokens, state, event.tokenIds, event.finishReason);
}
