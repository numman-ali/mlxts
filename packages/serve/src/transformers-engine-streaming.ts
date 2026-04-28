/**
 * Shared text-delta handling for transformer-backed streaming generation.
 * @module
 */

import type { GenerationResult, TokenGenerationEvent } from "@mlxts/transformers";
import { transformersRuntimeStrategy } from "./serve-runtime-strategy";
import type { TransformersGenerationEngineOptions } from "./transformers-engine";
import {
  decodeGeneratedTokenIds,
  emitGenerationProgress,
  promptHasOpenThinking,
  THINK_OPEN,
} from "./transformers-engine-shared";
import type {
  GenerationStreamEvent,
  GenerationUsage,
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

const REPLACEMENT_CHARACTER = "\uFFFD";

function commonPrefixLength(left: string, right: string): number {
  const limit = Math.min(left.length, right.length);
  let index = 0;
  while (index < limit && left[index] === right[index]) {
    index += 1;
  }
  return index;
}

function stableStreamingPrefix(text: string): string {
  let end = text.length;
  while (end > 0 && text[end - 1] === REPLACEMENT_CHARACTER) {
    end -= 1;
  }
  return end === text.length ? text : text.slice(0, end);
}

function flushDecodedText(
  options: TransformersGenerationEngineOptions,
  request: NormalizedGenerationRequest,
  tokenIds: readonly number[],
  text: string,
  flushOptions: { final?: boolean } = {},
): { text: string; delta: string } {
  const rawDecoded = decodeGeneratedTokenIds(options, request, tokenIds);
  const decoded = flushOptions.final === true ? rawDecoded : stableStreamingPrefix(rawDecoded);
  const delta = decoded.slice(commonPrefixLength(text, decoded));
  return { text: decoded, delta };
}

function shouldEmitProgress(
  request: NormalizedGenerationRequest,
  completionTokens: number,
): boolean {
  return completionTokens % 64 === 0 || completionTokens === request.sampling.maxTokens;
}

function streamUsage(
  promptTokens: number,
  completionTokens: number,
  cacheUsage: { readTokens: number; writeTokens: number } = { readTokens: 0, writeTokens: 0 },
): GenerationUsage {
  return {
    promptTokens,
    completionTokens,
    totalTokens: promptTokens + completionTokens,
    ...(cacheUsage.readTokens === 0 && cacheUsage.writeTokens === 0
      ? {}
      : {
          cacheReadTokens: cacheUsage.readTokens,
          cacheWriteTokens: cacheUsage.writeTokens,
        }),
  };
}

function flushStreamDelta(
  options: TransformersGenerationEngineOptions,
  request: NormalizedGenerationRequest,
  generatedTokenIds: readonly number[],
  decodedText: string,
  rawPrefix: string,
  flushOptions: { final?: boolean } = {},
): { decodedText: string; rawPrefix: string; text?: string } {
  const flushed = flushDecodedText(options, request, generatedTokenIds, decodedText, flushOptions);
  const text = rawPrefix === "" ? flushed.delta : `${rawPrefix}${flushed.delta}`;
  return {
    decodedText: flushed.text,
    rawPrefix: text === "" ? rawPrefix : "",
    ...(text === "" ? {} : { text }),
  };
}

function finishReason(reason: GenerationResult["finishReason"]): NormalizedFinishReason {
  return reason === "eos" ? "eos" : "length";
}

function configuredStreamDecodeInterval(options: TransformersGenerationEngineOptions): number {
  return transformersRuntimeStrategy(options).streaming.decodeInterval;
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
    options,
    request,
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
  request: NormalizedGenerationRequest,
  options: TransformersGenerationEngineOptions,
  promptTokens: number,
  state: StreamingDecodeState,
  tokenIds: readonly number[],
  reason: GenerationResult["finishReason"],
  cacheUsage?: { readTokens: number; writeTokens: number },
): { text?: string; done: Extract<GenerationStreamEvent, { type: "done" }> } {
  state.generatedTokenIds = [...tokenIds];
  const flushed = flushStreamDelta(
    options,
    request,
    state.generatedTokenIds,
    state.decodedText,
    state.rawPrefix,
    { final: true },
  );
  state.decodedText = flushed.decodedText;
  state.rawPrefix = flushed.rawPrefix;
  return {
    ...(flushed.text === undefined ? {} : { text: flushed.text }),
    done: {
      type: "done",
      finishReason: finishReason(reason),
      usage: streamUsage(promptTokens, state.generatedTokenIds.length, cacheUsage),
    },
  };
}

export function handleStreamingDoneEvent(
  request: NormalizedGenerationRequest,
  options: TransformersGenerationEngineOptions,
  promptTokens: number,
  state: StreamingDecodeState,
  event: Extract<TokenGenerationEvent, { type: "done" }>,
  cacheUsage?: { readTokens: number; writeTokens: number },
): { text?: string; done: Extract<GenerationStreamEvent, { type: "done" }> } {
  return handleStreamingDone(
    request,
    options,
    promptTokens,
    state,
    event.tokenIds,
    event.finishReason,
    cacheUsage,
  );
}
