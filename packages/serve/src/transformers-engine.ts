/**
 * Adapter from @mlxts/transformers generation into the serving engine contract.
 * @module
 */

import type { Tokenizer } from "@mlxts/tokenizers";
import {
  type CausalLM,
  generatePreparedTokenEvents,
  generateTokenEvents,
  type InteractionProfile,
  type TokenGenerationEvent,
} from "@mlxts/transformers";
import { ModelExecutionLane } from "./model-execution-lane";
import { createContinuousTransformersGeneration } from "./transformers-engine-continuous";
import {
  generateSinglePreparedRequest,
  generateTransformersBatch,
  prepareGenerationRequest,
} from "./transformers-engine-generation";
import { emitGenerationRouteDecision } from "./transformers-engine-routing";
import {
  compileMessagePrompt,
  createPrefillProgressReporter,
  emitGenerationProgress,
  enforceGenerationMemoryBudget,
  enforcePromptTokenLimit,
  enforceTotalTokenLimit,
  generationOptions,
  promptHasOpenThinking,
  promptTokenCount,
  promptTokenIds,
  THINK_OPEN,
} from "./transformers-engine-shared";
import type {
  GenerationEngine,
  GenerationStreamEvent,
  NormalizedGenerationRequest,
  NormalizedGenerationResult,
  ServeEvent,
} from "./types";

export type TransformersGenerationEngineOptions = {
  model: CausalLM;
  tokenizer: Tokenizer;
  interactionProfile?: InteractionProfile;
  maxPromptTokens?: number;
  maxTotalTokens?: number;
  maxBatchSize?: number;
  batchWindowMs?: number;
  maxConcurrentRequests?: number;
  gpuMemoryUtilization?: number;
  onEvent?: (event: ServeEvent) => void;
};

const PROGRESS_TOKEN_INTERVAL = 64;
const STREAM_DECODE_INTERVAL = 8;

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

function streamDecodeInterval(stop: readonly string[] | undefined): number {
  return stop === undefined || stop.length === 0 ? STREAM_DECODE_INTERVAL : 1;
}

function streamUsage(promptTokens: number, completionTokens: number) {
  return {
    promptTokens,
    completionTokens,
    totalTokens: promptTokens + completionTokens,
  };
}

function shouldEmitProgress(
  request: NormalizedGenerationRequest,
  completionTokens: number,
): boolean {
  return (
    completionTokens % PROGRESS_TOKEN_INTERVAL === 0 ||
    completionTokens === request.sampling.maxTokens
  );
}

function streamTokenEventsForRequest(
  request: NormalizedGenerationRequest,
  options: TransformersGenerationEngineOptions,
  prompt: { tokenIds: readonly number[] } | null,
  onPrefillProgress: Parameters<typeof generationOptions>[1],
) {
  if (request.input.kind === "text") {
    return generateTokenEvents(
      options.model,
      options.tokenizer.encode(request.input.text, { addSpecialTokens: true }),
      {
        ...generationOptions(request, onPrefillProgress),
        ...(request.sampling.ignoreEos === true || options.tokenizer.eosTokenIds.length === 0
          ? {}
          : { eosTokenIds: [...options.tokenizer.eosTokenIds] }),
      },
    );
  }

  return generatePreparedTokenEvents(
    options.model,
    { tokenIds: promptTokenIds(request, prompt) },
    generationOptions(request, onPrefillProgress),
  );
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

type StreamingDecodeState = {
  generatedTokenIds: number[];
  decodedText: string;
  rawPrefix: string;
};

function createStreamingDecodeState(prompt: { text: string } | null): StreamingDecodeState {
  return {
    generatedTokenIds: [],
    decodedText: "",
    rawPrefix: promptHasOpenThinking(prompt) ? THINK_OPEN : "",
  };
}

function handleStreamingTokenEvent(
  request: NormalizedGenerationRequest,
  options: TransformersGenerationEngineOptions,
  promptTokens: number,
  decodeInterval: number,
  state: StreamingDecodeState,
  event: Extract<TokenGenerationEvent, { type: "token" }>,
): string | undefined {
  state.generatedTokenIds.push(event.tokenId);
  if (shouldEmitProgress(request, event.completionTokens)) {
    emitGenerationProgress(options, request, promptTokens, event.completionTokens);
  }
  if (event.completionTokens % decodeInterval !== 0) {
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

function handleStreamingDoneEvent(
  options: TransformersGenerationEngineOptions,
  promptTokens: number,
  state: StreamingDecodeState,
  event: Extract<TokenGenerationEvent, { type: "done" }>,
): { text?: string; done: Extract<GenerationStreamEvent, { type: "done" }> } {
  state.generatedTokenIds = [...event.tokenIds];
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
      finishReason: event.finishReason === "eos" ? "eos" : "length",
      usage: streamUsage(promptTokens, state.generatedTokenIds.length),
    },
  };
}

/** Create a text-generation engine from an already loaded CausalLM and tokenizer. */
export function createTransformersGenerationEngine(
  options: TransformersGenerationEngineOptions,
): GenerationEngine {
  const lane = new ModelExecutionLane(options.maxConcurrentRequests ?? 1);
  const continuous = createContinuousTransformersGeneration(options, lane);

  function generate(
    request: NormalizedGenerationRequest,
  ): NormalizedGenerationResult | Promise<NormalizedGenerationResult> {
    const scheduled = continuous.generate(request);
    if (scheduled !== null) {
      return scheduled;
    }
    const prepared = prepareGenerationRequest(request, options);
    return lane.run(() => generateSinglePreparedRequest(prepared, options), request.abortSignal);
  }

  return {
    generate,
    generateBatch(requests) {
      if ((options.maxBatchSize ?? 1) > 1) {
        return Promise.all(requests.map((request) => generate(request)));
      }
      return generateTransformersBatch(requests, options);
    },
    async *stream(request) {
      const release = await lane.acquire(request.abortSignal);
      try {
        emitGenerationRouteDecision(options, request, "single", false, "streaming");
        const prompt = compileMessagePrompt(request, options);
        const promptTokens = promptTokenCount(request, options, prompt);
        enforcePromptTokenLimit(options, request, promptTokens);
        enforceTotalTokenLimit(options, request, promptTokens);
        enforceGenerationMemoryBudget(options, request, promptTokens);
        emitGenerationProgress(options, request, promptTokens, 0);
        const onPrefillProgress = createPrefillProgressReporter(options, request, promptTokens);
        const decodeInterval = streamDecodeInterval(request.sampling.stop);
        const tokenEvents = streamTokenEventsForRequest(
          request,
          options,
          prompt,
          onPrefillProgress,
        );
        const state = createStreamingDecodeState(prompt);

        for await (const event of tokenEvents) {
          if (event.type === "token") {
            const text = handleStreamingTokenEvent(
              request,
              options,
              promptTokens,
              decodeInterval,
              state,
              event,
            );
            if (text !== undefined) {
              yield { type: "text", text };
            }
            continue;
          }

          const finished = handleStreamingDoneEvent(options, promptTokens, state, event);
          if (finished.text !== undefined) {
            yield { type: "text", text: finished.text };
          }
          yield finished.done;
          return;
        }
      } finally {
        release();
      }
    },
  };
}
