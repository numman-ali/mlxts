/**
 * Adapter from @mlxts/transformers generation into the serving engine contract.
 * @module
 */

import type { Tokenizer } from "@mlxts/tokenizers";
import {
  type CausalLM,
  type GenerationOptions,
  generateTextStream,
  generateTokens,
  type InteractionProfile,
} from "@mlxts/transformers";
import { ServeError } from "./errors";
import { readGenerationMemoryUsage } from "./memory-telemetry";
import type {
  GenerationEngine,
  NormalizedFinishReason,
  NormalizedGenerationRequest,
  NormalizedGenerationResult,
  ServeEvent,
} from "./types";

export type TransformersGenerationEngineOptions = {
  model: CausalLM;
  tokenizer: Tokenizer;
  interactionProfile?: InteractionProfile;
  maxTotalTokens?: number;
  onEvent?: (event: ServeEvent) => void;
};

const THINK_OPEN = "<think>";
const THINK_CLOSE = "</think>";
const PROGRESS_TOKEN_INTERVAL = 64;

function generationOptions(request: NormalizedGenerationRequest): GenerationOptions {
  return {
    maxTokens: request.sampling.maxTokens,
    ...(request.sampling.temperature === undefined
      ? {}
      : { temperature: request.sampling.temperature }),
    ...(request.sampling.topP === undefined ? {} : { topP: request.sampling.topP }),
    ...(request.sampling.topK === undefined ? {} : { topK: request.sampling.topK }),
    ...(request.sampling.seed === undefined ? {} : { seed: request.sampling.seed }),
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function positiveIntegerField(record: Record<string, unknown>, key: string): number | undefined {
  const value = record[key];
  return Number.isInteger(value) && typeof value === "number" && value > 0 ? value : undefined;
}

function modelContextLimit(options: TransformersGenerationEngineOptions): number | undefined {
  return (
    positiveIntegerField(options.model.config.rawConfig, "max_position_embeddings") ??
    (isRecord(options.model.config.rawConfig.text_config)
      ? positiveIntegerField(options.model.config.rawConfig.text_config, "max_position_embeddings")
      : undefined)
  );
}

function enforceTotalTokenLimit(
  options: TransformersGenerationEngineOptions,
  request: NormalizedGenerationRequest,
  promptTokens: number,
): void {
  const configuredLimit = options.maxTotalTokens;
  const contextLimit = modelContextLimit(options);
  const limits = [configuredLimit, contextLimit].filter(
    (limit): limit is number => limit !== undefined,
  );
  const limit = limits.length === 0 ? undefined : Math.min(...limits);
  if (limit === undefined) {
    return;
  }

  const requestedTotal = promptTokens + request.sampling.maxTokens;
  if (requestedTotal <= limit) {
    return;
  }

  throw new ServeError(
    `Requested prompt_tokens ${promptTokens} plus max_tokens ${request.sampling.maxTokens} exceeds this server's total token limit of ${limit}.`,
    {
      code: "context_length_exceeded",
      param: "max_tokens",
    },
  );
}

function emitGenerationProgress(
  options: TransformersGenerationEngineOptions,
  request: NormalizedGenerationRequest,
  promptTokens: number,
  completionTokens: number,
): void {
  const memory = readGenerationMemoryUsage();
  options.onEvent?.({
    type: "generation_progress",
    id: request.id,
    protocol: request.protocol,
    model: request.model,
    promptTokens,
    completionTokens,
    maxTokens: request.sampling.maxTokens,
    ...(memory === undefined ? {} : { memory }),
  });
}

function createProgressReporter(
  options: TransformersGenerationEngineOptions,
  request: NormalizedGenerationRequest,
  promptTokens: number,
): (tokenId: number, generatedTokenIds: readonly number[]) => void {
  return (_tokenId, generatedTokenIds) => {
    const completionTokens = generatedTokenIds.length;
    if (
      completionTokens % PROGRESS_TOKEN_INTERVAL === 0 ||
      completionTokens === request.sampling.maxTokens
    ) {
      emitGenerationProgress(options, request, promptTokens, completionTokens);
    }
  };
}

function applyStopSequences(
  text: string,
  stop: readonly string[] | undefined,
): { text: string; stopped: boolean } {
  if (stop === undefined || stop.length === 0) {
    return { text, stopped: false };
  }

  const matches = stop
    .map((sequence) => text.indexOf(sequence))
    .filter((index) => index >= 0)
    .sort((left, right) => left - right);

  const firstMatch = matches[0];
  return firstMatch === undefined
    ? { text, stopped: false }
    : { text: text.slice(0, firstMatch), stopped: true };
}

function finishReason(generatedReason: "length" | "eos", stopped: boolean): NormalizedFinishReason {
  if (stopped) {
    return "stop";
  }
  return generatedReason === "eos" ? "eos" : "length";
}

function promptHasOpenThinking(prompt: { text: string } | null): boolean {
  if (prompt === null) {
    return false;
  }
  const lastOpen = prompt.text.lastIndexOf(THINK_OPEN);
  const lastClose = prompt.text.lastIndexOf(THINK_CLOSE);
  return lastOpen >= 0 && lastOpen > lastClose && prompt.text.slice(lastOpen).trim() === THINK_OPEN;
}

function cleanReasoningContent(text: string): string {
  return text.replace(/<tool_call>[\s\S]*?<\/tool_call>/g, "").trim();
}

function reasoningResult(text: string, reasoning: string) {
  const reasoningContent = cleanReasoningContent(reasoning);
  return reasoningContent === "" ? { text } : { text, reasoningContent };
}

function splitPromptOpenReasoning(prompt: { text: string } | null, text: string) {
  if (!promptHasOpenThinking(prompt)) {
    return { text };
  }

  const generated = text.startsWith(THINK_OPEN) ? text.slice(THINK_OPEN.length) : text;
  const closeIndex = generated.indexOf(THINK_CLOSE);
  if (closeIndex < 0) {
    return reasoningResult("", generated);
  }

  return reasoningResult(
    generated.slice(closeIndex + THINK_CLOSE.length).trimStart(),
    generated.slice(0, closeIndex),
  );
}

function compileMessagePrompt(
  request: NormalizedGenerationRequest,
  options: TransformersGenerationEngineOptions,
): { text: string; tokenIds: number[] } | null {
  if (request.input.kind !== "messages") {
    return null;
  }
  if (options.interactionProfile === undefined) {
    throw new ServeError(
      "The transformers generation engine requires an interaction profile for message input.",
      { code: "unsupported_input" },
    );
  }
  return options.interactionProfile.compileMessages(options.tokenizer, request.input.messages, {
    addGenerationPrompt: true,
    ...(request.input.tools === undefined ? {} : { tools: request.input.tools }),
    ...(request.input.chatTemplate?.enableThinking === undefined
      ? {}
      : { enableThinking: request.input.chatTemplate.enableThinking }),
    ...(request.input.chatTemplate?.preserveThinking === undefined
      ? {}
      : { preserveThinking: request.input.chatTemplate.preserveThinking }),
  });
}

function promptTokenIds(
  request: NormalizedGenerationRequest,
  prompt: { tokenIds: readonly number[] } | null,
): readonly number[] {
  if (request.input.kind === "tokens") {
    return request.input.tokenIds;
  }
  if (prompt !== null) {
    return prompt.tokenIds;
  }
  throw new ServeError("The transformers generation engine could not compile message input.", {
    code: "unsupported_input",
  });
}

function generateTokenPrompt(
  request: NormalizedGenerationRequest,
  options: TransformersGenerationEngineOptions,
  prompt: { tokenIds: readonly number[] } | null,
  onToken?: (tokenId: number, generatedTokenIds: readonly number[]) => void,
) {
  const generated = generateTokens(
    options.model,
    promptTokenIds(request, prompt),
    generationOptions(request),
    onToken,
  );
  return {
    ...generated,
    text: options.tokenizer.decode(generated.tokenIds, { skipSpecialTokens: true }),
  };
}

function promptTokenCount(
  request: NormalizedGenerationRequest,
  options: TransformersGenerationEngineOptions,
  prompt: { tokenIds: readonly number[] } | null,
): number {
  if (request.input.kind === "text") {
    return options.tokenizer.encode(request.input.text).length;
  }
  if (request.input.kind === "tokens") {
    return request.input.tokenIds.length;
  }
  return prompt?.tokenIds.length ?? 0;
}

/** Create a simple text-generation engine from an already loaded CausalLM and tokenizer. */
export function createTransformersGenerationEngine(
  options: TransformersGenerationEngineOptions,
): GenerationEngine {
  return {
    generate(request): NormalizedGenerationResult {
      const prompt = compileMessagePrompt(request, options);
      const promptTokens = promptTokenCount(request, options, prompt);
      enforceTotalTokenLimit(options, request, promptTokens);
      const onToken = createProgressReporter(options, request, promptTokens);
      emitGenerationProgress(options, request, promptTokens, 0);
      const result =
        request.input.kind === "text"
          ? generateTextStream(
              options.model,
              options.tokenizer,
              request.input.text,
              generationOptions(request),
              () => undefined,
              onToken,
            )
          : generateTokenPrompt(request, options, prompt, onToken);
      const reasoning = splitPromptOpenReasoning(prompt, result.text);
      const stopped = applyStopSequences(reasoning.text, request.sampling.stop);

      return {
        text: stopped.text,
        ...(reasoning.reasoningContent === undefined
          ? {}
          : { reasoningContent: reasoning.reasoningContent }),
        finishReason: finishReason(result.finishReason, stopped.stopped),
        tokenIds: result.tokenIds,
        usage: {
          promptTokens,
          completionTokens: result.tokenIds.length,
          totalTokens: promptTokens + result.tokenIds.length,
        },
      };
    },
  };
}
