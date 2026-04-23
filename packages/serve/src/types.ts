/**
 * Protocol-neutral generation contracts for serving adapters.
 * @module
 */

import type { ChatMessage, ChatTool } from "@mlxts/transformers";

export type GenerationInput =
  | {
      kind: "text";
      text: string;
    }
  | {
      kind: "tokens";
      tokenIds: readonly number[];
    }
  | {
      kind: "messages";
      messages: readonly ChatMessage[];
      tools?: readonly ChatTool[];
      chatTemplate?: {
        enableThinking?: boolean;
        preserveThinking?: boolean;
      };
    };

export type GenerationSamplingOptions = {
  maxTokens: number;
  temperature?: number;
  topP?: number;
  topK?: number;
  seed?: number;
  stop?: readonly string[];
};

export type GenerationProtocol =
  | "openai.completions"
  | "openai.chat_completions"
  | "openai.responses"
  | "anthropic.messages";

export type NormalizedGenerationRequest = {
  id: string;
  model: string;
  input: GenerationInput;
  sampling: GenerationSamplingOptions;
  stream: boolean;
  protocol: GenerationProtocol;
  metadata?: Record<string, unknown>;
};

export type GenerationUsage = {
  promptTokens?: number;
  completionTokens?: number;
  totalTokens?: number;
};

export type GenerationMemoryUsage = {
  activeBytes: number;
  cacheBytes: number;
  peakBytes: number;
  limitBytes: number;
};

export type NormalizedFinishReason = "stop" | "length" | "eos" | "cancelled" | "error";

export type NormalizedGenerationResult = {
  text: string;
  reasoningContent?: string;
  finishReason: NormalizedFinishReason;
  tokenIds?: readonly number[];
  usage?: GenerationUsage;
};

export type GenerationStreamEvent =
  | {
      type: "text";
      text: string;
    }
  | {
      type: "done";
      finishReason: NormalizedFinishReason;
      usage?: GenerationUsage;
    };

export type ServeEvent =
  | {
      type: "request_start";
      method: string;
      path: string;
    }
  | {
      type: "request_complete";
      method: string;
      path: string;
      status: number;
      durationMs: number;
    }
  | {
      type: "request_error";
      method: string;
      path: string;
      message: string;
      code: string;
      status: number;
      durationMs: number;
    }
  | {
      type: "generation_start";
      id: string;
      protocol: GenerationProtocol;
      model: string;
      inputKind: GenerationInput["kind"];
      maxTokens: number;
    }
  | {
      type: "generation_progress";
      id: string;
      protocol: GenerationProtocol;
      model: string;
      promptTokens: number;
      completionTokens: number;
      maxTokens: number;
      memory?: GenerationMemoryUsage;
    }
  | {
      type: "generation_complete";
      id: string;
      protocol: GenerationProtocol;
      model: string;
      finishReason: NormalizedFinishReason;
      promptTokens?: number;
      completionTokens?: number;
      totalTokens?: number;
      durationMs: number;
      memory?: GenerationMemoryUsage;
    };

export type GenerationEngine = {
  generate(
    request: NormalizedGenerationRequest,
  ): NormalizedGenerationResult | Promise<NormalizedGenerationResult>;
  generateBatch?(
    requests: readonly NormalizedGenerationRequest[],
  ): readonly NormalizedGenerationResult[] | Promise<readonly NormalizedGenerationResult[]>;
  stream?(
    request: NormalizedGenerationRequest,
  ): AsyncIterable<GenerationStreamEvent> | Promise<AsyncIterable<GenerationStreamEvent>>;
};
