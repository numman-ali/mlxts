/**
 * Media-content prompt preparation for transformer-backed serving.
 * @module
 */

import type { Tokenizer } from "@mlxts/tokenizers";
import {
  type CausalLM,
  type ChatMessage,
  type DecodedQwen3_5Image,
  GenerationAbortError,
  type InteractionProfile,
  type PreparedPrompt,
  prepareQwen3_5ImageBatch,
  prepareQwen3_5ImagePrompt,
  QWEN3_5_IMAGE_MARKER,
  type Qwen3_5VisionPreprocessorConfig,
  smartResizeQwen3_5Image,
} from "@mlxts/transformers";
import { ServeError } from "./errors";
import {
  decodeResizedImageBytes,
  type ImageReadOptions,
  readImageBytesSize,
  readImageSourceBytes,
} from "./media-image";
import type { TransformersGenerationEngineOptions } from "./transformers-engine";
import {
  batchGenerationOptions,
  type PreparedGenerationRequest,
} from "./transformers-engine-generation";
import type { PromptPrefixCacheIdentity } from "./transformers-engine-prefix-cache";
import {
  type CompiledPrompt,
  enforceGenerationMemoryBudget,
  enforcePromptTokenLimit,
  enforceTotalTokenLimit,
} from "./transformers-engine-shared";
import type {
  GenerationContentMessage,
  GenerationContentPart,
  NormalizedGenerationRequest,
} from "./types";

/** Host-loaded media prompt that can prepare model tensors inside the model lane. */
export type LoadedContentPrompt = {
  prompt: CompiledPrompt;
  promptCacheIdentity?: PromptPrefixCacheIdentity;
  preparePrompt(
    context: TransformersContentAdapterModelContext,
  ): PreparedPrompt | Promise<PreparedPrompt>;
};

/** Tokenizer and request state available during host-side media loading. */
export type TransformersContentAdapterLoadContext = {
  tokenizer: Tokenizer;
  interactionProfile?: InteractionProfile;
  signal?: AbortSignal;
};

/** Model state available only while the model execution lane is held. */
export type TransformersContentAdapterModelContext = {
  model: CausalLM;
};

/** Model-family hook that converts protocol-neutral media content into decoder inputs. */
export type TransformersContentAdapter = {
  /** Load host media and create the request-local model-preparation closure. */
  load(
    request: NormalizedGenerationRequest,
    context: TransformersContentAdapterLoadContext,
  ): Promise<LoadedContentPrompt>;
};

type LoadedContentGenerationRequest = LoadedContentPrompt & {
  request: NormalizedGenerationRequest;
  startedAt: number;
};

type DecodedQwenImage = {
  image: DecodedQwen3_5Image;
  cacheKey: string;
};

function rejectUnsupportedContent(message: string): never {
  throw new ServeError(message, { code: "unsupported_input", param: "messages" });
}

function throwIfRequestAborted(request: NormalizedGenerationRequest, context: string): void {
  if (request.abortSignal?.aborted === true) {
    throw new GenerationAbortError(`${context}: generation was cancelled.`);
  }
}

function textForQwenPart(part: GenerationContentPart): string {
  switch (part.kind) {
    case "text":
      return part.text;
    case "image":
      return QWEN3_5_IMAGE_MARKER;
    case "audio":
      return rejectUnsupportedContent("Qwen image serving does not support audio inputs yet.");
    case "file":
      return rejectUnsupportedContent("Qwen image serving does not support file inputs yet.");
  }
}

function textForQwenContent(parts: readonly GenerationContentPart[]): string {
  return parts.map(textForQwenPart).join("");
}

function chatMessageFromContent(message: GenerationContentMessage): ChatMessage {
  const content = textForQwenContent(message.content);
  switch (message.role) {
    case "system":
    case "user":
      return { role: message.role, content };
    case "assistant":
      return {
        role: "assistant",
        content,
        ...(message.reasoning_content === undefined
          ? {}
          : { reasoning_content: message.reasoning_content }),
        ...(message.tool_calls === undefined ? {} : { tool_calls: message.tool_calls }),
      };
    case "tool":
      return {
        role: "tool",
        content,
        ...(message.name === undefined ? {} : { name: message.name }),
        ...(message.tool_call_id === undefined ? {} : { tool_call_id: message.tool_call_id }),
      };
  }
}

function imageParts(messages: readonly GenerationContentMessage[]): GenerationContentPart[] {
  const images: GenerationContentPart[] = [];
  for (const message of messages) {
    if (message.role === "system" && message.content.some((part) => part.kind === "image")) {
      rejectUnsupportedContent("Qwen image serving does not allow images in system messages.");
    }
    for (const part of message.content) {
      if (part.kind === "image") {
        images.push(part);
      }
    }
  }
  return images;
}

function compileQwenContentPrompt(
  request: NormalizedGenerationRequest,
  context: TransformersContentAdapterLoadContext,
): CompiledPrompt {
  if (request.input.kind !== "content") {
    throw new Error("compileQwenContentPrompt requires content input.");
  }
  const messages = request.input.messages.map(chatMessageFromContent);
  if (context.interactionProfile?.kind === "chat") {
    return context.interactionProfile.compileMessages(context.tokenizer, messages, {
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
  rejectUnsupportedContent("Qwen image serving requires a chat interaction profile.");
}

function emitPromptPrepare(
  request: NormalizedGenerationRequest,
  options: TransformersGenerationEngineOptions,
  phase: "start" | "complete",
  promptTokens?: number,
  durationMs?: number,
): void {
  if (phase === "start") {
    options.onEvent?.({
      type: "generation_prompt_prepare",
      phase: "start",
      id: request.id,
      protocol: request.protocol,
      model: request.model,
      inputKind: request.input.kind,
    });
    return;
  }
  if (promptTokens === undefined || durationMs === undefined) {
    throw new Error("emitPromptPrepare complete events require token count and duration.");
  }
  options.onEvent?.({
    type: "generation_prompt_prepare",
    phase: "complete",
    id: request.id,
    protocol: request.protocol,
    model: request.model,
    inputKind: request.input.kind,
    promptTokens,
    durationMs,
  });
}

function imageReadOptions(signal: AbortSignal | undefined): ImageReadOptions {
  return signal === undefined ? {} : { signal };
}

function hexDigest(bytes: Uint8Array): string {
  return [...bytes].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

function arrayBufferFromBytes(bytes: Uint8Array): ArrayBuffer {
  const buffer = new ArrayBuffer(bytes.byteLength);
  new Uint8Array(buffer).set(bytes);
  return buffer;
}

async function sha256Hex(bytes: Uint8Array): Promise<string> {
  return hexDigest(
    new Uint8Array(await crypto.subtle.digest("SHA-256", arrayBufferFromBytes(bytes))),
  );
}

function preprocessorCacheKey(preprocessor: Qwen3_5VisionPreprocessorConfig): string {
  return JSON.stringify({
    size: preprocessor.size,
    patchSize: preprocessor.patchSize,
    temporalPatchSize: preprocessor.temporalPatchSize,
    mergeSize: preprocessor.mergeSize,
    imageMean: preprocessor.imageMean,
    imageStd: preprocessor.imageStd,
    processorClass: preprocessor.processorClass,
    imageProcessorType: preprocessor.imageProcessorType,
  });
}

async function decodeQwenImage(
  part: GenerationContentPart,
  preprocessor: Qwen3_5VisionPreprocessorConfig,
  signal: AbortSignal | undefined,
): Promise<DecodedQwenImage> {
  if (part.kind !== "image") {
    throw new Error("decodeQwenImage requires an image part.");
  }
  const readOptions = imageReadOptions(signal);
  const bytes = await readImageSourceBytes(part.source, readOptions);
  const originalSize = await readImageBytesSize(bytes, readOptions);
  const resizedSize = smartResizeQwen3_5Image(
    originalSize.height,
    originalSize.width,
    preprocessor,
  );
  const image = await decodeResizedImageBytes(bytes, resizedSize, readOptions);
  return {
    image,
    cacheKey: JSON.stringify({
      kind: "qwen-image",
      digest: await sha256Hex(bytes),
      originalSize,
      resizedSize,
      preprocessor: preprocessorCacheKey(preprocessor),
    }),
  };
}

/** Create the first Qwen 3.5/3.6 image-content adapter used by local serving. */
export function createQwen3_5ImageContentAdapter(
  preprocessor: Qwen3_5VisionPreprocessorConfig,
): TransformersContentAdapter {
  return {
    async load(request, context) {
      if (request.input.kind !== "content") {
        throw new Error("Qwen image content adapter requires content input.");
      }
      const images = imageParts(request.input.messages);
      if (images.length === 0) {
        rejectUnsupportedContent("Qwen image content adapter received no image parts.");
      }
      const prompt = compileQwenContentPrompt(request, context);
      const decodedImages: DecodedQwen3_5Image[] = [];
      const contentKeys: string[] = [];
      for (const image of images) {
        const decoded = await decodeQwenImage(image, preprocessor, context.signal);
        decodedImages.push(decoded.image);
        contentKeys.push(decoded.cacheKey);
      }

      return {
        prompt,
        promptCacheIdentity: { contentKeys },
        preparePrompt(modelContext) {
          const preparedImages = prepareQwen3_5ImageBatch(decodedImages, preprocessor);
          try {
            return prepareQwen3_5ImagePrompt(
              modelContext.model,
              prompt.tokenIds,
              preparedImages.pixelValues,
              preparedImages.imageGridThw,
            );
          } finally {
            preparedImages.pixelValues.free();
            preparedImages.imageGridThw.free();
          }
        },
      };
    },
  };
}

function disposePreparedPrompt(prompt: PreparedPrompt): void {
  prompt.inputEmbeddings?.free();
  prompt.positionIds?.free();
}

/** Load host media and compile prompt text before entering the model execution lane. */
export async function loadContentGenerationRequest(
  request: NormalizedGenerationRequest,
  options: TransformersGenerationEngineOptions,
): Promise<LoadedContentGenerationRequest> {
  if (request.input.kind !== "content") {
    throw new Error("loadContentGenerationRequest requires content input.");
  }
  if (options.contentAdapter === undefined) {
    rejectUnsupportedContent(
      "This loaded model does not expose a media-content adapter for serving.",
    );
  }

  throwIfRequestAborted(request, "loadContentGenerationRequest");
  const startedAt = performance.now();
  emitPromptPrepare(request, options, "start");
  const loaded = await options.contentAdapter.load(request, {
    tokenizer: options.tokenizer,
    ...(options.interactionProfile === undefined
      ? {}
      : { interactionProfile: options.interactionProfile }),
    ...(request.abortSignal === undefined ? {} : { signal: request.abortSignal }),
  });
  throwIfRequestAborted(request, "loadContentGenerationRequest");
  return { request, ...loaded, startedAt };
}

/** Prepare model-owned tensors for a loaded media-content request inside the model lane. */
export async function prepareLoadedContentGenerationRequest(
  loaded: LoadedContentGenerationRequest,
  options: TransformersGenerationEngineOptions,
): Promise<PreparedGenerationRequest> {
  throwIfRequestAborted(loaded.request, "prepareLoadedContentGenerationRequest");
  const preparedPrompt = await loaded.preparePrompt({ model: options.model });
  const promptTokens = preparedPrompt.tokenIds.length;
  try {
    throwIfRequestAborted(loaded.request, "prepareLoadedContentGenerationRequest");
    enforcePromptTokenLimit(options, loaded.request, promptTokens);
    enforceTotalTokenLimit(options, loaded.request, promptTokens);
    enforceGenerationMemoryBudget(options, loaded.request, promptTokens);
    emitPromptPrepare(
      loaded.request,
      options,
      "complete",
      promptTokens,
      performance.now() - loaded.startedAt,
    );
    return {
      request: loaded.request,
      prompt: loaded.prompt,
      promptTokens,
      tokenIds: preparedPrompt.tokenIds,
      preparedPrompt,
      ...(loaded.promptCacheIdentity === undefined
        ? {}
        : { promptCacheIdentity: loaded.promptCacheIdentity }),
      batchOptions: batchGenerationOptions(loaded.request, options),
    };
  } catch (error) {
    disposePreparedPrompt(preparedPrompt);
    throw error;
  }
}
