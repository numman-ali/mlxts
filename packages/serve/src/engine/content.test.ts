import { describe, expect, test } from "bun:test";

import { array, type MxArray, type ParameterTree } from "@mlxts/core";
import type { Tokenizer } from "@mlxts/tokenizers";
import {
  type BaseModelConfig,
  type CacheLayerKind,
  type CausalLM,
  type ChatMessage,
  type ForwardOptions,
  type InteractionProfile,
  KVCache,
  type Qwen3_5VisionPreprocessorConfig,
  type TransformerCache,
  type TransformerCacheForkOptions,
  type TransformerCacheSnapshot,
} from "@mlxts/transformers";
import { Qwen3_5ForConditionalGeneration } from "../../../transformers/src/families/qwen3_5/multimodal/conditional";
import type {
  Qwen3_5Config,
  Qwen3_5TextConfig,
  Qwen3_5VisionConfig,
} from "../../../transformers/src/families/qwen3_5/types";
import { DecodedImageCache } from "../media/decoded-image-cache";
import type { NormalizedGenerationRequest } from "../types";
import {
  createQwen3_5ImageContentAdapter,
  loadContentGenerationRequest,
  prepareLoadedContentGenerationRequest,
} from "./content";
import { PromptPrefixCache } from "./prefix-cache";

class TinyTokenizer implements Tokenizer {
  readonly vocabSize = 8;
  readonly bosTokenId: number | undefined = undefined;
  readonly eosTokenIds: number[] = [];
  readonly padTokenId: number | undefined = undefined;

  encode(text: string): number[] {
    return [...text].map(() => 0);
  }

  encodeWithOffsets(text: string) {
    return { ids: this.encode(text) };
  }

  encodeBatch(texts: string[]) {
    return texts.map((text) => this.encodeWithOffsets(text));
  }

  decode(tokenIds: number[]): string {
    return tokenIds.map((tokenId) => String.fromCharCode(97 + tokenId)).join("");
  }

  decodeBatch(batch: number[][]): string[] {
    return batch.map((tokenIds) => this.decode(tokenIds));
  }
}

class QwenImageTokenizer extends TinyTokenizer {
  override encode(text: string): number[] {
    return text.includes("<|image_pad|>") ? [7, 28, 9] : super.encode(text);
  }
}

class TinyModel implements CausalLM {
  readonly family = "llama" as const;
  readonly layerCount = 1;
  readonly config: BaseModelConfig = {
    family: "llama",
    modelType: "tiny",
    rawConfig: {},
    vocabSize: 8,
    hiddenSize: 1,
    numHiddenLayers: 1,
  };

  forward(inputIds: MxArray, options?: ForwardOptions): MxArray {
    options?.cache?.advance(inputIds.shape[1] ?? 0);
    return array([[[0, 1, 0, 0, 0, 0, 0, 0]]], "float32");
  }

  createCache(): TransformerCache {
    return new KVCache(this.layerCount);
  }

  parameters(): ParameterTree {
    return {};
  }

  trainableParameters(): ParameterTree {
    return {};
  }

  update(_params: ParameterTree): void {}

  freeze(): this {
    return this;
  }

  unfreeze(): this {
    return this;
  }

  eval(): this {
    return this;
  }

  train(): this {
    return this;
  }

  [Symbol.dispose](): void {}
}

class FakeCache implements TransformerCache {
  readonly layerCount = 0;
  readonly layerKinds: readonly CacheLayerKind[] = ["full"];
  readonly offset: number;

  constructor(offset: number) {
    this.offset = offset;
  }

  updateAndFetch(): { keys: MxArray; values: MxArray } {
    throw new Error("FakeCache.updateAndFetch should not be called.");
  }

  advance(): void {
    throw new Error("FakeCache.advance should not be called.");
  }

  isEmpty(): boolean {
    return this.offset === 0;
  }

  isTrimmable(): boolean {
    return true;
  }

  snapshot(): TransformerCacheSnapshot {
    return new FakeSnapshot(this.offset);
  }

  arrays(): MxArray[] {
    return [];
  }

  [Symbol.dispose](): void {}
}

class FakeSnapshot implements TransformerCacheSnapshot {
  readonly layerKinds: readonly CacheLayerKind[] = ["full"];
  readonly trimmable = true;
  readonly estimatedByteSize: number;
  readonly offset: number;

  constructor(offset: number) {
    this.offset = offset;
    this.estimatedByteSize = offset * 4;
  }

  canFork(options: TransformerCacheForkOptions = {}): boolean {
    return (options.offset ?? this.offset) <= this.offset;
  }

  fork(options: TransformerCacheForkOptions = {}): TransformerCache {
    return new FakeCache(options.offset ?? this.offset);
  }

  [Symbol.dispose](): void {}
}

const qwenPreprocessor: Qwen3_5VisionPreprocessorConfig = {
  size: { shortestEdge: 1, longestEdge: 4 },
  patchSize: 1,
  temporalPatchSize: 1,
  mergeSize: 1,
  imageMean: [0, 0, 0],
  imageStd: [1, 1, 1],
  processorClass: "Qwen3VLProcessor",
  imageProcessorType: "Qwen2VLImageProcessorFast",
};

function qwenTextConfig(): Qwen3_5TextConfig {
  return {
    family: "qwen",
    modelType: "qwen3_5_text",
    rawConfig: {},
    vocabSize: 32,
    hiddenSize: 8,
    intermediateSize: 16,
    feedForwardKind: "dense",
    moeIntermediateSize: null,
    sharedExpertIntermediateSize: null,
    numExperts: null,
    numExpertsPerToken: null,
    routerAuxLossCoef: null,
    numHiddenLayers: 1,
    numAttentionHeads: 2,
    numKeyValueHeads: 1,
    headDim: 4,
    hiddenAct: "silu",
    maxPositionEmbeddings: 128,
    initializerRange: 0.02,
    rmsNormEps: 1e-6,
    useCache: true,
    tieWordEmbeddings: false,
    attentionBias: false,
    attentionDropout: 0,
    attnOutputGate: true,
    outputGateType: null,
    linearConvKernelDim: 2,
    linearKeyHeadDim: 2,
    linearValueHeadDim: 2,
    linearNumKeyHeads: 1,
    linearNumValueHeads: 2,
    layerTypes: ["full_attention"],
    fullAttentionInterval: 1,
    ropeParameters: {
      ropeType: "default",
      ropeTheta: 10000,
      partialRotaryFactor: 1,
      mropeSection: [1, 1, 0],
      mropeInterleaved: true,
    },
    partialRotaryFactor: 1,
    mtpNumHiddenLayers: 0,
    mtpUseDedicatedEmbeddings: false,
    mambaSsmDtype: null,
    bosTokenId: null,
    eosTokenId: null,
    padTokenId: null,
  };
}

function qwenVisionConfig(): Qwen3_5VisionConfig {
  return {
    family: "qwen",
    modelType: "qwen3_5",
    rawConfig: {},
    depth: 1,
    hiddenSize: 8,
    hiddenAct: "gelu_pytorch_tanh",
    intermediateSize: 16,
    numHeads: 2,
    inChannels: 3,
    patchSize: 1,
    spatialMergeSize: 1,
    temporalPatchSize: 1,
    outHiddenSize: 8,
    numPositionEmbeddings: 16,
    deepstackVisualIndexes: [],
    initializerRange: 0.02,
  };
}

function qwenConfig(): Qwen3_5Config {
  const textConfig = qwenTextConfig();
  const visionConfig = qwenVisionConfig();
  return {
    family: "qwen",
    modelType: "qwen3_5",
    rawConfig: {},
    vocabSize: textConfig.vocabSize,
    hiddenSize: textConfig.hiddenSize,
    numHiddenLayers: textConfig.numHiddenLayers,
    textConfig,
    visionConfig,
    imageTokenId: 28,
    videoTokenId: 29,
    visionStartTokenId: 26,
    visionEndTokenId: 27,
    tieWordEmbeddings: false,
    languageModelOnly: false,
  };
}

function uint16le(value: number): number[] {
  return [value & 0xff, (value >> 8) & 0xff];
}

function int32le(value: number): number[] {
  return [value & 0xff, (value >> 8) & 0xff, (value >> 16) & 0xff, (value >> 24) & 0xff];
}

function uint32le(value: number): number[] {
  return int32le(value >>> 0);
}

function bmpBytes(width: number, height: number, pixels: readonly number[]): Uint8Array {
  const bytesPerPixel = 3;
  const rowStride = Math.ceil((width * bytesPerPixel) / 4) * 4;
  const pixelArraySize = rowStride * height;
  const fileSize = 54 + pixelArraySize;
  const header = [
    0x42,
    0x4d,
    ...uint32le(fileSize),
    0,
    0,
    0,
    0,
    ...uint32le(54),
    ...uint32le(40),
    ...int32le(width),
    ...int32le(-height),
    ...uint16le(1),
    ...uint16le(24),
    ...uint32le(0),
    ...uint32le(pixelArraySize),
    ...uint32le(0),
    ...uint32le(0),
    ...uint32le(0),
    ...uint32le(0),
  ];

  const pixelBytes: number[] = [];
  for (let row = 0; row < height; row += 1) {
    for (let column = 0; column < width; column += 1) {
      const index = (row * width + column) * 3;
      const red = pixels[index];
      const green = pixels[index + 1];
      const blue = pixels[index + 2];
      if (red === undefined || green === undefined || blue === undefined) {
        throw new Error("bmpBytes: missing RGB pixel data.");
      }
      pixelBytes.push(blue, green, red);
    }
    while (pixelBytes.length % rowStride !== 0) {
      pixelBytes.push(0);
    }
  }

  return new Uint8Array([...header, ...pixelBytes]);
}

function imageDataUrl(): string {
  const binary = String.fromCharCode(...bmpBytes(1, 1, [255, 0, 0]));
  return btoa(binary);
}

function arrayBufferFromBytes(bytes: Uint8Array): ArrayBuffer {
  const buffer = new ArrayBuffer(bytes.byteLength);
  new Uint8Array(buffer).set(bytes);
  return buffer;
}

function contentRequest(input: NormalizedGenerationRequest["input"]): NormalizedGenerationRequest {
  return {
    id: "content",
    model: "qwen",
    input,
    sampling: { maxTokens: 1, temperature: 0 },
    stream: false,
    protocol: "openai.chat_completions",
  };
}

function chatProfile(seenMessages: ChatMessage[]): InteractionProfile {
  return {
    kind: "chat",
    chatTemplate: {
      template: "{{ messages }}",
      format(messages) {
        return messages.map((message) => `${message.role}:${message.content}`).join("\n");
      },
    },
    compileTextPrompt(tokenizer, prompt) {
      return { text: prompt, tokenIds: tokenizer.encode(prompt) };
    },
    compileMessages(tokenizer, messages, options = {}) {
      seenMessages.push(...messages);
      const suffix = options.enableThinking === false ? "\n/no-think" : "";
      const text = messages.map((message) => `${message.role}:${message.content}`).join("\n");
      return { text: `${text}${suffix}`, tokenIds: tokenizer.encode(`${text}${suffix}`) };
    },
  };
}

describe("transformers media-content preparation", () => {
  test("loads Qwen image content and defers tensor prep to the model lane", async () => {
    const seenMessages: ChatMessage[] = [];
    const adapter = createQwen3_5ImageContentAdapter(qwenPreprocessor);
    const request = contentRequest({
      kind: "content",
      messages: [
        {
          role: "user",
          content: [
            { kind: "text", text: "Describe: " },
            {
              kind: "image",
              source: { kind: "data", mediaType: "image/bmp", data: imageDataUrl() },
            },
          ],
        },
        {
          role: "assistant",
          content: [{ kind: "text", text: "Thinking" }],
          reasoning_content: "reason",
          tool_calls: [{ type: "function", function: { name: "note", arguments: "{}" } }],
        },
        {
          role: "tool",
          content: [{ kind: "text", text: "observation" }],
          name: "note",
          tool_call_id: "call-1",
        },
      ],
      chatTemplate: { enableThinking: false },
    });

    const loaded = await adapter.load(request, {
      tokenizer: new TinyTokenizer(),
      interactionProfile: chatProfile(seenMessages),
    });

    expect(loaded.prompt.text).toContain("<|vision_start|><|image_pad|><|vision_end|>");
    expect(loaded.prompt.text).toContain("/no-think");
    expect(loaded.promptCacheIdentity?.contentKeys).toHaveLength(1);
    expect(loaded.promptCacheIdentity?.contentKeys[0]).toContain("qwen-image");
    expect(seenMessages[1]).toMatchObject({
      role: "assistant",
      reasoning_content: "reason",
      tool_calls: [{ function: { name: "note" } }],
    });
    expect(seenMessages[2]).toMatchObject({
      role: "tool",
      name: "note",
      tool_call_id: "call-1",
    });
    await expect((async () => loaded.preparePrompt({ model: new TinyModel() }))()).rejects.toThrow(
      "expected a Qwen 3.5 conditional checkpoint",
    );
  });

  test("builds a Qwen image token plan before full tensor preparation", async () => {
    const adapter = createQwen3_5ImageContentAdapter(qwenPreprocessor);
    using model = new Qwen3_5ForConditionalGeneration(qwenConfig());
    const loaded = await adapter.load(
      contentRequest({
        kind: "content",
        messages: [
          {
            role: "user",
            content: [
              { kind: "text", text: "Describe: " },
              {
                kind: "image",
                source: { kind: "data", mediaType: "image/bmp", data: imageDataUrl() },
              },
            ],
          },
        ],
      }),
      {
        tokenizer: new QwenImageTokenizer(),
        interactionProfile: chatProfile([]),
      },
    );

    const plan = loaded.prepareTokenPlan?.({ model });

    expect(plan?.tokenIds).toEqual([7, 28, 9]);
    expect(plan?.canSkipPromptPreparation(1)).toBe(false);
    expect(plan?.canSkipPromptPreparation(2)).toBe(true);
  });

  test("reuses decoded Qwen image bytes by content digest and preprocessor", async () => {
    const decodedImageCache = new DecodedImageCache(1024);
    const adapter = createQwen3_5ImageContentAdapter(qwenPreprocessor, { decodedImageCache });
    const request = contentRequest({
      kind: "content",
      messages: [
        {
          role: "user",
          content: [
            { kind: "text", text: "Describe: " },
            {
              kind: "image",
              source: { kind: "data", mediaType: "image/bmp", data: imageDataUrl() },
            },
          ],
        },
      ],
    });
    const context = {
      tokenizer: new QwenImageTokenizer(),
      interactionProfile: chatProfile([]),
    };

    const first = await adapter.load(request, context);
    const second = await adapter.load(request, context);

    expect(decodedImageCache.entryCount).toBe(1);
    expect(decodedImageCache.missCount).toBe(1);
    expect(decodedImageCache.hitCount).toBe(1);
    expect(second.promptCacheIdentity?.contentKeys).toEqual(first.promptCacheIdentity?.contentKeys);
  });

  test("keeps Qwen decoded-image cache keys preprocessor-specific", async () => {
    const decodedImageCache = new DecodedImageCache(1024);
    const alternatePreprocessor = {
      ...qwenPreprocessor,
      size: { ...qwenPreprocessor.size, shortestEdge: qwenPreprocessor.size.shortestEdge + 1 },
    };
    const request = contentRequest({
      kind: "content",
      messages: [
        {
          role: "user",
          content: [
            {
              kind: "image",
              source: { kind: "data", mediaType: "image/bmp", data: imageDataUrl() },
            },
          ],
        },
      ],
    });
    const context = {
      tokenizer: new QwenImageTokenizer(),
      interactionProfile: chatProfile([]),
    };

    await createQwen3_5ImageContentAdapter(qwenPreprocessor, { decodedImageCache }).load(
      request,
      context,
    );
    await createQwen3_5ImageContentAdapter(alternatePreprocessor, { decodedImageCache }).load(
      request,
      context,
    );

    expect(decodedImageCache.entryCount).toBe(2);
    expect(decodedImageCache.missCount).toBe(2);
    expect(decodedImageCache.hitCount).toBe(0);
  });

  test("refetches remote image URLs while reusing identical decoded bytes", async () => {
    const decodedImageCache = new DecodedImageCache(1024);
    const adapter = createQwen3_5ImageContentAdapter(qwenPreprocessor, { decodedImageCache });
    const request = contentRequest({
      kind: "content",
      messages: [
        {
          role: "user",
          content: [
            {
              kind: "image",
              source: { kind: "url", url: "https://example.com/image.bmp" },
            },
          ],
        },
      ],
    });
    let fetchCalls = 0;
    const context = {
      tokenizer: new QwenImageTokenizer(),
      interactionProfile: chatProfile([]),
      remoteImageHosts: ["example.com"],
      remoteResolve: async () => [{ address: "93.184.216.34", family: 4 as const }],
      remoteFetch: async () => {
        fetchCalls += 1;
        const bytes = bmpBytes(1, 1, [255, 0, 0]);
        const body = arrayBufferFromBytes(bytes);
        return new Response(body, {
          headers: {
            "content-type": "image/bmp",
            "content-length": String(bytes.byteLength),
          },
        });
      },
    };

    await adapter.load(request, context);
    await adapter.load(request, context);

    expect(fetchCalls).toBe(2);
    expect(decodedImageCache.missCount).toBe(1);
    expect(decodedImageCache.hitCount).toBe(1);
  });

  test("does not cache rejected image payloads", async () => {
    const decodedImageCache = new DecodedImageCache(1024);
    const adapter = createQwen3_5ImageContentAdapter(qwenPreprocessor, { decodedImageCache });

    await expect(
      adapter.load(
        contentRequest({
          kind: "content",
          messages: [
            {
              role: "user",
              content: [
                {
                  kind: "image",
                  source: { kind: "data", mediaType: "text/plain", data: btoa("not image") },
                },
              ],
            },
          ],
        }),
        {
          tokenizer: new QwenImageTokenizer(),
          interactionProfile: chatProfile([]),
        },
      ),
    ).rejects.toThrow("image/*");

    expect(decodedImageCache.entryCount).toBe(0);
    expect(decodedImageCache.missCount).toBe(0);
    expect(decodedImageCache.hitCount).toBe(0);
  });

  test("rejects unsupported Qwen content shapes before tensor preparation", async () => {
    const adapter = createQwen3_5ImageContentAdapter(qwenPreprocessor);
    const context = { tokenizer: new TinyTokenizer(), interactionProfile: chatProfile([]) };
    await expect(
      adapter.load(contentRequest({ kind: "text", text: "hi" }), context),
    ).rejects.toThrow("requires content input");
    await expect(
      adapter.load(
        contentRequest({
          kind: "content",
          messages: [{ role: "user", content: [{ kind: "text", text: "no image" }] }],
        }),
        context,
      ),
    ).rejects.toThrow("received no image parts");
    await expect(
      adapter.load(
        contentRequest({
          kind: "content",
          messages: [
            {
              role: "system",
              content: [
                {
                  kind: "image",
                  source: { kind: "data", mediaType: "image/bmp", data: imageDataUrl() },
                },
              ],
            },
          ],
        }),
        context,
      ),
    ).rejects.toThrow("does not allow images in system messages");
    await expect(
      adapter.load(
        contentRequest({
          kind: "content",
          messages: [
            {
              role: "user",
              content: [
                {
                  kind: "image",
                  source: { kind: "data", mediaType: "image/bmp", data: imageDataUrl() },
                },
                { kind: "audio", source: { kind: "data", mediaType: "audio/wav", data: "AA==" } },
              ],
            },
          ],
        }),
        context,
      ),
    ).rejects.toThrow("does not support audio");
    await expect(
      adapter.load(
        contentRequest({
          kind: "content",
          messages: [
            {
              role: "user",
              content: [
                {
                  kind: "image",
                  source: { kind: "data", mediaType: "image/bmp", data: imageDataUrl() },
                },
                {
                  kind: "file",
                  source: { kind: "file", fileId: "file-1" },
                  filename: "report.pdf",
                },
              ],
            },
          ],
        }),
        context,
      ),
    ).rejects.toThrow("does not support file");
  });

  test("rejects Qwen content without a chat profile", async () => {
    const adapter = createQwen3_5ImageContentAdapter(qwenPreprocessor);

    await expect(
      adapter.load(
        contentRequest({
          kind: "content",
          messages: [
            {
              role: "user",
              content: [
                {
                  kind: "image",
                  source: { kind: "data", mediaType: "image/bmp", data: imageDataUrl() },
                },
              ],
            },
          ],
        }),
        { tokenizer: new TinyTokenizer() },
      ),
    ).rejects.toThrow("requires a chat interaction profile");
  });

  test("validates content generation request preparation and cleanup", async () => {
    const tokenizer = new TinyTokenizer();
    using model = new TinyModel();
    await expect(
      loadContentGenerationRequest(contentRequest({ kind: "text", text: "hi" }), {
        model,
        tokenizer,
        contentAdapter: {
          async load() {
            throw new Error("should not load");
          },
        },
      }),
    ).rejects.toThrow("requires content input");
    await expect(
      loadContentGenerationRequest(
        contentRequest({
          kind: "content",
          messages: [{ role: "user", content: [{ kind: "text", text: "hi" }] }],
        }),
        { model, tokenizer },
      ),
    ).rejects.toThrow("does not expose a media-content adapter");

    const inputEmbeddings = array([[[0], [1]]], "float32");
    await expect(
      prepareLoadedContentGenerationRequest(
        {
          request: contentRequest({
            kind: "content",
            messages: [{ role: "user", content: [{ kind: "text", text: "hi" }] }],
          }),
          startedAt: performance.now(),
          prompt: { text: "hi", tokenIds: [0, 1] },
          preparePrompt() {
            return { tokenIds: [0, 1], inputEmbeddings };
          },
        },
        { model, tokenizer, maxPromptTokens: 1 },
      ),
    ).rejects.toThrow("prompt token limit");
    expect(inputEmbeddings.isDisposed).toBe(true);
  });

  test("uses token plans to skip media prompt tensors when cache covers the image prefix", async () => {
    const tokenizer = new TinyTokenizer();
    using model = new TinyModel();
    using promptCache = new PromptPrefixCache(1);
    const promptCacheIdentity = { contentKeys: ["image:stable"] };
    expect(promptCache.store([1, 2, 3, 4], new FakeSnapshot(3), promptCacheIdentity)).toBe(3);

    let preparePromptCalls = 0;
    const prepared = await prepareLoadedContentGenerationRequest(
      {
        request: contentRequest({
          kind: "content",
          messages: [{ role: "user", content: [{ kind: "text", text: "hi" }] }],
        }),
        startedAt: performance.now(),
        prompt: { text: "hi", tokenIds: [1, 2, 3, 4] },
        promptCacheIdentity,
        prepareTokenPlan() {
          return {
            tokenIds: [1, 2, 3, 4],
            canSkipPromptPreparation(cachedPrefixTokens) {
              return cachedPrefixTokens >= 3;
            },
          };
        },
        preparePrompt() {
          preparePromptCalls += 1;
          return {
            tokenIds: [1, 2, 3, 4],
            inputEmbeddings: array([[[0], [1], [2], [3]]], "float32"),
          };
        },
      },
      { model, tokenizer },
      promptCache,
    );

    expect(preparePromptCalls).toBe(0);
    expect(prepared.tokenIds).toEqual([1, 2, 3, 4]);
    expect("preparedPrompt" in prepared).toBe(false);
  });

  test("requires a media identity before skipping media prompt tensors", async () => {
    const tokenizer = new TinyTokenizer();
    using model = new TinyModel();
    using promptCache = new PromptPrefixCache(1);
    expect(promptCache.store([1, 2, 3, 4], new FakeSnapshot(3))).toBe(3);

    let preparePromptCalls = 0;
    const inputEmbeddings = array([[[0], [1], [2], [3]]], "float32");
    const prepared = await prepareLoadedContentGenerationRequest(
      {
        request: contentRequest({
          kind: "content",
          messages: [{ role: "user", content: [{ kind: "text", text: "hi" }] }],
        }),
        startedAt: performance.now(),
        prompt: { text: "hi", tokenIds: [1, 2, 3, 4] },
        prepareTokenPlan() {
          return {
            tokenIds: [1, 2, 3, 4],
            canSkipPromptPreparation() {
              return true;
            },
          };
        },
        preparePrompt() {
          preparePromptCalls += 1;
          return { tokenIds: [1, 2, 3, 4], inputEmbeddings };
        },
      },
      { model, tokenizer },
      promptCache,
    );

    expect(preparePromptCalls).toBe(1);
    expect(prepared.preparedPrompt?.inputEmbeddings).toBe(inputEmbeddings);
    prepared.preparedPrompt?.inputEmbeddings?.free();
  });

  test("falls back without cache and disposes prepared tensors for stale token plans", async () => {
    const tokenizer = new TinyTokenizer();
    using model = new TinyModel();
    let preparePromptCalls = 0;
    const preparedEmbeddings = array([[[0], [1], [2]]], "float32");

    const prepared = await prepareLoadedContentGenerationRequest(
      {
        request: contentRequest({
          kind: "content",
          messages: [{ role: "user", content: [{ kind: "text", text: "hi" }] }],
        }),
        startedAt: performance.now(),
        prompt: { text: "hi", tokenIds: [1, 2, 3] },
        prepareTokenPlan() {
          return {
            tokenIds: [1, 2, 3],
            canSkipPromptPreparation() {
              return true;
            },
          };
        },
        preparePrompt() {
          preparePromptCalls += 1;
          return { tokenIds: [1, 2, 3], inputEmbeddings: preparedEmbeddings };
        },
      },
      { model, tokenizer },
    );

    expect(preparePromptCalls).toBe(1);
    expect(prepared.preparedPrompt?.inputEmbeddings).toBe(preparedEmbeddings);
    prepared.preparedPrompt?.inputEmbeddings?.free();

    const staleEmbeddings = array([[[0], [1], [2]]], "float32");
    await expect(
      prepareLoadedContentGenerationRequest(
        {
          request: contentRequest({
            kind: "content",
            messages: [{ role: "user", content: [{ kind: "text", text: "hi" }] }],
          }),
          startedAt: performance.now(),
          prompt: { text: "hi", tokenIds: [1, 2, 3] },
          prepareTokenPlan() {
            return {
              tokenIds: [1, 2, 4],
              canSkipPromptPreparation() {
                return false;
              },
            };
          },
          preparePrompt() {
            return { tokenIds: [1, 2, 3], inputEmbeddings: staleEmbeddings };
          },
        },
        { model, tokenizer },
      ),
    ).rejects.toThrow("content token plan did not match prepared prompt token ids");
    expect(staleEmbeddings.isDisposed).toBe(true);

    const shortPlanEmbeddings = array([[[0], [1], [2]]], "float32");
    await expect(
      prepareLoadedContentGenerationRequest(
        {
          request: contentRequest({
            kind: "content",
            messages: [{ role: "user", content: [{ kind: "text", text: "hi" }] }],
          }),
          startedAt: performance.now(),
          prompt: { text: "hi", tokenIds: [1, 2, 3] },
          prepareTokenPlan() {
            return {
              tokenIds: [1, 2],
              canSkipPromptPreparation() {
                return false;
              },
            };
          },
          preparePrompt() {
            return { tokenIds: [1, 2, 3], inputEmbeddings: shortPlanEmbeddings };
          },
        },
        { model, tokenizer },
      ),
    ).rejects.toThrow("content token plan did not match prepared prompt token ids");
    expect(shortPlanEmbeddings.isDisposed).toBe(true);
  });
});
