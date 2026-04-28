import { describe, expect, test } from "bun:test";

import { array, type MxArray, type ParameterTree } from "@mlxts/core";
import type { Tokenizer } from "@mlxts/tokenizers";
import {
  type BaseModelConfig,
  type CausalLM,
  type ChatMessage,
  type ForwardOptions,
  type InteractionProfile,
  KVCache,
  type Qwen3_5VisionPreprocessorConfig,
  type TransformerCache,
} from "@mlxts/transformers";
import {
  createQwen3_5ImageContentAdapter,
  loadContentGenerationRequest,
  prepareLoadedContentGenerationRequest,
} from "./transformers-engine-content";
import type { NormalizedGenerationRequest } from "./types";

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

  test("rejects unsupported Qwen content shapes before tensor preparation", async () => {
    const adapter = createQwen3_5ImageContentAdapter(qwenPreprocessor);
    const context = { tokenizer: new TinyTokenizer(), interactionProfile: chatProfile([]) };
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
});
