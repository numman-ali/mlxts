import { describe, expect, test } from "bun:test";

import { array, type MxArray, type ParameterTree } from "@mlxts/core";
import type { Tokenizer } from "@mlxts/tokenizers";
import {
  type BaseModelConfig,
  BatchKVCache,
  type CausalLM,
  type ForwardOptions,
  type InteractionProfile,
  KVCache,
  type TransformerBatchCache,
  type TransformerCache,
  type TransformerCacheForkOptions,
  type TransformerCacheSnapshot,
} from "@mlxts/transformers";
import { createTransformersGenerationEngine } from "./transformers-engine";
import { enforceGenerationMemoryBudget } from "./transformers-engine-shared";
import type { GenerationStreamEvent, NormalizedGenerationRequest, ServeEvent } from "./types";

const ROUTE_STRATEGY = {
  schedulerMode: "auto",
  cacheBackend: "managed",
  attentionBackend: "auto",
  decodingBackend: "model",
} as const;

class TinyTokenizer implements Tokenizer {
  readonly vocabSize = 4;
  readonly bosTokenId: number | undefined = undefined;
  readonly eosTokenIds: number[] = [];
  readonly padTokenId: number | undefined = undefined;

  encode(text: string): number[] {
    return text.split("").map(() => 0);
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
    return batch.map((entry) => this.decode(entry));
  }
}

class UnstableEmojiTokenizer extends TinyTokenizer {
  override decode(tokenIds: number[]): string {
    return tokenIds.length < 2 ? "\uFFFD" : "😄";
  }
}

class EosTokenizer extends TinyTokenizer {
  override readonly eosTokenIds: number[] = [2];
}

class SpecialTokenTokenizer extends TinyTokenizer {
  override readonly bosTokenId: number | undefined = 3;

  override encode(text: string, options: Parameters<Tokenizer["encode"]>[1] = {}): number[] {
    const ids = super.encode(text);
    return options.addSpecialTokens === false ? ids : [3, ...ids];
  }
}

type TinyModelConfig = Partial<BaseModelConfig> & {
  slidingWindow?: number;
  layerTypes?: string[];
};

class TinyModel implements CausalLM {
  readonly family: BaseModelConfig["family"];
  readonly layerCount = 1;
  readonly config: BaseModelConfig & {
    slidingWindow?: number;
    layerTypes?: string[];
  };
  readonly forwardBatchSizes: number[] = [];
  readonly forwardSequenceLengths: number[] = [];

  constructor(config: TinyModelConfig = {}) {
    const family = config.family ?? "gemma";
    this.family = family;
    this.config = {
      family,
      modelType: config.modelType ?? "serve-test",
      rawConfig: config.rawConfig ?? {},
      vocabSize: 4,
      hiddenSize: 1,
      numHiddenLayers: 1,
      ...(config.generationDefaults === undefined
        ? {}
        : { generationDefaults: config.generationDefaults }),
      ...(config.slidingWindow === undefined ? {} : { slidingWindow: config.slidingWindow }),
      ...(config.layerTypes === undefined ? {} : { layerTypes: config.layerTypes }),
    };
  }

  get batchForwardCount(): number {
    return this.forwardBatchSizes.filter((batchSize) => batchSize > 1).length;
  }

  forward(inputIds: MxArray, options?: ForwardOptions): MxArray {
    const [batchSize, sequenceLength] = inputIds.shape;
    if (batchSize === undefined || sequenceLength === undefined) {
      throw new Error("TinyModel.forward expected rank-2 token ids.");
    }
    this.forwardBatchSizes.push(batchSize);
    this.forwardSequenceLengths.push(sequenceLength);
    options?.cache?.advance(sequenceLength);
    return array(
      Array.from({ length: batchSize }, () =>
        Array.from({ length: sequenceLength }, () => [0.1, 0.2, 0.9, 0.0]),
      ),
      "float32",
    );
  }

  createCache(): TransformerCache {
    return new KVCache(1);
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

class CountingCacheSnapshot implements TransformerCacheSnapshot {
  readonly offset: number;
  readonly trimmable: boolean;
  disposeCount = 0;

  constructor(offset: number, trimmable = true) {
    this.offset = offset;
    this.trimmable = trimmable;
  }

  canFork(options: TransformerCacheForkOptions = {}): boolean {
    const offset = options.offset ?? this.offset;
    if (offset > this.offset || this.disposeCount !== 0) {
      return false;
    }
    return offset === this.offset || this.trimmable;
  }

  fork(options: TransformerCacheForkOptions = {}): TransformerCache {
    return new CountingCache([], options.offset ?? this.offset);
  }

  [Symbol.dispose](): void {
    this.disposeCount += 1;
  }
}

class CountingCache implements TransformerCache {
  readonly layerCount = 1;
  offset: number;
  readonly #snapshots: CountingCacheSnapshot[];
  readonly #trimmable: boolean;

  constructor(snapshots: CountingCacheSnapshot[], offset = 0, trimmable = true) {
    this.#snapshots = snapshots;
    this.offset = offset;
    this.#trimmable = trimmable;
  }

  updateAndFetch(): { keys: MxArray; values: MxArray } {
    throw new Error("CountingCache.updateAndFetch should not be called.");
  }

  advance(sequenceLength: number): void {
    this.offset += sequenceLength;
  }

  isEmpty(): boolean {
    return this.offset === 0;
  }

  isTrimmable(): boolean {
    return this.#trimmable;
  }

  snapshot(): TransformerCacheSnapshot {
    const snapshot = new CountingCacheSnapshot(this.offset, this.#trimmable);
    this.#snapshots.push(snapshot);
    return snapshot;
  }

  arrays(): MxArray[] {
    return [];
  }

  [Symbol.dispose](): void {}
}

class SnapshotCountingModel extends TinyModel {
  readonly snapshots: CountingCacheSnapshot[] = [];

  constructor(
    config: TinyModelConfig = {},
    readonly trimmableCache = true,
  ) {
    super(config);
  }

  override createCache(): TransformerCache {
    return new CountingCache(this.snapshots, 0, this.trimmableCache);
  }
}

class TinyQwenHybridModel extends TinyModel {
  constructor() {
    super({ family: "qwen", modelType: "qwen3_5_text" });
  }

  createBatchCache(leftPadding: readonly number[]): TransformerBatchCache {
    return new BatchKVCache(this.layerCount, leftPadding);
  }
}

const chatProfile: InteractionProfile = {
  kind: "chat",
  chatTemplate: {
    template: "{{ messages }}",
    format(messages) {
      return messages.map((message) => `${message.role}:${message.content}`).join("\n");
    },
  },
  compileTextPrompt(tokenizer, prompt, options = {}) {
    return {
      text: prompt,
      tokenIds: tokenizer.encode(
        prompt,
        options.addSpecialTokens === undefined
          ? {}
          : { addSpecialTokens: options.addSpecialTokens },
      ),
    };
  },
  compileMessages(tokenizer, messages, options = {}) {
    const text = messages.map((message) => `${message.role}:${message.content}`).join("\n");
    const suffix = options.enableThinking === false ? "\nno-thinking" : "";
    return { text: `${text}${suffix}`, tokenIds: tokenizer.encode(`${text}${suffix}`) };
  },
};

const qwenThinkingReplayProfile: InteractionProfile = {
  kind: "chat",
  chatTemplate: {
    template: "qwen-thinking-replay",
    format(messages) {
      return messages.map((message) => `${message.role}:${message.content}`).join("\n");
    },
  },
  compileTextPrompt(tokenizer, prompt, options = {}) {
    return {
      text: prompt,
      tokenIds: tokenizer.encode(
        prompt,
        options.addSpecialTokens === undefined
          ? {}
          : { addSpecialTokens: options.addSpecialTokens },
      ),
    };
  },
  compileMessages(_tokenizer, messages, options = {}) {
    const stableAssistantStart = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21];
    const disabledThinkingTail = [30];
    const replayedAssistantTail = [40, 41, 42, 43];
    const tokenIds =
      options.enableThinking === false && options.preserveThinking === true && messages.length > 1
        ? [...stableAssistantStart, ...replayedAssistantTail]
        : [...stableAssistantStart, ...disabledThinkingTail];
    return {
      text: messages.map((message) => `${message.role}:${message.content}`).join("\n"),
      tokenIds,
    };
  },
};

function textRequest(
  id: string,
  sampling: NormalizedGenerationRequest["sampling"] = { maxTokens: 2, temperature: 0 },
): NormalizedGenerationRequest {
  return {
    id,
    model: "tiny",
    input: { kind: "text", text: "hi" },
    sampling,
    stream: false,
    protocol: "openai.completions",
  };
}

async function collectStreamEvents(
  stream: NonNullable<ReturnType<typeof createTransformersGenerationEngine>["stream"]>,
  request: NormalizedGenerationRequest,
): Promise<GenerationStreamEvent[]> {
  const output: GenerationStreamEvent[] = [];
  for await (const event of await stream(request)) {
    output.push(event);
  }
  return output;
}

function batchEligibleModel(config: TinyModelConfig = {}): TinyModel {
  return new TinyModel({ family: "llama", modelType: "llama", ...config });
}

describe("transformers generation engine", () => {
  test("adapts text requests to generateTextStream", async () => {
    using model = new TinyModel();
    const tokenizer = new TinyTokenizer();
    const events: string[] = [];
    const prefillEvents: string[] = [];
    const engine = createTransformersGenerationEngine({
      model,
      tokenizer,
      onEvent(event) {
        if (event.type === "generation_progress") {
          events.push(`${event.promptTokens}:${event.completionTokens}/${event.maxTokens}`);
        }
        if (event.type === "generation_prefill_progress") {
          prefillEvents.push(
            `${event.processedPrefillTokens}/${event.totalPrefillTokens}:${event.chunkTokens}`,
          );
        }
      },
    });

    const result = await engine.generate({
      id: "request-1",
      model: "tiny",
      input: { kind: "text", text: "hi" },
      sampling: { maxTokens: 3, temperature: 0 },
      stream: false,
      protocol: "openai.completions",
    });

    expect(result.text).toBe("ccc");
    expect(result.finishReason).toBe("length");
    expect(result.usage).toEqual({ promptTokens: 2, completionTokens: 3, totalTokens: 5 });
    expect(events).toEqual(["2:0/3", "2:3/3"]);
    expect(prefillEvents).toEqual(["1/1:1"]);
  });

  test("uses configured cold prompt prefill chunk size", async () => {
    using model = new TinyModel();
    const tokenizer = new TinyTokenizer();
    const prefillEvents: string[] = [];
    const engine = createTransformersGenerationEngine({
      model,
      tokenizer,
      prefillStepSize: 2,
      onEvent(event) {
        if (event.type === "generation_prefill_progress") {
          prefillEvents.push(
            `${event.processedPrefillTokens}/${event.totalPrefillTokens}:${event.chunkTokens}`,
          );
        }
      },
    });

    await engine.generate({
      id: "prefill-chunk-size",
      model: "tiny",
      input: { kind: "text", text: "hello" },
      sampling: { maxTokens: 1, temperature: 0 },
      stream: false,
      protocol: "openai.completions",
    });

    expect(prefillEvents).toEqual(["2/4:2", "4/4:2"]);
  });

  test("emits model lane queue timing for single-route requests", async () => {
    using model = new TinyModel();
    const tokenizer = new TinyTokenizer();
    const events: ServeEvent[] = [];
    const engine = createTransformersGenerationEngine({
      model,
      tokenizer,
      onEvent(event) {
        events.push(event);
      },
    });

    await Promise.all([engine.generate(textRequest("one")), engine.generate(textRequest("two"))]);

    const waits = events.filter((event) => event.type === "generation_model_lane_wait");
    expect(waits.map((event) => event.id)).toEqual(["one", "two"]);
    expect(waits.map((event) => event.inFlightAtQueue)).toEqual([0, 1]);
    expect(waits.every((event) => event.waitMs >= 0)).toBe(true);
  });

  test("rejects media content before text-only prompt preparation", () => {
    using model = new TinyModel();
    const tokenizer = new TinyTokenizer();
    const events: ServeEvent[] = [];
    const engine = createTransformersGenerationEngine({
      model,
      tokenizer,
      onEvent(event) {
        events.push(event);
      },
    });

    expect(() =>
      engine.generate({
        id: "media-request",
        model: "tiny",
        input: {
          kind: "content",
          messages: [
            {
              role: "user",
              content: [
                { kind: "text", text: "Describe this." },
                { kind: "image", source: { kind: "url", url: "file://image.png" } },
              ],
            },
          ],
        },
        sampling: { maxTokens: 3, temperature: 0 },
        stream: false,
        protocol: "openai.chat_completions",
      }),
    ).toThrow("media-shaped input");

    expect(events).toContainEqual({
      type: "generation_route_decision",
      id: "media-request",
      protocol: "openai.chat_completions",
      model: "tiny",
      route: "single",
      eligible: false,
      reason: "media_input",
      modelType: "serve-test",
      maxBatchSize: 1,
      ...ROUTE_STRATEGY,
      stream: false,
    });
  });

  test("rejects batched media content before text-only prompt preparation", async () => {
    using model = new TinyModel();
    const tokenizer = new TinyTokenizer();
    const events: ServeEvent[] = [];
    const engine = createTransformersGenerationEngine({
      model,
      tokenizer,
      onEvent(event) {
        events.push(event);
      },
    });
    const generateBatch = engine.generateBatch;
    if (generateBatch === undefined) {
      throw new Error("Expected transformers engine to expose generateBatch.");
    }

    await expect(
      generateBatch([
        {
          id: "batch-media-request",
          model: "tiny",
          input: {
            kind: "content",
            messages: [
              {
                role: "user",
                content: [
                  { kind: "text", text: "Describe this." },
                  { kind: "image", source: { kind: "url", url: "file://image.png" } },
                ],
              },
            ],
          },
          sampling: { maxTokens: 3, temperature: 0 },
          stream: false,
          protocol: "openai.chat_completions",
        },
      ]),
    ).rejects.toThrow("media-shaped input");

    expect(events).toContainEqual({
      type: "generation_route_decision",
      id: "batch-media-request",
      protocol: "openai.chat_completions",
      model: "tiny",
      route: "single",
      eligible: false,
      reason: "media_input",
      modelType: "serve-test",
      maxBatchSize: 1,
      ...ROUTE_STRATEGY,
      stream: false,
    });
  });

  test("applies text stop sequences above token-level EOS handling", async () => {
    using model = new TinyModel();
    const tokenizer = new TinyTokenizer();
    const engine = createTransformersGenerationEngine({ model, tokenizer });

    const result = await engine.generate({
      id: "request-1",
      model: "tiny",
      input: { kind: "text", text: "hi" },
      sampling: { maxTokens: 3, temperature: 0, stop: ["cc"] },
      stream: false,
      protocol: "openai.completions",
    });

    expect(result.text).toBe("");
    expect(result.finishReason).toBe("stop");
  });

  test("can ignore EOS explicitly for exact-length throughput runs", async () => {
    using model = new TinyModel();
    const tokenizer = new EosTokenizer();
    const engine = createTransformersGenerationEngine({ model, tokenizer });

    const stopped = await engine.generate(
      textRequest("honors-eos", { maxTokens: 3, temperature: 0 }),
    );
    const ignored = await engine.generate(
      textRequest("ignores-eos", { maxTokens: 3, temperature: 0, ignoreEos: true }),
    );

    expect(stopped.tokenIds).toEqual([2]);
    expect(stopped.finishReason).toBe("eos");
    expect(ignored.tokenIds).toEqual([2, 2, 2]);
    expect(ignored.finishReason).toBe("length");
  });

  test("can ignore EOS explicitly on streaming requests", async () => {
    using model = new TinyModel();
    const tokenizer = new EosTokenizer();
    const engine = createTransformersGenerationEngine({ model, tokenizer });
    const stream = await engine.stream?.({
      ...textRequest("stream-ignore-eos", { maxTokens: 3, temperature: 0, ignoreEos: true }),
      stream: true,
    });
    if (stream === undefined) {
      throw new Error("Expected transformers engine to expose stream.");
    }

    const events: GenerationStreamEvent[] = [];
    for await (const event of stream) {
      events.push(event);
    }

    expect(events).toEqual([
      { type: "text", text: "c" },
      { type: "text", text: "c" },
      { type: "text", text: "c" },
      {
        type: "done",
        finishReason: "length",
        usage: { promptTokens: 2, completionTokens: 3, totalTokens: 5 },
      },
    ]);
  });

  test("counts text prompt tokens with the special tokens used for generation", async () => {
    using model = new TinyModel();
    const tokenizer = new SpecialTokenTokenizer();
    const engine = createTransformersGenerationEngine({ model, tokenizer });

    const result = await engine.generate({
      id: "request-1",
      model: "tiny",
      input: { kind: "text", text: "hi" },
      sampling: { maxTokens: 1, temperature: 0 },
      stream: false,
      protocol: "openai.completions",
    });

    expect(result.usage).toEqual({ promptTokens: 3, completionTokens: 1, totalTokens: 4 });
  });

  test("rejects requests over the configured total token budget before generation", () => {
    using model = new TinyModel();
    const tokenizer = new TinyTokenizer();
    const engine = createTransformersGenerationEngine({ model, tokenizer, maxTotalTokens: 4 });

    expect(() => {
      void engine.generate({
        id: "request-1",
        model: "tiny",
        input: { kind: "text", text: "hi" },
        sampling: { maxTokens: 3, temperature: 0 },
        stream: false,
        protocol: "openai.completions",
      });
    }).toThrow("total token limit");
  });

  test("rejects prompts over the configured prompt token budget before generation", async () => {
    using model = new TinyModel();
    const tokenizer = new TinyTokenizer();
    const engine = createTransformersGenerationEngine({ model, tokenizer, maxPromptTokens: 1 });
    const request = textRequest("too-long", { maxTokens: 1, temperature: 0 });

    expect(() => {
      void engine.generate(request);
    }).toThrow("prompt token limit");

    const generateBatch = engine.generateBatch;
    if (generateBatch === undefined) {
      throw new Error("Expected transformers engine to expose generateBatch.");
    }
    await expect(generateBatch([request])).rejects.toThrow("prompt token limit");

    await expect(
      (async () => {
        const stream = await engine.stream?.({ ...request, stream: true });
        if (stream === undefined) {
          throw new Error("Expected transformers engine to expose stream.");
        }
        for await (const _event of stream) {
          // Exhaust the stream so admission errors surface.
        }
      })(),
    ).rejects.toThrow("prompt token limit");
  });

  test("rejects requests over the configured MLX memory budget before generation", () => {
    using model = new TinyModel({
      rawConfig: {
        num_hidden_layers: 1,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        hidden_size: 100,
        head_dim: 100,
      },
    });
    const tokenizer = new TinyTokenizer();

    expect(() =>
      enforceGenerationMemoryBudget(
        { model, tokenizer, gpuMemoryUtilization: 0.5 },
        textRequest("memory-heavy", { maxTokens: 2, temperature: 0 }),
        2,
        { activeBytes: 100, cacheBytes: 0, peakBytes: 0, limitBytes: 2000 },
      ),
    ).toThrow("memory budget");
    expect(() =>
      enforceGenerationMemoryBudget(
        { model, tokenizer },
        textRequest("memory-unchecked", { maxTokens: 2, temperature: 0 }),
        2,
        { activeBytes: 100, cacheBytes: 0, peakBytes: 0, limitBytes: 2000 },
      ),
    ).not.toThrow();
  });

  test("adapts token prompts without re-encoding them as text", async () => {
    using model = new TinyModel();
    const tokenizer = new TinyTokenizer();
    const engine = createTransformersGenerationEngine({ model, tokenizer });

    const result = await engine.generate({
      id: "request-1",
      model: "tiny",
      input: { kind: "tokens", tokenIds: [1, 2, 3] },
      sampling: { maxTokens: 2, temperature: 0 },
      stream: false,
      protocol: "openai.completions",
    });

    expect(result.text).toBe("cc");
    expect(result.usage).toEqual({ promptTokens: 3, completionTokens: 2, totalTokens: 5 });
  });

  test("adapts message prompts through an interaction profile", async () => {
    using model = new TinyModel();
    const tokenizer = new TinyTokenizer();
    const engine = createTransformersGenerationEngine({
      model,
      tokenizer,
      interactionProfile: chatProfile,
    });

    const result = await engine.generate({
      id: "request-1",
      model: "tiny",
      input: { kind: "messages", messages: [{ role: "user", content: "hi" }] },
      sampling: { maxTokens: 2, temperature: 0 },
      stream: false,
      protocol: "openai.chat_completions",
    });

    expect(result.text).toBe("cc");
    expect(result.usage).toEqual({
      promptTokens: 7,
      completionTokens: 2,
      totalTokens: 9,
      cacheWriteTokens: 6,
      cacheReadTokens: 0,
    });
  });

  test("reuses message prompt prefixes through the single-request prompt cache", async () => {
    using model = new TinyModel();
    const tokenizer = new TinyTokenizer();
    const events: ServeEvent[] = [];
    const engine = createTransformersGenerationEngine({
      model,
      tokenizer,
      interactionProfile: chatProfile,
      onEvent(event) {
        events.push(event);
      },
    });
    const request = (id: string): NormalizedGenerationRequest => ({
      id,
      model: "tiny",
      input: { kind: "messages", messages: [{ role: "user", content: "hi" }] },
      sampling: { maxTokens: 2, temperature: 0 },
      stream: false,
      protocol: "openai.chat_completions",
    });

    const first = await engine.generate(request("first"));
    model.forwardSequenceLengths.length = 0;
    events.length = 0;
    const second = await engine.generate(request("second"));

    expect(first.usage).toMatchObject({ cacheWriteTokens: 6, cacheReadTokens: 0 });
    expect(second.usage).toEqual({
      promptTokens: 7,
      completionTokens: 2,
      totalTokens: 9,
      cacheReadTokens: 6,
      cacheWriteTokens: 0,
    });
    expect(model.forwardSequenceLengths).toEqual([1, 1]);
    expect(events).toContainEqual({
      type: "generation_route_decision",
      id: "second",
      protocol: "openai.chat_completions",
      model: "tiny",
      route: "single",
      eligible: false,
      reason: "prompt_prefix_cache",
      modelType: "serve-test",
      maxBatchSize: 1,
      ...ROUTE_STRATEGY,
      stream: false,
    });
    expect(events).toContainEqual({
      type: "generation_prompt_cache",
      id: "second",
      protocol: "openai.chat_completions",
      model: "tiny",
      result: "hit",
      promptTokens: 7,
      cacheReadTokens: 6,
      cacheWriteTokens: 0,
    });
  });

  test("reuses Pi-style Qwen disabled-thinking replay prompts with exact-only caches", async () => {
    using model = new SnapshotCountingModel({ family: "qwen", modelType: "qwen3_5_text" }, false);
    const tokenizer = new TinyTokenizer();
    const engine = createTransformersGenerationEngine({
      model,
      tokenizer,
      interactionProfile: qwenThinkingReplayProfile,
    });

    const first = await engine.generate({
      id: "first",
      model: "tiny",
      input: {
        kind: "messages",
        messages: [{ role: "user", content: "hi" }],
        chatTemplate: { enableThinking: false, preserveThinking: true },
      },
      sampling: { maxTokens: 1, temperature: 0 },
      stream: false,
      protocol: "openai.chat_completions",
    });
    model.forwardSequenceLengths.length = 0;

    const second = await engine.generate({
      id: "second",
      model: "tiny",
      input: {
        kind: "messages",
        messages: [
          { role: "user", content: "hi" },
          { role: "assistant", content: "Hello!" },
          { role: "user", content: "Ping" },
        ],
        chatTemplate: { enableThinking: false, preserveThinking: true },
      },
      sampling: { maxTokens: 1, temperature: 0 },
      stream: false,
      protocol: "openai.chat_completions",
    });

    expect(first.usage).toMatchObject({ cacheReadTokens: 0, cacheWriteTokens: 12 });
    expect(second.usage).toMatchObject({ cacheReadTokens: 12, cacheWriteTokens: 3 });
    expect(model.forwardSequenceLengths).toEqual([3, 1]);
  });

  test("disposes stored prompt cache snapshots when the engine is disposed", async () => {
    using model = new SnapshotCountingModel();
    const tokenizer = new TinyTokenizer();
    const engine = createTransformersGenerationEngine({
      model,
      tokenizer,
      interactionProfile: chatProfile,
    });

    await engine.generate({
      id: "request-1",
      model: "tiny",
      input: { kind: "messages", messages: [{ role: "user", content: "hi" }] },
      sampling: { maxTokens: 1, temperature: 0 },
      stream: false,
      protocol: "openai.chat_completions",
    });

    expect(model.snapshots).toHaveLength(1);
    expect(model.snapshots[0]?.disposeCount).toBe(0);
    engine[Symbol.dispose]?.();
    expect(model.snapshots[0]?.disposeCount).toBe(1);
  });

  test("uses static batch generation for eligible greedy full-cache requests", async () => {
    using model = batchEligibleModel();
    const tokenizer = new TinyTokenizer();
    const events: ServeEvent[] = [];
    const engine = createTransformersGenerationEngine({
      model,
      tokenizer,
      onEvent(event) {
        events.push(event);
      },
    });
    const generateBatch = engine.generateBatch;
    if (generateBatch === undefined) {
      throw new Error("Expected transformers engine to expose generateBatch.");
    }

    const results = await generateBatch([textRequest("one"), textRequest("two")]);

    expect(results.map((result) => result.text)).toEqual(["cc", "cc"]);
    expect(results.map((result) => result.usage)).toEqual([
      { promptTokens: 2, completionTokens: 2, totalTokens: 4 },
      { promptTokens: 2, completionTokens: 2, totalTokens: 4 },
    ]);
    expect(events).toContainEqual({
      type: "generation_route_decision",
      id: "one",
      protocol: "openai.completions",
      model: "tiny",
      route: "static",
      eligible: true,
      reason: "eligible",
      modelType: "llama",
      maxBatchSize: 1,
      ...ROUTE_STRATEGY,
      stream: false,
    });
    expect(events).toContainEqual({
      type: "generation_batch_start",
      mode: "static",
      model: "tiny",
      ids: ["one", "two"],
      batchSize: 2,
      maxTokens: 2,
      maxTokensByRequest: [2, 2],
    });
    expect(model.batchForwardCount).toBeGreaterThan(0);
  });

  test("emits scheduler phases for eligible continuous generation", async () => {
    using model = batchEligibleModel();
    const tokenizer = new TinyTokenizer();
    const events: ServeEvent[] = [];
    const engine = createTransformersGenerationEngine({
      model,
      tokenizer,
      maxBatchSize: 2,
      batchWindowMs: 1,
      onEvent(event) {
        events.push(event);
      },
    });
    const generateBatch = engine.generateBatch;
    if (generateBatch === undefined) {
      throw new Error("Expected transformers engine to expose generateBatch.");
    }

    await generateBatch([textRequest("one"), textRequest("two")]);

    expect(events).toContainEqual({
      type: "generation_route_decision",
      id: "one",
      protocol: "openai.completions",
      model: "tiny",
      route: "continuous",
      eligible: true,
      reason: "eligible",
      modelType: "llama",
      maxBatchSize: 2,
      ...ROUTE_STRATEGY,
      stream: false,
    });
    expect(
      events.some(
        (event) => event.type === "generation_batch_start" && event.mode === "continuous",
      ),
    ).toBe(false);
    expect(
      events.some(
        (event) =>
          event.type === "generation_scheduler_phase" &&
          event.phase === "admitted" &&
          event.batchSize === 2,
      ),
    ).toBe(true);
    expect(
      events.some(
        (event) => event.type === "generation_scheduler_phase" && event.phase === "first_token",
      ),
    ).toBe(true);
    expect(model.batchForwardCount).toBeGreaterThan(0);
  });

  test("streams eligible greedy full-cache requests through the continuous scheduler", async () => {
    using model = batchEligibleModel();
    const tokenizer = new TinyTokenizer();
    const events: ServeEvent[] = [];
    const engine = createTransformersGenerationEngine({
      model,
      tokenizer,
      maxBatchSize: 2,
      batchWindowMs: 1,
      onEvent(event) {
        events.push(event);
      },
    });
    const stream = engine.stream;
    if (stream === undefined) {
      throw new Error("Expected transformers engine to expose stream.");
    }

    const [first, second] = await Promise.all([
      collectStreamEvents(stream, { ...textRequest("stream-one"), stream: true }),
      collectStreamEvents(stream, { ...textRequest("stream-two"), stream: true }),
    ]);

    expect(first).toEqual([
      { type: "text", text: "c" },
      { type: "text", text: "c" },
      {
        type: "done",
        finishReason: "length",
        usage: { promptTokens: 2, completionTokens: 2, totalTokens: 4 },
      },
    ]);
    expect(second).toEqual(first);
    expect(events).toContainEqual({
      type: "generation_route_decision",
      id: "stream-one",
      protocol: "openai.completions",
      model: "tiny",
      route: "continuous",
      eligible: true,
      reason: "eligible",
      modelType: "llama",
      maxBatchSize: 2,
      ...ROUTE_STRATEGY,
      stream: true,
    });
    expect(
      events.some(
        (event) =>
          event.type === "generation_scheduler_phase" &&
          event.phase === "admitted" &&
          event.batchSize === 2,
      ),
    ).toBe(true);
    expect(model.batchForwardCount).toBeGreaterThan(0);
  });

  test("streams sampled full-cache requests through the continuous scheduler", async () => {
    using model = batchEligibleModel();
    const tokenizer = new TinyTokenizer();
    const events: ServeEvent[] = [];
    const engine = createTransformersGenerationEngine({
      model,
      tokenizer,
      maxBatchSize: 2,
      batchWindowMs: 1,
      onEvent(event) {
        events.push(event);
      },
    });
    const stream = engine.stream;
    if (stream === undefined) {
      throw new Error("Expected transformers engine to expose stream.");
    }

    const sampledRequest = (id: string) => ({
      ...textRequest(id, { maxTokens: 2, temperature: 1, topK: 1 }),
      stream: true,
    });
    const [first, second] = await Promise.all([
      collectStreamEvents(stream, sampledRequest("sampled-stream-one")),
      collectStreamEvents(stream, sampledRequest("sampled-stream-two")),
    ]);

    expect(first).toEqual([
      { type: "text", text: "c" },
      { type: "text", text: "c" },
      {
        type: "done",
        finishReason: "length",
        usage: { promptTokens: 2, completionTokens: 2, totalTokens: 4 },
      },
    ]);
    expect(second).toEqual(first);
    expect(
      events.filter(
        (event) =>
          event.type === "generation_route_decision" &&
          event.route === "continuous" &&
          event.eligible &&
          event.stream,
      ),
    ).toHaveLength(2);
    expect(model.batchForwardCount).toBeGreaterThan(0);
  });

  test("emits scheduler prefill phases for waiting continuous rows", async () => {
    using model = batchEligibleModel();
    const tokenizer = new TinyTokenizer();
    const events: ServeEvent[] = [];
    const engine = createTransformersGenerationEngine({
      model,
      tokenizer,
      maxBatchSize: 2,
      activeDecodeStepsPerPrefillChunk: 1,
      onEvent(event) {
        events.push(event);
      },
    });

    const first = engine.generate(textRequest("first", { maxTokens: 8, temperature: 0 }));
    await Bun.sleep(0);
    const second = engine.generate({
      ...textRequest("second", { maxTokens: 1, temperature: 0 }),
      input: { kind: "text", text: "longer" },
    });

    await Promise.all([first, second]);

    expect(
      events.some(
        (event) => event.type === "generation_scheduler_phase" && event.phase === "prefill_start",
      ),
    ).toBe(true);
  });

  test("admits short initial continuous rows while long initial rows prefill in chunks", async () => {
    using model = batchEligibleModel();
    const tokenizer = new TinyTokenizer();
    const order: string[] = [];
    const events: ServeEvent[] = [];
    const engine = createTransformersGenerationEngine({
      model,
      tokenizer,
      maxBatchSize: 2,
      batchWindowMs: 1,
      onEvent(event) {
        events.push(event);
        if (
          event.type === "generation_scheduler_phase" &&
          event.phase === "first_token" &&
          event.id === "short"
        ) {
          order.push("short-first-token");
        }
        if (event.type === "generation_prefill_progress" && event.id === "long") {
          order.push(`long-prefill:${event.processedPrefillTokens}`);
        }
      },
    });

    const long = engine.generate({
      ...textRequest("long", { maxTokens: 1, temperature: 0 }),
      input: { kind: "text", text: "x".repeat(4096) },
    });
    const short = engine.generate({
      ...textRequest("short", { maxTokens: 1, temperature: 0 }),
      input: { kind: "text", text: "y" },
    });

    await Promise.all([long, short]);

    expect(order.indexOf("short-first-token")).toBeGreaterThanOrEqual(0);
    expect(order.indexOf("short-first-token")).toBeLessThan(order.indexOf("long-prefill:2048"));
    expect(model.forwardSequenceLengths).not.toContain(4096);
    expect(
      events.filter(
        (event) => event.type === "generation_scheduler_phase" && event.phase === "prefill_start",
      ),
    ).toHaveLength(2);
  });

  test("emits scheduler cancellation phases for aborted continuous rows", async () => {
    using model = batchEligibleModel();
    const tokenizer = new TinyTokenizer();
    const controller = new AbortController();
    const events: ServeEvent[] = [];
    const engine = createTransformersGenerationEngine({
      model,
      tokenizer,
      maxBatchSize: 2,
      batchWindowMs: 10,
      onEvent(event) {
        events.push(event);
      },
    });

    const aborted = engine.generate({
      ...textRequest("aborted", { maxTokens: 1, temperature: 0 }),
      abortSignal: controller.signal,
    });
    controller.abort();

    await expect(aborted).rejects.toThrow("cancelled");
    expect(
      events.some(
        (event) => event.type === "generation_scheduler_phase" && event.phase === "cancelled",
      ),
    ).toBe(true);
  });

  test("statically batches Qwen and Gemma layer-pattern prompts without continuous scheduling", async () => {
    using qwenModel = new TinyModel({ family: "qwen", modelType: "qwen3_5_text" });
    using gemmaSlidingModel = new TinyModel({
      family: "gemma",
      modelType: "gemma4_text",
      slidingWindow: 16,
      layerTypes: ["sliding_attention"],
    });
    const tokenizer = new TinyTokenizer();
    const qwenEvents: ServeEvent[] = [];
    const gemmaEvents: ServeEvent[] = [];
    const qwenEngine = createTransformersGenerationEngine({
      model: qwenModel,
      tokenizer,
      onEvent(event) {
        qwenEvents.push(event);
      },
    });
    const gemmaEngine = createTransformersGenerationEngine({
      model: gemmaSlidingModel,
      tokenizer,
      onEvent(event) {
        gemmaEvents.push(event);
      },
    });
    const qwenBatch = qwenEngine.generateBatch;
    const gemmaBatch = gemmaEngine.generateBatch;
    if (qwenBatch === undefined || gemmaBatch === undefined) {
      throw new Error("Expected transformers engine to expose generateBatch.");
    }

    await qwenBatch([textRequest("qwen-one"), textRequest("qwen-two")]);
    await gemmaBatch([textRequest("gemma-one"), textRequest("gemma-two")]);

    expect(qwenModel.batchForwardCount).toBeGreaterThan(0);
    expect(gemmaSlidingModel.batchForwardCount).toBeGreaterThan(0);
    expect(qwenEvents.some((event) => event.type === "generation_scheduler_phase")).toBe(false);
    expect(gemmaEvents.some((event) => event.type === "generation_scheduler_phase")).toBe(false);
    expect(qwenEvents).toContainEqual({
      type: "generation_route_decision",
      id: "qwen-one",
      protocol: "openai.completions",
      model: "tiny",
      route: "static",
      eligible: true,
      reason: "eligible",
      modelType: "qwen3_5_text",
      maxBatchSize: 1,
      ...ROUTE_STRATEGY,
      stream: false,
    });
    expect(qwenEvents).toContainEqual({
      type: "generation_batch_start",
      mode: "static",
      model: "tiny",
      ids: ["qwen-one", "qwen-two"],
      batchSize: 2,
      maxTokens: 2,
      maxTokensByRequest: [2, 2],
    });
    expect(gemmaEvents).toContainEqual({
      type: "generation_route_decision",
      id: "gemma-one",
      protocol: "openai.completions",
      model: "tiny",
      route: "static",
      eligible: true,
      reason: "eligible",
      modelType: "gemma4_text",
      maxBatchSize: 1,
      ...ROUTE_STRATEGY,
      stream: false,
    });
    expect(gemmaEvents).toContainEqual({
      type: "generation_batch_start",
      mode: "static",
      model: "tiny",
      ids: ["gemma-one", "gemma-two"],
      batchSize: 2,
      maxTokens: 2,
      maxTokensByRequest: [2, 2],
    });
  });

  test("coalesces concurrent Qwen requests through static batching only", async () => {
    using model = new TinyModel({ family: "qwen", modelType: "qwen3_5_text" });
    const tokenizer = new TinyTokenizer();
    const events: ServeEvent[] = [];
    const engine = createTransformersGenerationEngine({
      model,
      tokenizer,
      maxBatchSize: 2,
      batchWindowMs: 1,
      onEvent(event) {
        events.push(event);
      },
    });

    const results = await Promise.all([
      engine.generate(textRequest("qwen-one")),
      engine.generate(textRequest("qwen-two")),
    ]);

    expect(results.map((result) => result.text)).toEqual(["cc", "cc"]);
    expect(model.batchForwardCount).toBeGreaterThan(0);
    expect(events.some((event) => event.type === "generation_scheduler_phase")).toBe(false);
    expect(events).toContainEqual({
      type: "generation_route_decision",
      id: "qwen-one",
      protocol: "openai.completions",
      model: "tiny",
      route: "static",
      eligible: true,
      reason: "eligible",
      modelType: "qwen3_5_text",
      maxBatchSize: 2,
      ...ROUTE_STRATEGY,
      stream: false,
    });
    expect(events).toContainEqual({
      type: "generation_batch_start",
      mode: "static",
      model: "tiny",
      ids: ["qwen-one", "qwen-two"],
      batchSize: 2,
      maxTokens: 2,
      maxTokensByRequest: [2, 2],
    });
  });

  test("routes Qwen hybrid-cache requests through continuous batching", async () => {
    using model = new TinyQwenHybridModel();
    const tokenizer = new TinyTokenizer();
    const events: ServeEvent[] = [];
    const engine = createTransformersGenerationEngine({
      model,
      tokenizer,
      maxBatchSize: 2,
      batchWindowMs: 1,
      onEvent(event) {
        events.push(event);
      },
    });

    const results = await Promise.all([
      engine.generate(textRequest("qwen-one")),
      engine.generate(textRequest("qwen-two")),
    ]);

    expect(results.map((result) => result.text)).toEqual(["cc", "cc"]);
    expect(model.batchForwardCount).toBeGreaterThan(0);
    expect(
      events.some(
        (event) =>
          event.type === "generation_scheduler_phase" &&
          event.phase === "admitted" &&
          event.batchSize === 2,
      ),
    ).toBe(true);
    expect(events).toContainEqual({
      type: "generation_route_decision",
      id: "qwen-one",
      protocol: "openai.completions",
      model: "tiny",
      route: "continuous",
      eligible: true,
      reason: "eligible",
      modelType: "qwen3_5_text",
      maxBatchSize: 2,
      ...ROUTE_STRATEGY,
      stream: false,
    });
    expect(
      events.some((event) => event.type === "generation_batch_start" && event.mode === "static"),
    ).toBe(false);
  });

  test("streams Qwen hybrid-cache requests through continuous batching", async () => {
    using model = new TinyQwenHybridModel();
    const tokenizer = new TinyTokenizer();
    const events: ServeEvent[] = [];
    const engine = createTransformersGenerationEngine({
      model,
      tokenizer,
      maxBatchSize: 2,
      batchWindowMs: 1,
      onEvent(event) {
        events.push(event);
      },
    });
    const stream = engine.stream;
    if (stream === undefined) {
      throw new Error("Expected transformers engine to expose stream.");
    }

    const [first, second] = await Promise.all([
      collectStreamEvents(stream, { ...textRequest("qwen-stream-one"), stream: true }),
      collectStreamEvents(stream, { ...textRequest("qwen-stream-two"), stream: true }),
    ]);

    expect(first.map((event) => event.type)).toEqual(["text", "text", "done"]);
    expect(second.map((event) => event.type)).toEqual(first.map((event) => event.type));
    expect(
      events.some(
        (event) =>
          event.type === "generation_scheduler_phase" &&
          event.phase === "admitted" &&
          event.batchSize === 2,
      ),
    ).toBe(true);
    expect(model.batchForwardCount).toBeGreaterThan(0);
    expect(events).toContainEqual({
      type: "generation_route_decision",
      id: "qwen-stream-one",
      protocol: "openai.completions",
      model: "tiny",
      route: "continuous",
      eligible: true,
      reason: "eligible",
      modelType: "qwen3_5_text",
      maxBatchSize: 2,
      ...ROUTE_STRATEGY,
      stream: true,
    });
  });

  test("aborts scheduler-backed Qwen streams when the iterator closes early", async () => {
    using model = new TinyQwenHybridModel();
    const tokenizer = new TinyTokenizer();
    const events: ServeEvent[] = [];
    const engine = createTransformersGenerationEngine({
      model,
      tokenizer,
      maxBatchSize: 2,
      onEvent(event) {
        events.push(event);
      },
    });
    const stream = engine.stream;
    if (stream === undefined) {
      throw new Error("Expected transformers engine to expose stream.");
    }

    const iterable = await stream({
      ...textRequest("qwen-close-early", { maxTokens: 128, temperature: 0 }),
      stream: true,
    });
    const iterator = iterable[Symbol.asyncIterator]();
    const first = await iterator.next();
    expect(first.value).toEqual({ type: "text", text: "c" });
    await iterator.return?.();

    expect(
      events.some(
        (event) =>
          event.type === "generation_scheduler_phase" &&
          event.phase === "cancelled" &&
          event.id === "qwen-close-early",
      ),
    ).toBe(true);
    expect(model.forwardSequenceLengths.length).toBeLessThan(128);
  });

  test("routes concurrent Gemma layer-pattern requests through continuous batching", async () => {
    using model = new TinyModel({
      family: "gemma",
      modelType: "gemma4_text",
      slidingWindow: 16,
      layerTypes: ["sliding_attention"],
    });
    const tokenizer = new TinyTokenizer();
    const events: ServeEvent[] = [];
    const engine = createTransformersGenerationEngine({
      model,
      tokenizer,
      maxBatchSize: 2,
      batchWindowMs: 1,
      onEvent(event) {
        events.push(event);
      },
    });

    const results = await Promise.all([
      engine.generate(textRequest("gemma-one")),
      engine.generate(textRequest("gemma-two")),
    ]);

    expect(results.map((result) => result.text)).toEqual(["cc", "cc"]);
    expect(model.batchForwardCount).toBeGreaterThan(0);
    expect(
      events.some(
        (event) =>
          event.type === "generation_scheduler_phase" &&
          event.phase === "admitted" &&
          event.batchSize === 2,
      ),
    ).toBe(true);
    expect(events).toContainEqual({
      type: "generation_route_decision",
      id: "gemma-one",
      protocol: "openai.completions",
      model: "tiny",
      route: "continuous",
      eligible: true,
      reason: "eligible",
      modelType: "gemma4_text",
      maxBatchSize: 2,
      ...ROUTE_STRATEGY,
      stream: false,
    });
    expect(
      events.some((event) => event.type === "generation_batch_start" && event.mode === "static"),
    ).toBe(false);
  });

  test("streams Gemma layer-pattern requests through continuous batching", async () => {
    using model = new TinyModel({
      family: "gemma",
      modelType: "gemma4_text",
      slidingWindow: 16,
      layerTypes: ["sliding_attention"],
    });
    const tokenizer = new TinyTokenizer();
    const events: ServeEvent[] = [];
    const engine = createTransformersGenerationEngine({
      model,
      tokenizer,
      maxBatchSize: 2,
      batchWindowMs: 1,
      onEvent(event) {
        events.push(event);
      },
    });
    const stream = engine.stream;
    if (stream === undefined) {
      throw new Error("Expected transformers engine to expose stream.");
    }

    const [first, second] = await Promise.all([
      collectStreamEvents(stream, { ...textRequest("gemma-stream-one"), stream: true }),
      collectStreamEvents(stream, { ...textRequest("gemma-stream-two"), stream: true }),
    ]);

    expect(first.map((event) => event.type)).toEqual(["text", "text", "done"]);
    expect(second.map((event) => event.type)).toEqual(first.map((event) => event.type));
    expect(
      events.some(
        (event) =>
          event.type === "generation_scheduler_phase" &&
          event.phase === "admitted" &&
          event.batchSize === 2,
      ),
    ).toBe(true);
    expect(model.batchForwardCount).toBeGreaterThan(0);
    expect(events).toContainEqual({
      type: "generation_route_decision",
      id: "gemma-stream-one",
      protocol: "openai.completions",
      model: "tiny",
      route: "continuous",
      eligible: true,
      reason: "eligible",
      modelType: "gemma4_text",
      maxBatchSize: 2,
      ...ROUTE_STRATEGY,
      stream: true,
    });
  });

  test("aborts scheduler-backed Gemma streams when the iterator closes early", async () => {
    using model = new TinyModel({
      family: "gemma",
      modelType: "gemma4_text",
      slidingWindow: 16,
      layerTypes: ["sliding_attention"],
    });
    const tokenizer = new TinyTokenizer();
    const events: ServeEvent[] = [];
    const engine = createTransformersGenerationEngine({
      model,
      tokenizer,
      maxBatchSize: 2,
      onEvent(event) {
        events.push(event);
      },
    });
    const stream = engine.stream;
    if (stream === undefined) {
      throw new Error("Expected transformers engine to expose stream.");
    }

    const iterable = await stream({
      ...textRequest("gemma-close-early", { maxTokens: 128, temperature: 0 }),
      stream: true,
    });
    const iterator = iterable[Symbol.asyncIterator]();
    const first = await iterator.next();
    expect(first.value).toEqual({ type: "text", text: "c" });
    await iterator.return?.();

    expect(
      events.some(
        (event) =>
          event.type === "generation_scheduler_phase" &&
          event.phase === "cancelled" &&
          event.id === "gemma-close-early",
      ),
    ).toBe(true);
    expect(model.forwardSequenceLengths.length).toBeLessThan(128);
  });

  test("keeps per-row stop handling on static batch results", async () => {
    using model = batchEligibleModel();
    const tokenizer = new TinyTokenizer();
    const engine = createTransformersGenerationEngine({ model, tokenizer });
    const generateBatch = engine.generateBatch;
    if (generateBatch === undefined) {
      throw new Error("Expected transformers engine to expose generateBatch.");
    }

    const results = await generateBatch([
      textRequest("stopped", { maxTokens: 2, temperature: 0, stop: ["c"] }),
      textRequest("plain"),
    ]);

    expect(results[0]?.text).toBe("");
    expect(results[0]?.finishReason).toBe("stop");
    expect(results[1]?.text).toBe("cc");
    expect(results[1]?.finishReason).toBe("length");
    expect(model.batchForwardCount).toBeGreaterThan(0);
  });

  test("uses static batch generation for mixed generation lengths", async () => {
    using model = batchEligibleModel();
    const tokenizer = new TinyTokenizer();
    const engine = createTransformersGenerationEngine({ model, tokenizer });
    const generateBatch = engine.generateBatch;
    if (generateBatch === undefined) {
      throw new Error("Expected transformers engine to expose generateBatch.");
    }

    const results = await generateBatch([
      textRequest("short", { maxTokens: 1, temperature: 0 }),
      textRequest("long", { maxTokens: 2, temperature: 0 }),
    ]);

    expect(results.map((result) => result.text)).toEqual(["c", "cc"]);
    expect(model.batchForwardCount).toBeGreaterThan(0);
  });

  test("can ignore EOS explicitly on static batch generation", async () => {
    using model = batchEligibleModel();
    const tokenizer = new EosTokenizer();
    const engine = createTransformersGenerationEngine({ model, tokenizer });
    const generateBatch = engine.generateBatch;
    if (generateBatch === undefined) {
      throw new Error("Expected transformers engine to expose generateBatch.");
    }

    const stopped = await generateBatch([
      textRequest("honors-eos-1", { maxTokens: 3, temperature: 0 }),
      textRequest("honors-eos-2", { maxTokens: 3, temperature: 0 }),
    ]);
    const ignored = await generateBatch([
      textRequest("ignores-eos-1", { maxTokens: 3, temperature: 0, ignoreEos: true }),
      textRequest("ignores-eos-2", { maxTokens: 3, temperature: 0, ignoreEos: true }),
    ]);

    expect(stopped.map((result) => result.tokenIds)).toEqual([[2], [2]]);
    expect(stopped.map((result) => result.finishReason)).toEqual(["eos", "eos"]);
    expect(ignored.map((result) => result.tokenIds)).toEqual([
      [2, 2, 2],
      [2, 2, 2],
    ]);
    expect(ignored.map((result) => result.finishReason)).toEqual(["length", "length"]);
    expect(model.batchForwardCount).toBeGreaterThan(0);
  });

  test("routes explicit and model-default sampled requests through continuous batching", async () => {
    using explicitSampled = batchEligibleModel();
    using defaultSampled = batchEligibleModel({
      generationDefaults: { temperature: 1, topK: 1 },
    });
    const tokenizer = new TinyTokenizer();
    const explicitEvents: ServeEvent[] = [];
    const defaultEvents: ServeEvent[] = [];
    const explicitEngine = createTransformersGenerationEngine({
      model: explicitSampled,
      tokenizer,
      maxBatchSize: 2,
      onEvent(event) {
        explicitEvents.push(event);
      },
    });
    const defaultEngine = createTransformersGenerationEngine({
      model: defaultSampled,
      tokenizer,
      maxBatchSize: 2,
      onEvent(event) {
        defaultEvents.push(event);
      },
    });
    const explicitBatch = explicitEngine.generateBatch;
    const defaultBatch = defaultEngine.generateBatch;
    if (explicitBatch === undefined || defaultBatch === undefined) {
      throw new Error("Expected transformers engine to expose generateBatch.");
    }

    await explicitBatch([
      textRequest("sampled-one", { maxTokens: 2, temperature: 1, topK: 1 }),
      textRequest("sampled-two", { maxTokens: 2, temperature: 1, topK: 1 }),
    ]);
    await defaultBatch([
      textRequest("default-one", { maxTokens: 2 }),
      textRequest("default-two", { maxTokens: 2 }),
    ]);

    expect(explicitSampled.batchForwardCount).toBeGreaterThan(0);
    expect(defaultSampled.batchForwardCount).toBeGreaterThan(0);
    expect(
      explicitEvents.filter(
        (event) =>
          event.type === "generation_route_decision" &&
          event.route === "continuous" &&
          event.eligible,
      ),
    ).toHaveLength(2);
    expect(
      defaultEvents.filter(
        (event) =>
          event.type === "generation_route_decision" &&
          event.route === "continuous" &&
          event.eligible,
      ),
    ).toHaveLength(2);
  });

  test("preserves prompt-open reasoning handling on static chat batches", async () => {
    using model = batchEligibleModel();
    const tokenizer = new TinyTokenizer();
    const thinkingProfile: InteractionProfile = {
      ...chatProfile,
      compileMessages(tokenizer, messages) {
        const text = `${messages.map((message) => `${message.role}:${message.content}`).join("\n")}\n<think>\n`;
        return { text, tokenIds: tokenizer.encode(text) };
      },
    };
    const engine = createTransformersGenerationEngine({
      model,
      tokenizer,
      interactionProfile: thinkingProfile,
    });
    const generateBatch = engine.generateBatch;
    if (generateBatch === undefined) {
      throw new Error("Expected transformers engine to expose generateBatch.");
    }

    const results = await generateBatch([
      {
        id: "chat-one",
        model: "tiny",
        input: { kind: "messages", messages: [{ role: "user", content: "hi" }] },
        sampling: { maxTokens: 1, temperature: 0 },
        stream: false,
        protocol: "openai.chat_completions",
      },
      {
        id: "chat-two",
        model: "tiny",
        input: { kind: "messages", messages: [{ role: "user", content: "hello" }] },
        sampling: { maxTokens: 1, temperature: 0 },
        stream: false,
        protocol: "openai.chat_completions",
      },
    ]);

    expect(results.map((result) => result.text)).toEqual(["", ""]);
    expect(results.map((result) => result.reasoningContent)).toEqual(["c", "c"]);
    expect(model.batchForwardCount).toBeGreaterThan(0);
  });

  test("passes chat template thinking controls through the interaction profile", async () => {
    using model = new TinyModel();
    const tokenizer = new TinyTokenizer();
    const engine = createTransformersGenerationEngine({
      model,
      tokenizer,
      interactionProfile: chatProfile,
    });

    const result = await engine.generate({
      id: "request-1",
      model: "tiny",
      input: {
        kind: "messages",
        messages: [{ role: "user", content: "hi" }],
        chatTemplate: { enableThinking: false },
      },
      sampling: { maxTokens: 1, temperature: 0 },
      stream: false,
      protocol: "openai.chat_completions",
    });

    expect(result.usage?.promptTokens).toBe("user:hi\nno-thinking".length);
  });

  test("moves unfinished prompt-open thinking into reasoning content", async () => {
    using model = new TinyModel();
    const tokenizer = new TinyTokenizer();
    const thinkingProfile: InteractionProfile = {
      ...chatProfile,
      compileMessages(tokenizer, messages) {
        const text = `${messages.map((message) => `${message.role}:${message.content}`).join("\n")}\n<think>\n`;
        return { text, tokenIds: tokenizer.encode(text) };
      },
    };
    const engine = createTransformersGenerationEngine({
      model,
      tokenizer,
      interactionProfile: thinkingProfile,
    });

    const result = await engine.generate({
      id: "request-1",
      model: "tiny",
      input: { kind: "messages", messages: [{ role: "user", content: "hi" }] },
      sampling: { maxTokens: 2, temperature: 0 },
      stream: false,
      protocol: "openai.chat_completions",
    });

    expect(result.text).toBe("");
    expect(result.reasoningContent).toBe("cc");
    expect(result.finishReason).toBe("length");
  });

  test("streams generated text for text prompts", async () => {
    using model = new TinyModel();
    const tokenizer = new TinyTokenizer();
    const serveEvents: ServeEvent[] = [];
    const engine = createTransformersGenerationEngine({
      model,
      tokenizer,
      onEvent(event) {
        serveEvents.push(event);
      },
    });
    const events: GenerationStreamEvent[] = [];

    for await (const event of (await engine.stream?.({
      id: "request-1",
      model: "tiny",
      input: { kind: "text", text: "hi" },
      sampling: { maxTokens: 2, temperature: 0 },
      stream: true,
      protocol: "openai.completions",
    })) ?? []) {
      events.push(event);
    }

    expect(events).toEqual([
      { type: "text", text: "c" },
      { type: "text", text: "c" },
      {
        type: "done",
        finishReason: "length",
        usage: { promptTokens: 2, completionTokens: 2, totalTokens: 4 },
      },
    ]);
    expect(serveEvents.some((event) => event.type === "generation_model_lane_wait")).toBe(true);
  });

  test("coalesces streaming text when a decode interval is configured", async () => {
    using model = new TinyModel();
    const tokenizer = new TinyTokenizer();
    const engine = createTransformersGenerationEngine({
      model,
      tokenizer,
      streamDecodeInterval: 2,
    });
    const events: GenerationStreamEvent[] = [];

    for await (const event of (await engine.stream?.({
      id: "request-1",
      model: "tiny",
      input: { kind: "text", text: "hi" },
      sampling: { maxTokens: 2, temperature: 0 },
      stream: true,
      protocol: "openai.completions",
    })) ?? []) {
      events.push(event);
    }

    expect(events).toEqual([
      { type: "text", text: "cc" },
      {
        type: "done",
        finishReason: "length",
        usage: { promptTokens: 2, completionTokens: 2, totalTokens: 4 },
      },
    ]);
  });

  test("does not stream an unstable replacement character before an emoji completes", async () => {
    using model = new TinyModel();
    const tokenizer = new UnstableEmojiTokenizer();
    const engine = createTransformersGenerationEngine({ model, tokenizer });
    const events: GenerationStreamEvent[] = [];

    for await (const event of (await engine.stream?.({
      id: "request-1",
      model: "tiny",
      input: { kind: "text", text: "hi" },
      sampling: { maxTokens: 2, temperature: 0 },
      stream: true,
      protocol: "openai.completions",
    })) ?? []) {
      events.push(event);
    }

    expect(events).toEqual([
      { type: "text", text: "😄" },
      {
        type: "done",
        finishReason: "length",
        usage: { promptTokens: 2, completionTokens: 2, totalTokens: 4 },
      },
    ]);
  });

  test("streams prompt-open thinking as raw think-tagged text", async () => {
    using model = new TinyModel();
    const tokenizer = new TinyTokenizer();
    const thinkingProfile: InteractionProfile = {
      ...chatProfile,
      compileMessages(tokenizer, messages) {
        const text = `${messages.map((message) => `${message.role}:${message.content}`).join("\n")}\n<think>\n`;
        return { text, tokenIds: tokenizer.encode(text) };
      },
    };
    const engine = createTransformersGenerationEngine({
      model,
      tokenizer,
      interactionProfile: thinkingProfile,
    });
    const events: GenerationStreamEvent[] = [];

    for await (const event of (await engine.stream?.({
      id: "request-1",
      model: "tiny",
      input: { kind: "messages", messages: [{ role: "user", content: "hi" }] },
      sampling: { maxTokens: 2, temperature: 0 },
      stream: true,
      protocol: "openai.chat_completions",
    })) ?? []) {
      events.push(event);
    }

    expect(events[0]).toEqual({ type: "text", text: "<think>c" });
    expect(events[1]).toEqual({ type: "text", text: "c" });
    expect(events[2]).toEqual({
      type: "done",
      finishReason: "length",
      usage: {
        promptTokens: "user:hi\n<think>\n".length,
        completionTokens: 2,
        totalTokens: 18,
        cacheReadTokens: 0,
        cacheWriteTokens: "user:hi\n<think>\n".length - 1,
      },
    });
  });

  test("reuses message prompt prefixes for streaming chat requests", async () => {
    using model = new TinyModel();
    const tokenizer = new TinyTokenizer();
    const engine = createTransformersGenerationEngine({
      model,
      tokenizer,
      interactionProfile: chatProfile,
    });
    const request = (id: string): NormalizedGenerationRequest => ({
      id,
      model: "tiny",
      input: { kind: "messages", messages: [{ role: "user", content: "hi" }] },
      sampling: { maxTokens: 2, temperature: 0 },
      stream: true,
      protocol: "openai.chat_completions",
    });

    for await (const _event of (await engine.stream?.(request("first"))) ?? []) {
      // Prime the prompt cache.
    }
    model.forwardSequenceLengths.length = 0;

    const events: GenerationStreamEvent[] = [];
    for await (const event of (await engine.stream?.(request("second"))) ?? []) {
      events.push(event);
    }

    expect(model.forwardSequenceLengths).toEqual([1, 1]);
    expect(events.at(-1)).toEqual({
      type: "done",
      finishReason: "length",
      usage: {
        promptTokens: 7,
        completionTokens: 2,
        totalTokens: 9,
        cacheReadTokens: 6,
        cacheWriteTokens: 0,
      },
    });
  });

  test("prepares streaming message prompts before acquiring the model lane", async () => {
    using model = new TinyModel();
    const tokenizer = new TinyTokenizer();
    const serveEvents: ServeEvent[] = [];
    const engine = createTransformersGenerationEngine({
      model,
      tokenizer,
      interactionProfile: chatProfile,
      onEvent(event) {
        serveEvents.push(event);
      },
    });

    const stream = engine.stream;
    if (stream === undefined) {
      throw new Error("Expected transformers engine to expose stream.");
    }
    const events = await collectStreamEvents(stream, {
      id: "stream-chat-prepare",
      model: "tiny",
      input: { kind: "messages", messages: [{ role: "user", content: "hi" }] },
      sampling: { maxTokens: 1, temperature: 0 },
      stream: true,
      protocol: "openai.chat_completions",
    });

    expect(events.at(-1)).toMatchObject({ type: "done", finishReason: "length" });
    const prepareIndex = serveEvents.findIndex(
      (event) => event.type === "generation_prompt_prepare" && event.phase === "complete",
    );
    const laneIndex = serveEvents.findIndex((event) => event.type === "generation_model_lane_wait");
    expect(prepareIndex).toBeGreaterThanOrEqual(0);
    expect(laneIndex).toBeGreaterThanOrEqual(0);
    expect(prepareIndex).toBeLessThan(laneIndex);
    expect(serveEvents[prepareIndex]).toMatchObject({
      type: "generation_prompt_prepare",
      phase: "complete",
      id: "stream-chat-prepare",
      promptTokens: 7,
    });
  });

  test("rejects message input without an interaction profile", () => {
    using model = new TinyModel();
    const tokenizer = new TinyTokenizer();
    const engine = createTransformersGenerationEngine({ model, tokenizer });

    expect(() => {
      void engine.generate({
        id: "request-1",
        model: "tiny",
        input: { kind: "messages", messages: [{ role: "user", content: "hi" }] },
        sampling: { maxTokens: 1 },
        stream: false,
        protocol: "openai.chat_completions",
      });
    }).toThrow("requires an interaction profile");
  });
});
