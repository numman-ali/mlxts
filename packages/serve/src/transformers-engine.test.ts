import { describe, expect, test } from "bun:test";

import { array, type MxArray, type ParameterTree } from "@mlxts/core";
import type { Tokenizer } from "@mlxts/tokenizers";
import {
  type BaseModelConfig,
  type CausalLM,
  type ForwardOptions,
  type InteractionProfile,
  KVCache,
} from "@mlxts/transformers";
import { createTransformersGenerationEngine } from "./transformers-engine";
import type { GenerationStreamEvent, NormalizedGenerationRequest, ServeEvent } from "./types";

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

class SpecialTokenTokenizer extends TinyTokenizer {
  override readonly bosTokenId: number | undefined = 3;

  override encode(text: string, options: Parameters<Tokenizer["encode"]>[1] = {}): number[] {
    const ids = super.encode(text);
    return options.addSpecialTokens === false ? ids : [3, ...ids];
  }
}

class TinyModel implements CausalLM {
  readonly family: BaseModelConfig["family"];
  readonly layerCount = 1;
  readonly config: BaseModelConfig;
  readonly forwardBatchSizes: number[] = [];

  constructor(config: Partial<BaseModelConfig> = {}) {
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
    options?.cache?.advance(sequenceLength);
    return array(
      Array.from({ length: batchSize }, () =>
        Array.from({ length: sequenceLength }, () => [0.1, 0.2, 0.9, 0.0]),
      ),
      "float32",
    );
  }

  createCache() {
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

function batchEligibleModel(config: Partial<BaseModelConfig> = {}): TinyModel {
  return new TinyModel({ family: "llama", modelType: "llama", ...config });
}

describe("transformers generation engine", () => {
  test("adapts text requests to generateTextStream", async () => {
    using model = new TinyModel();
    const tokenizer = new TinyTokenizer();
    const events: string[] = [];
    const engine = createTransformersGenerationEngine({
      model,
      tokenizer,
      onEvent(event) {
        if (event.type === "generation_progress") {
          events.push(`${event.promptTokens}:${event.completionTokens}/${event.maxTokens}`);
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
    expect(result.usage).toEqual({ promptTokens: 7, completionTokens: 2, totalTokens: 9 });
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

  test("falls back for explicit or model-default sampled requests", async () => {
    using explicitSampled = batchEligibleModel();
    using defaultSampled = batchEligibleModel({
      generationDefaults: { temperature: 1, topK: 1 },
    });
    const tokenizer = new TinyTokenizer();
    const explicitEngine = createTransformersGenerationEngine({
      model: explicitSampled,
      tokenizer,
    });
    const defaultEngine = createTransformersGenerationEngine({ model: defaultSampled, tokenizer });
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

    expect(explicitSampled.batchForwardCount).toBe(0);
    expect(defaultSampled.batchForwardCount).toBe(0);
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
      { type: "text", text: "cc" },
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

    expect(events[0]).toEqual({ type: "text", text: "<think>cc" });
    expect(events[1]).toEqual({
      type: "done",
      finishReason: "length",
      usage: { promptTokens: "user:hi\n<think>\n".length, completionTokens: 2, totalTokens: 18 },
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
