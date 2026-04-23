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

class TinyTokenizer implements Tokenizer {
  readonly vocabSize = 4;
  readonly bosTokenId = undefined;
  readonly eosTokenIds: number[] = [];
  readonly padTokenId = undefined;

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

class TinyModel implements CausalLM {
  readonly family = "gemma";
  readonly layerCount = 1;
  readonly config: BaseModelConfig = {
    family: "gemma",
    modelType: "serve-test",
    rawConfig: {},
    vocabSize: 4,
    hiddenSize: 1,
    numHiddenLayers: 1,
  };

  forward(_inputIds: MxArray, _options?: ForwardOptions): MxArray {
    return array([[[0.1, 0.2, 0.9, 0.0]]], "float32");
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
