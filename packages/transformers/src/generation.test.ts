import { describe, expect, test } from "bun:test";

import { array, type MxArray, type ParameterTree } from "@mlxts/core";
import type { Tokenizer } from "@mlxts/tokenizers";
import { generateStep, generateTextStream, generateTokens, makePromptCache } from "./generation";
import { KVCache } from "./infrastructure/cache";
import type { BaseModelConfig, CausalLM, ForwardOptions, TransformerCache } from "./types";

class DeterministicGenerationModel implements CausalLM {
  readonly family = "gemma";
  readonly layerCount = 1;
  readonly config: BaseModelConfig = {
    family: "gemma",
    modelType: "deterministic-test",
    rawConfig: {},
    vocabSize: 3,
    hiddenSize: 1,
    numHiddenLayers: 1,
  };
  lastForwardCache: TransformerCache | undefined;
  lastInputEmbeddings: MxArray | undefined;
  cacheCreates = 0;

  forward(_inputIds: MxArray, options?: ForwardOptions): MxArray {
    this.lastForwardCache = options?.cache;
    this.lastInputEmbeddings = options?.inputEmbeddings;
    return array([[[0.1, 0.2, 0.9]]], "float32");
  }

  createCache(): TransformerCache {
    this.cacheCreates += 1;
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

class CountingTokenizer implements Tokenizer {
  readonly vocabSize = 8;
  readonly bosTokenId = undefined;
  readonly eosTokenIds: number[] = [];
  readonly padTokenId = undefined;
  decodeCalls = 0;

  encode(_text: string): number[] {
    return [0];
  }

  encodeWithOffsets(text: string) {
    return { ids: this.encode(text) };
  }

  encodeBatch(texts: string[]) {
    return texts.map((text) => this.encodeWithOffsets(text));
  }

  decode(tokenIds: number[]): string {
    this.decodeCalls += 1;
    return tokenIds.map((tokenId) => String.fromCharCode(97 + tokenId)).join("");
  }

  decodeBatch(batch: number[][]): string[] {
    return batch.map((entry) => this.decode(entry));
  }
}

describe("generation", () => {
  test("makePromptCache delegates to model.createCache", () => {
    using model = new DeterministicGenerationModel();
    using cache = makePromptCache(model);

    expect(model.cacheCreates).toBe(1);
    expect(cache).toBeInstanceOf(KVCache);
  });

  test("generateStep samples the next token and forwards the provided cache", () => {
    using model = new DeterministicGenerationModel();
    using cache = new KVCache(1);

    const tokenId = generateStep(model, [0, 1], cache, [0, 1], { temperature: 0 });

    expect(tokenId).toBe(2);
    expect(model.lastForwardCache).toBe(cache);
  });

  test("generateStep forwards caller-provided input embeddings", () => {
    using model = new DeterministicGenerationModel();
    using inputEmbeddings = array([[[1], [2]]], "float32");

    const tokenId = generateStep(
      model,
      [0, 1],
      undefined,
      [0, 1],
      { temperature: 0 },
      inputEmbeddings,
    );

    expect(tokenId).toBe(2);
    expect(model.lastInputEmbeddings).toBe(inputEmbeddings);
  });

  test("generateTextStream batches decode work for longer continuations", () => {
    using model = new DeterministicGenerationModel();
    const tokenizer = new CountingTokenizer();
    const chunks: string[] = [];

    const result = generateTextStream(
      model,
      tokenizer,
      "prompt",
      { maxTokens: 96, temperature: 0, eosTokenIds: [] },
      (chunk) => {
        chunks.push(chunk);
      },
    );

    expect(result.tokenIds).toHaveLength(96);
    expect(chunks.join("")).toBe(result.text);
    expect(tokenizer.decodeCalls).toBeLessThan(result.tokenIds.length);
  });

  test("generation helpers expose token progress without forcing text decode", () => {
    using model = new DeterministicGenerationModel();
    const counts: number[] = [];

    const result = generateTokens(
      model,
      [0],
      { maxTokens: 3, temperature: 0, eosTokenIds: [] },
      (_tokenId, tokenIds) => {
        counts.push(tokenIds.length);
      },
    );

    expect(result.tokenIds).toHaveLength(3);
    expect(counts).toEqual([1, 2, 3]);
  });
});
