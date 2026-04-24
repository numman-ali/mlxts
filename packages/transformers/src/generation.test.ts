import { describe, expect, test } from "bun:test";

import { array, type MxArray, type ParameterTree } from "@mlxts/core";
import type { Tokenizer } from "@mlxts/tokenizers";
import {
  generateBatchTokens,
  generatePreparedTokenEvents,
  generateStep,
  generateTextStream,
  generateTokenEvents,
  generateTokens,
  makePromptCache,
} from "./generation";
import { KVCache } from "./infrastructure/cache";
import type {
  BaseModelConfig,
  BatchTokenGenerationEvent,
  CausalLM,
  DecoderCache,
  ForwardOptions,
  PrefillProgressEvent,
  TokenGenerationEvent,
  TransformerCache,
} from "./types";

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
  lastForwardCache: DecoderCache | undefined;
  lastInputEmbeddings: MxArray | undefined;
  cacheCreates = 0;
  readonly forwardBatchSizes: number[] = [];

  forward(inputIds: MxArray, options?: ForwardOptions): MxArray {
    this.lastForwardCache = options?.cache;
    this.lastInputEmbeddings = options?.inputEmbeddings;
    const [batchSize, sequenceLength] = inputIds.shape;
    if (batchSize === undefined || sequenceLength === undefined) {
      throw new Error("DeterministicGenerationModel.forward expected rank-2 token ids.");
    }
    this.forwardBatchSizes.push(batchSize);
    options?.cache?.advance(sequenceLength);
    return array(
      Array.from({ length: batchSize }, () =>
        Array.from({ length: sequenceLength }, () => [0.1, 0.2, 0.9]),
      ),
      "float32",
    );
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

  test("generation helpers expose prompt prefill progress for chunked prompts", () => {
    using model = new DeterministicGenerationModel();
    const progress: PrefillProgressEvent[] = [];

    const result = generateTokens(model, [0, 1, 2, 0, 1], {
      maxTokens: 1,
      temperature: 0,
      eosTokenIds: [],
      prefillStepSize: 2,
      onPrefillProgress(event) {
        progress.push(event);
      },
    });

    expect(result.tokenIds).toEqual([2]);
    expect(progress).toEqual([
      { processedTokens: 2, totalTokens: 4, chunkTokens: 2 },
      { processedTokens: 4, totalTokens: 4, chunkTokens: 2 },
    ]);
  });

  test("generateBatchTokens supports per-row lengths and token events", () => {
    using model = new DeterministicGenerationModel();
    const events: BatchTokenGenerationEvent[] = [];

    const results = generateBatchTokens(
      model,
      [[0], [1], [2]],
      {
        maxTokens: [1, 3, 0],
        temperature: 0,
        eosTokenIds: [],
      },
      (event) => {
        events.push(event);
      },
    );

    expect(results).toEqual([
      { tokenIds: [2], finishReason: "length" },
      { tokenIds: [2, 2, 2], finishReason: "length" },
      { tokenIds: [], finishReason: "length" },
    ]);
    expect(model.forwardBatchSizes[0]).toBe(2);
    expect(events).toEqual([
      { type: "done", batchIndex: 2, tokenIds: [], finishReason: "length" },
      { type: "token", batchIndex: 0, tokenId: 2, completionTokens: 1 },
      { type: "done", batchIndex: 0, tokenIds: [2], finishReason: "length" },
      { type: "token", batchIndex: 1, tokenId: 2, completionTokens: 1 },
      { type: "token", batchIndex: 1, tokenId: 2, completionTokens: 2 },
      { type: "token", batchIndex: 1, tokenId: 2, completionTokens: 3 },
      { type: "done", batchIndex: 1, tokenIds: [2, 2, 2], finishReason: "length" },
    ]);
  });

  test("generateBatchTokens validates per-row length options", () => {
    using model = new DeterministicGenerationModel();

    expect(() =>
      generateBatchTokens(model, [[0], [1]], {
        maxTokens: [1],
        temperature: 0,
        eosTokenIds: [],
      }),
    ).toThrow("maxTokens length 1 must match batch size 2");
    expect(() =>
      generateBatchTokens(model, [[0]], {
        maxTokens: [-1],
        temperature: 0,
        eosTokenIds: [],
      }),
    ).toThrow("maxTokens values must be non-negative integers");
  });

  test("generateTokenEvents streams token progress and a final summary", async () => {
    using model = new DeterministicGenerationModel();
    const events: TokenGenerationEvent[] = [];

    for await (const event of generateTokenEvents(model, [0], {
      maxTokens: 3,
      temperature: 0,
      eosTokenIds: [],
    })) {
      events.push(event);
    }

    expect(events).toEqual([
      { type: "token", tokenId: 2, completionTokens: 1 },
      { type: "token", tokenId: 2, completionTokens: 2 },
      { type: "token", tokenId: 2, completionTokens: 3 },
      { type: "done", tokenIds: [2, 2, 2], finishReason: "length" },
    ]);
  });

  test("generateTokenEvents validates prompt and token limits", () => {
    using model = new DeterministicGenerationModel();

    expect(() =>
      generateTokenEvents(model, [], {
        maxTokens: 1,
        temperature: 0,
        eosTokenIds: [],
      }),
    ).toThrow("promptTokenIds must contain at least one token");
    expect(() =>
      generateTokenEvents(model, [0], {
        maxTokens: -1,
        temperature: 0,
        eosTokenIds: [],
      }),
    ).toThrow("maxTokens must be >= 0");
  });

  test("generateTokenEvents returns an immediate done event for zero-token requests", async () => {
    using model = new DeterministicGenerationModel();
    const events: TokenGenerationEvent[] = [];

    for await (const event of generateTokenEvents(model, [0], {
      maxTokens: 0,
      temperature: 0,
      eosTokenIds: [],
    })) {
      events.push(event);
    }

    expect(events).toEqual([{ type: "done", tokenIds: [], finishReason: "length" }]);
    expect(model.cacheCreates).toBe(0);
  });

  test("generateTokenEvents supports uncached streaming", async () => {
    using model = new DeterministicGenerationModel();
    const events: TokenGenerationEvent[] = [];

    for await (const event of generateTokenEvents(model, [0], {
      maxTokens: 2,
      temperature: 0,
      eosTokenIds: [],
      useCache: false,
    })) {
      events.push(event);
    }

    expect(events).toEqual([
      { type: "token", tokenId: 2, completionTokens: 1 },
      { type: "token", tokenId: 2, completionTokens: 2 },
      { type: "done", tokenIds: [2, 2], finishReason: "length" },
    ]);
    expect(model.lastForwardCache).toBeUndefined();
    expect(model.cacheCreates).toBe(0);
  });

  test("generateTokenEvents reuses an external cache and stops on EOS after prefill", async () => {
    using model = new DeterministicGenerationModel();
    using cache = new KVCache(1);
    const events: TokenGenerationEvent[] = [];

    for await (const event of generateTokenEvents(model, [0, 1], {
      maxTokens: 4,
      temperature: 0,
      eosTokenIds: [2],
      cache,
      prefillStepSize: 1,
    })) {
      events.push(event);
    }

    expect(events).toEqual([
      { type: "token", tokenId: 2, completionTokens: 1 },
      { type: "done", tokenIds: [2], finishReason: "eos" },
    ]);
    expect(model.lastForwardCache).toBe(cache);
    expect(model.cacheCreates).toBe(0);
  });

  test("generatePreparedTokenEvents preserves prepared prompt generation", async () => {
    using model = new DeterministicGenerationModel();
    const doneEvents: TokenGenerationEvent[] = [];

    for await (const event of generatePreparedTokenEvents(
      model,
      { tokenIds: [0] },
      { maxTokens: 1, temperature: 0, eosTokenIds: [] },
    )) {
      if (event.type === "done") {
        doneEvents.push(event);
      }
    }

    expect(doneEvents).toEqual([{ type: "done", tokenIds: [2], finishReason: "length" }]);
  });
});
