import { describe, expect, test } from "bun:test";

import { array, type MxArray, type ParameterTree } from "@mlxts/core";
import { createContinuousBatchTokenScheduler, GenerationAbortError } from "../../generation";
import type {
  BaseModelConfig,
  CausalLM,
  DecoderCache,
  ForwardOptions,
  TransformerCache,
} from "../../types";
import { KVCache } from "../cache";

class DeterministicBatchModel implements CausalLM {
  readonly family = "gemma";
  readonly layerCount = 1;
  readonly config: BaseModelConfig = {
    family: "gemma",
    modelType: "deterministic-batch-test",
    rawConfig: {},
    vocabSize: 3,
    hiddenSize: 1,
    numHiddenLayers: 1,
  };
  lastForwardCache: DecoderCache | undefined;
  readonly forwardBatchSizes: number[] = [];
  readonly forwardSequenceLengths: number[] = [];

  forward(inputIds: MxArray, options?: ForwardOptions): MxArray {
    this.lastForwardCache = options?.cache;
    const [batchSize, sequenceLength] = inputIds.shape;
    if (batchSize === undefined || sequenceLength === undefined) {
      throw new Error("DeterministicBatchModel.forward expected rank-2 token ids.");
    }
    this.forwardBatchSizes.push(batchSize);
    this.forwardSequenceLengths.push(sequenceLength);
    options?.cache?.advance(sequenceLength);
    return array(
      Array.from({ length: batchSize }, () =>
        Array.from({ length: sequenceLength }, () => [0.1, 0.2, 0.9]),
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

describe("continuous batch token scheduler", () => {
  test("admits a row after decode has started", async () => {
    using model = new DeterministicBatchModel();
    const batches: number[] = [];
    const scheduler = createContinuousBatchTokenScheduler(model, {
      maxBatchSize: 2,
      temperature: 0,
      eosTokenIds: [],
      onBatch(event) {
        batches.push(event.batchSize);
      },
    });
    let second: Promise<unknown> | undefined;

    const first = scheduler.enqueue({
      id: "first",
      promptTokenIds: [0],
      maxTokens: 3,
      onToken(_tokenId, tokenIds) {
        if (tokenIds.length === 1 && second === undefined) {
          second = scheduler.enqueue({ id: "second", promptTokenIds: [1], maxTokens: 2 });
        }
      },
    });

    await expect(first).resolves.toEqual({ tokenIds: [2, 2, 2], finishReason: "length" });
    await expect(second).resolves.toEqual({ tokenIds: [2, 2], finishReason: "length" });
    expect(batches).toContain(2);
    expect(model.forwardBatchSizes).toContain(2);
  });

  test("chunks a long waiting prompt while active rows keep decoding", async () => {
    using model = new DeterministicBatchModel();
    const prefillProgress: string[] = [];
    const scheduler = createContinuousBatchTokenScheduler(model, {
      maxBatchSize: 2,
      prefillStepSize: 2,
      temperature: 0,
      eosTokenIds: [],
    });
    let second: Promise<unknown> | undefined;

    const first = scheduler.enqueue({
      id: "first",
      promptTokenIds: [0],
      maxTokens: 4,
      onToken(_tokenId, tokenIds) {
        if (tokenIds.length === 1 && second === undefined) {
          second = scheduler.enqueue({
            id: "second",
            promptTokenIds: [1, 1, 1, 1, 1, 1],
            maxTokens: 1,
            onPrefillProgress(event) {
              prefillProgress.push(
                `${event.processedTokens}/${event.totalTokens}:${event.chunkTokens}`,
              );
            },
          });
        }
      },
    });

    await expect(first).resolves.toEqual({ tokenIds: [2, 2, 2, 2], finishReason: "length" });
    await expect(second).resolves.toEqual({ tokenIds: [2], finishReason: "length" });

    expect(prefillProgress).toEqual(["2/5:2", "4/5:2", "5/5:1"]);
    const chunkForwards = model.forwardSequenceLengths
      .map((sequenceLength, index) => ({ sequenceLength, index }))
      .filter(({ sequenceLength }) => sequenceLength === 2);
    expect(chunkForwards).toHaveLength(2);
    expect(
      model.forwardSequenceLengths.slice(
        (chunkForwards[0]?.index ?? 0) + 1,
        chunkForwards[1]?.index ?? 0,
      ),
    ).toContain(1);
    expect(model.forwardSequenceLengths).not.toContain(6);
  });

  test("rejects rows aborted during partial prefill", async () => {
    using model = new DeterministicBatchModel();
    const controller = new AbortController();
    const scheduler = createContinuousBatchTokenScheduler(model, {
      maxBatchSize: 2,
      prefillStepSize: 2,
      temperature: 0,
      eosTokenIds: [],
    });
    let second: Promise<unknown> | undefined;

    const first = scheduler.enqueue({
      id: "first",
      promptTokenIds: [0],
      maxTokens: 3,
      onToken(_tokenId, tokenIds) {
        if (tokenIds.length === 1 && second === undefined) {
          second = scheduler.enqueue({
            id: "second",
            promptTokenIds: [1, 1, 1, 1, 1, 1],
            maxTokens: 1,
            abortSignal: controller.signal,
            onPrefillProgress() {
              controller.abort();
            },
          });
          second.catch(() => undefined);
        }
      },
    });

    await expect(first).resolves.toEqual({ tokenIds: [2, 2, 2], finishReason: "length" });
    await expect(second).rejects.toBeInstanceOf(GenerationAbortError);
    expect(model.forwardSequenceLengths).not.toContain(6);
  });

  test("filters mixed length rows independently", async () => {
    using model = new DeterministicBatchModel();
    const scheduler = createContinuousBatchTokenScheduler(model, {
      maxBatchSize: 2,
      temperature: 0,
      eosTokenIds: [],
    });

    const results = await Promise.all([
      scheduler.enqueue({ promptTokenIds: [0], maxTokens: 1 }),
      scheduler.enqueue({ promptTokenIds: [1], maxTokens: 3 }),
    ]);

    expect(results).toEqual([
      { tokenIds: [2], finishReason: "length" },
      { tokenIds: [2, 2, 2], finishReason: "length" },
    ]);
    expect(model.forwardBatchSizes).toContain(2);
  });

  test("uses the batch window before the first decode loop starts", async () => {
    using model = new DeterministicBatchModel();
    const scheduler = createContinuousBatchTokenScheduler(model, {
      maxBatchSize: 2,
      batchWindowMs: 10,
      temperature: 0,
      eosTokenIds: [],
    });

    const first = scheduler.enqueue({ promptTokenIds: [0], maxTokens: 1 });
    await Bun.sleep(1);
    expect(model.forwardBatchSizes).toEqual([]);
    const second = scheduler.enqueue({ promptTokenIds: [1], maxTokens: 1 });

    await expect(Promise.all([first, second])).resolves.toEqual([
      { tokenIds: [2], finishReason: "length" },
      { tokenIds: [2], finishReason: "length" },
    ]);
    expect(model.forwardBatchSizes[0]).toBe(2);
  });

  test("removes active aborted rows", async () => {
    using model = new DeterministicBatchModel();
    const controller = new AbortController();
    const scheduler = createContinuousBatchTokenScheduler(model, {
      maxBatchSize: 2,
      temperature: 0,
      eosTokenIds: [],
    });

    const aborted = scheduler.enqueue({
      promptTokenIds: [0],
      maxTokens: 3,
      abortSignal: controller.signal,
      onToken() {
        controller.abort();
      },
    });
    const survivor = scheduler.enqueue({ promptTokenIds: [1], maxTokens: 2 });

    await expect(aborted).rejects.toBeInstanceOf(GenerationAbortError);
    await expect(survivor).resolves.toEqual({ tokenIds: [2, 2], finishReason: "length" });
  });

  test("admits waiting rows after all active rows abort", async () => {
    using model = new DeterministicBatchModel();
    const controller = new AbortController();
    const scheduler = createContinuousBatchTokenScheduler(model, {
      maxBatchSize: 1,
      temperature: 0,
      eosTokenIds: [],
    });
    let second: Promise<unknown> | undefined;

    const aborted = scheduler.enqueue({
      promptTokenIds: [0],
      maxTokens: 3,
      abortSignal: controller.signal,
      onToken() {
        controller.abort();
        second ??= scheduler.enqueue({ promptTokenIds: [1], maxTokens: 2 });
      },
    });

    await expect(aborted).rejects.toBeInstanceOf(GenerationAbortError);
    await expect(second).resolves.toEqual({ tokenIds: [2, 2], finishReason: "length" });
  });

  test("rejects queued rows aborted while waiting behind the model lane", async () => {
    using model = new DeterministicBatchModel();
    const controller = new AbortController();
    let releaseLane: (() => void) | undefined;
    const scheduler = createContinuousBatchTokenScheduler(model, {
      maxBatchSize: 1,
      temperature: 0,
      eosTokenIds: [],
      async runExclusive(work) {
        await new Promise<void>((resolve) => {
          releaseLane = resolve;
        });
        return work();
      },
    });

    const queued = scheduler.enqueue({
      promptTokenIds: [0],
      maxTokens: 1,
      abortSignal: controller.signal,
    });
    while (releaseLane === undefined) {
      await Bun.sleep(0);
    }

    controller.abort();
    releaseLane();

    await expect(queued).rejects.toBeInstanceOf(GenerationAbortError);
    expect(model.forwardBatchSizes).toEqual([]);
  });
});
