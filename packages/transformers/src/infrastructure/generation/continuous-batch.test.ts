import { describe, expect, test } from "bun:test";

import { array, type MxArray, type ParameterTree } from "@mlxts/core";
import { createContinuousBatchTokenScheduler, GenerationAbortError } from "../../generation";
import type {
  BaseModelConfig,
  CausalLM,
  DecoderCache,
  ForwardOptions,
  TransformerBatchCache,
  TransformerCache,
} from "../../types";
import { BatchKVCache, KVCache } from "../cache";
import type {
  ContinuousBatchAdmissionController,
  ContinuousBatchAdmissionRequest,
  ContinuousBatchAdmissionReservation,
} from "./continuous-batch-types";

class TrackingBatchCache implements TransformerBatchCache {
  readonly inner: BatchKVCache;
  extendCount = 0;
  filterCount = 0;

  constructor(leftPadding: readonly number[]) {
    this.inner = new BatchKVCache(1, leftPadding);
  }

  get layerCount(): number {
    return this.inner.layerCount;
  }

  get batchSize(): number {
    return this.inner.batchSize;
  }

  get length(): number {
    return this.inner.length;
  }

  get leftPadding(): readonly number[] {
    return this.inner.leftPadding;
  }

  get offsets(): readonly number[] {
    return this.inner.offsets;
  }

  updateAndFetch(layerIndex: number, keys: MxArray, values: MxArray) {
    return this.inner.updateAndFetch(layerIndex, keys, values);
  }

  advance(sequenceLength: number): void {
    this.inner.advance(sequenceLength);
  }

  filter(batchIndices: readonly number[]): void {
    this.filterCount += 1;
    this.inner.filter(batchIndices);
  }

  extend(other: TransformerBatchCache): void {
    if (!(other instanceof TrackingBatchCache)) {
      throw new Error("TrackingBatchCache.extend expected another tracking cache.");
    }
    this.extendCount += 1 + other.extendCount;
    this.inner.extend(other.inner);
  }

  extract(batchIndex: number): TransformerCache {
    return this.inner.extract(batchIndex);
  }

  offsetTensor(): MxArray {
    return this.inner.offsetTensor();
  }

  leftPaddingTensor(): MxArray {
    return this.inner.leftPaddingTensor();
  }

  isEmpty(): boolean {
    return this.inner.isEmpty();
  }

  isTrimmable(): boolean {
    return this.inner.isTrimmable();
  }

  arrays(): MxArray[] {
    return this.inner.arrays();
  }

  [Symbol.dispose](): void {
    this.inner[Symbol.dispose]();
  }
}

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

class ModelOwnedBatchCacheModel extends DeterministicBatchModel {
  readonly createdBatchCaches: TrackingBatchCache[] = [];

  createBatchCache(leftPadding: readonly number[]): TransformerBatchCache {
    const cache = new TrackingBatchCache(leftPadding);
    this.createdBatchCaches.push(cache);
    return cache;
  }
}

class DefaultSampledBatchModel extends DeterministicBatchModel {
  override readonly config: BaseModelConfig = {
    family: "gemma",
    modelType: "deterministic-batch-test",
    rawConfig: {},
    vocabSize: 3,
    hiddenSize: 1,
    numHiddenLayers: 1,
    generationDefaults: { temperature: 1, topK: 1 },
  };
}

class RepetitionPenaltyBatchModel extends DeterministicBatchModel {
  override forward(inputIds: MxArray, options?: ForwardOptions): MxArray {
    this.lastForwardCache = options?.cache;
    const [batchSize, sequenceLength] = inputIds.shape;
    if (batchSize === undefined || sequenceLength === undefined) {
      throw new Error("RepetitionPenaltyBatchModel.forward expected rank-2 token ids.");
    }
    this.forwardBatchSizes.push(batchSize);
    this.forwardSequenceLengths.push(sequenceLength);
    options?.cache?.advance(sequenceLength);
    return array(
      Array.from({ length: batchSize }, () =>
        Array.from({ length: sequenceLength }, () => [4, 3, 1]),
      ),
      "float32",
    );
  }
}

class ThrowingPrefillModel extends DeterministicBatchModel {
  override forward(inputIds: MxArray, options?: ForwardOptions): MxArray {
    const sequenceLength = inputIds.shape[1];
    if (sequenceLength === 2) {
      throw new Error("prefill failed");
    }
    return super.forward(inputIds, options);
  }
}

class TestTokenBudget implements ContinuousBatchAdmissionController {
  readonly #maxScheduledTotalTokens: number;
  readonly #listeners = new Set<() => void>();
  #scheduledTotalTokens = 0;

  constructor(maxScheduledTotalTokens: number) {
    this.#maxScheduledTotalTokens = maxScheduledTotalTokens;
  }

  tryReserve(request: ContinuousBatchAdmissionRequest) {
    if (request.totalTokens > this.#maxScheduledTotalTokens) {
      return {
        type: "rejected" as const,
        message: `${request.id} exceeds test token budget`,
      };
    }
    if (this.#scheduledTotalTokens + request.totalTokens > this.#maxScheduledTotalTokens) {
      return {
        type: "deferred" as const,
        reason: "scheduled_token_budget" as const,
        ...this.snapshot(),
      };
    }
    this.#scheduledTotalTokens += request.totalTokens;
    return {
      type: "reserved" as const,
      reservation: this.#reservation(request.totalTokens),
    };
  }

  snapshot() {
    return {
      scheduledPromptTokens: this.#scheduledTotalTokens,
      maxScheduledPromptTokens: this.#maxScheduledTotalTokens,
      scheduledCompletionTokens: 0,
      maxScheduledCompletionTokens: this.#maxScheduledTotalTokens,
      scheduledTotalTokens: this.#scheduledTotalTokens,
      maxScheduledTotalTokens: this.#maxScheduledTotalTokens,
    };
  }

  onRelease(listener: () => void) {
    this.#listeners.add(listener);
    return () => this.#listeners.delete(listener);
  }

  #reservation(totalTokens: number): ContinuousBatchAdmissionReservation {
    return {
      [Symbol.dispose]: () => {
        this.#scheduledTotalTokens = Math.max(0, this.#scheduledTotalTokens - totalTokens);
        for (const listener of this.#listeners) {
          listener();
        }
      },
    };
  }
}

function tokenBudget(maxScheduledTotalTokens: number): ContinuousBatchAdmissionController {
  return new TestTokenBudget(maxScheduledTotalTokens);
}

describe("continuous batch token scheduler", () => {
  test("admits a row after decode has started", async () => {
    using model = new DeterministicBatchModel();
    const batches: number[] = [];
    const phases: string[] = [];
    const scheduler = createContinuousBatchTokenScheduler(model, {
      maxBatchSize: 2,
      temperature: 0,
      eosTokenIds: [],
      onBatch(event) {
        batches.push(event.batchSize);
      },
      onSchedulerEvent(event) {
        phases.push(event.type);
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
    expect(phases).toContain("queued");
    expect(phases).toContain("admitted");
    expect(phases).toContain("first_token");
    expect(phases).toContain("finished");
    expect(model.forwardBatchSizes).toContain(2);
  });

  test("uses model-owned batch caches for staggered row admission", async () => {
    using model = new ModelOwnedBatchCacheModel();
    const scheduler = createContinuousBatchTokenScheduler(model, {
      maxBatchSize: 2,
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
          second = scheduler.enqueue({ id: "second", promptTokenIds: [1], maxTokens: 2 });
        }
      },
    });

    await expect(first).resolves.toEqual({ tokenIds: [2, 2, 2], finishReason: "length" });
    await expect(second).resolves.toEqual({ tokenIds: [2, 2], finishReason: "length" });
    expect(model.createdBatchCaches.length).toBeGreaterThan(1);
    expect(model.createdBatchCaches.some((cache) => cache.extendCount > 0)).toBe(true);
  });

  test("chunks a long waiting prompt while active rows keep decoding", async () => {
    using model = new DeterministicBatchModel();
    const prefillProgress: string[] = [];
    const phases: string[] = [];
    const scheduler = createContinuousBatchTokenScheduler(model, {
      maxBatchSize: 2,
      prefillStepSize: 2,
      temperature: 0,
      eosTokenIds: [],
      onSchedulerEvent(event) {
        phases.push(event.type);
      },
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
    expect(phases).toContain("prefill_start");
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

  test("chunks long initial prompts instead of full-prefilling before the first decode", async () => {
    using model = new DeterministicBatchModel();
    const order: string[] = [];
    const prefillProgress: string[] = [];
    const phases: string[] = [];
    const scheduler = createContinuousBatchTokenScheduler(model, {
      maxBatchSize: 2,
      prefillStepSize: 2,
      temperature: 0,
      eosTokenIds: [],
      onSchedulerEvent(event) {
        phases.push(event.type);
      },
    });

    const first = scheduler.enqueue({
      id: "first",
      promptTokenIds: [0, 0, 0, 0, 0, 0],
      maxTokens: 1,
      onPrefillProgress(event) {
        const progress = `${event.processedTokens}/${event.totalTokens}:${event.chunkTokens}`;
        prefillProgress.push(progress);
        order.push(`first-prefill:${progress}`);
      },
      onToken() {
        order.push("first-token");
      },
    });
    const second = scheduler.enqueue({
      id: "second",
      promptTokenIds: [1],
      maxTokens: 1,
      onToken() {
        order.push("second-token");
      },
    });

    await expect(Promise.all([first, second])).resolves.toEqual([
      { tokenIds: [2], finishReason: "length" },
      { tokenIds: [2], finishReason: "length" },
    ]);

    expect(prefillProgress).toEqual(["2/5:2", "4/5:2", "5/5:1"]);
    expect(phases.filter((phase) => phase === "prefill_start")).toHaveLength(2);
    expect(order.indexOf("second-token")).toBeGreaterThanOrEqual(0);
    expect(order.indexOf("second-token")).toBeLessThan(order.indexOf("first-prefill:2/5:2"));
    expect(model.forwardSequenceLengths).not.toContain(6);
    expect(model.forwardBatchSizes[0]).toBe(1);
  });

  test("starts newly waiting rows while an initial long prompt is still prefilling", async () => {
    using model = new DeterministicBatchModel();
    const order: string[] = [];
    const scheduler = createContinuousBatchTokenScheduler(model, {
      maxBatchSize: 2,
      prefillStepSize: 2,
      temperature: 0,
      eosTokenIds: [],
    });
    let short: Promise<unknown> | undefined;

    const long = scheduler.enqueue({
      id: "long",
      promptTokenIds: [0, 0, 0, 0, 0, 0],
      maxTokens: 1,
      onPrefillProgress(event) {
        const progress = `${event.processedTokens}/${event.totalTokens}:${event.chunkTokens}`;
        order.push(`long-prefill:${progress}`);
        if (event.processedTokens === 2 && short === undefined) {
          short = scheduler.enqueue({
            id: "short",
            promptTokenIds: [1],
            maxTokens: 1,
            onToken() {
              order.push("short-token");
            },
          });
        }
      },
      onToken() {
        order.push("long-token");
      },
    });

    await expect(long).resolves.toEqual({ tokenIds: [2], finishReason: "length" });
    if (short === undefined) {
      throw new Error("Expected the short request to be enqueued during long prefill.");
    }
    await expect(short).resolves.toEqual({ tokenIds: [2], finishReason: "length" });

    expect(order.indexOf("short-token")).toBeGreaterThan(order.indexOf("long-prefill:2/5:2"));
    expect(order.indexOf("short-token")).toBeLessThan(order.indexOf("long-prefill:4/5:2"));
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

  test("rejects prefilling rows when chunked prefill fails", async () => {
    using model = new ThrowingPrefillModel();
    const scheduler = createContinuousBatchTokenScheduler(model, {
      maxBatchSize: 1,
      prefillStepSize: 2,
      temperature: 0,
      eosTokenIds: [],
    });

    await expect(
      scheduler.enqueue({
        id: "failing-prefill",
        promptTokenIds: [0, 0, 0, 0],
        maxTokens: 1,
      }),
    ).rejects.toThrow("prefill failed");
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

  test("defers rows that exceed the scheduled token budget until reservations release", async () => {
    using model = new DeterministicBatchModel();
    const phases: string[] = [];
    const scheduledTokenSnapshots: number[] = [];
    const scheduler = createContinuousBatchTokenScheduler(model, {
      maxBatchSize: 2,
      temperature: 0,
      eosTokenIds: [],
      admissionController: tokenBudget(4),
      onSchedulerEvent(event) {
        phases.push(event.type);
        scheduledTokenSnapshots.push(event.scheduledTotalTokens);
      },
    });

    const results = await Promise.all([
      scheduler.enqueue({ id: "first", promptTokenIds: [0], maxTokens: 3 }),
      scheduler.enqueue({ id: "second", promptTokenIds: [1], maxTokens: 3 }),
    ]);

    expect(results).toEqual([
      { tokenIds: [2, 2, 2], finishReason: "length" },
      { tokenIds: [2, 2, 2], finishReason: "length" },
    ]);
    expect(phases).toContain("deferred");
    expect(scheduledTokenSnapshots).toContain(4);
    expect(model.forwardBatchSizes).not.toContain(2);
  });

  test("samples rows after one batched forward pass", async () => {
    using model = new DeterministicBatchModel();
    const scheduler = createContinuousBatchTokenScheduler(model, {
      maxBatchSize: 2,
      temperature: 1,
      topK: 1,
      eosTokenIds: [],
    });

    const results = await Promise.all([
      scheduler.enqueue({ promptTokenIds: [0], maxTokens: 2 }),
      scheduler.enqueue({ promptTokenIds: [1], maxTokens: 2 }),
    ]);

    expect(results).toEqual([
      { tokenIds: [2, 2], finishReason: "length" },
      { tokenIds: [2, 2], finishReason: "length" },
    ]);
    expect(model.forwardBatchSizes).toContain(2);
  });

  test("honors model-default sampled options", async () => {
    using model = new DefaultSampledBatchModel();
    const scheduler = createContinuousBatchTokenScheduler(model, {
      maxBatchSize: 2,
      eosTokenIds: [],
    });

    const results = await Promise.all([
      scheduler.enqueue({ promptTokenIds: [0], maxTokens: 1 }),
      scheduler.enqueue({ promptTokenIds: [1], maxTokens: 1 }),
    ]);

    expect(results).toEqual([
      { tokenIds: [2], finishReason: "length" },
      { tokenIds: [2], finishReason: "length" },
    ]);
    expect(model.forwardBatchSizes).toContain(2);
  });

  test("keeps seeded sampled rows deterministic", async () => {
    using model = new DeterministicBatchModel();
    const scheduler = createContinuousBatchTokenScheduler(model, {
      maxBatchSize: 2,
      temperature: 1,
      topK: 3,
      seed: 123,
      eosTokenIds: [],
    });

    const results = await Promise.all([
      scheduler.enqueue({ promptTokenIds: [0], maxTokens: 3 }),
      scheduler.enqueue({ promptTokenIds: [0], maxTokens: 3 }),
    ]);

    expect(results[0]?.tokenIds).toEqual(results[1]?.tokenIds);
    expect(model.forwardBatchSizes).toContain(2);
  });

  test("keeps repetition-penalty history per row", async () => {
    using model = new RepetitionPenaltyBatchModel();
    const scheduler = createContinuousBatchTokenScheduler(model, {
      maxBatchSize: 2,
      temperature: 0,
      repetitionPenalty: 2,
      eosTokenIds: [],
    });

    const results = await Promise.all([
      scheduler.enqueue({ promptTokenIds: [0], maxTokens: 2 }),
      scheduler.enqueue({ promptTokenIds: [2], maxTokens: 2 }),
    ]);

    expect(results).toEqual([
      { tokenIds: [1, 0], finishReason: "length" },
      { tokenIds: [0, 1], finishReason: "length" },
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
    const phases: string[] = [];
    const scheduler = createContinuousBatchTokenScheduler(model, {
      maxBatchSize: 1,
      temperature: 0,
      eosTokenIds: [],
      onSchedulerEvent(event) {
        phases.push(event.type);
      },
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
    expect(phases).toContain("cancelled");
    expect(model.forwardBatchSizes).toEqual([]);
  });
});
