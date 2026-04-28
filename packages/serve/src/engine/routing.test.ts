import { describe, expect, test } from "bun:test";

import type { MxArray, ParameterTree } from "@mlxts/core";
import type { Tokenizer } from "@mlxts/tokenizers";
import type {
  BaseModelConfig,
  CacheLayerKind,
  CausalLM,
  ForwardOptions,
  TransformerCache,
  TransformerCacheForkOptions,
  TransformerCacheSnapshot,
} from "@mlxts/transformers";
import type { NormalizedGenerationRequest } from "../types";
import type { TransformersGenerationEngineOptions } from "./index";
import {
  canUseStaticBatchGeneration,
  continuousBatchIneligibilityReason,
  routeDecisionForRequest,
  staticBatchIneligibilityReason,
} from "./routing";

class RoutingTokenizer implements Tokenizer {
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
    return tokenIds.map(() => "a").join("");
  }

  decodeBatch(batch: number[][]): string[] {
    return batch.map((tokenIds) => this.decode(tokenIds));
  }
}

class RoutingCacheSnapshot implements TransformerCacheSnapshot {
  readonly offset: number;
  readonly layerKinds: readonly CacheLayerKind[];
  readonly trimmable = true;

  constructor(layerKinds: readonly CacheLayerKind[], offset: number) {
    this.layerKinds = [...layerKinds];
    this.offset = offset;
  }

  canFork(options: TransformerCacheForkOptions = {}): boolean {
    return (options.offset ?? this.offset) <= this.offset;
  }

  fork(options: TransformerCacheForkOptions = {}): TransformerCache {
    return new RoutingCache(this.layerKinds, options.offset ?? this.offset);
  }

  [Symbol.dispose](): void {}
}

class RoutingCache implements TransformerCache {
  readonly layerCount: number;
  readonly layerKinds: readonly CacheLayerKind[];
  offset: number;
  readonly #onDispose: (() => void) | undefined;

  constructor(layerKinds: readonly CacheLayerKind[], offset = 0, onDispose?: () => void) {
    this.layerKinds = [...layerKinds];
    this.layerCount = layerKinds.length;
    this.offset = offset;
    this.#onDispose = onDispose;
  }

  updateAndFetch(): { keys: MxArray; values: MxArray } {
    throw new Error("RoutingCache.updateAndFetch is not used by routing tests.");
  }

  advance(sequenceLength: number): void {
    this.offset += sequenceLength;
  }

  isEmpty(): boolean {
    return this.offset === 0;
  }

  isTrimmable(): boolean {
    return true;
  }

  snapshot(): TransformerCacheSnapshot {
    return new RoutingCacheSnapshot(this.layerKinds, this.offset);
  }

  arrays(): MxArray[] {
    return [];
  }

  [Symbol.dispose](): void {
    this.#onDispose?.();
  }
}

type RoutingModelOptions = {
  family?: BaseModelConfig["family"];
  modelType?: string;
  layerKinds: readonly CacheLayerKind[];
  hasBatchCache?: boolean;
  rawConfig?: Record<string, unknown>;
  generationDefaults?: BaseModelConfig["generationDefaults"];
};

class RoutingModel implements CausalLM {
  readonly family: BaseModelConfig["family"];
  readonly layerCount: number;
  readonly config: BaseModelConfig;
  readonly createBatchCache: (() => void) | undefined;
  cacheCreations = 0;
  disposedProbeCaches = 0;
  readonly #layerKinds: readonly CacheLayerKind[];

  constructor(options: RoutingModelOptions) {
    const family = options.family ?? "llama";
    this.family = family;
    this.#layerKinds = [...options.layerKinds];
    this.layerCount = options.layerKinds.length;
    this.config = {
      family,
      modelType: options.modelType ?? "llama",
      rawConfig: options.rawConfig ?? {},
      vocabSize: 4,
      hiddenSize: 1,
      numHiddenLayers: options.layerKinds.length,
      ...(options.generationDefaults === undefined
        ? {}
        : { generationDefaults: options.generationDefaults }),
    };
    this.createBatchCache = options.hasBatchCache === true ? () => {} : undefined;
  }

  forward(_inputIds: MxArray, _options?: ForwardOptions): MxArray {
    throw new Error("RoutingModel.forward is not used by routing tests.");
  }

  createCache(): TransformerCache {
    this.cacheCreations += 1;
    return new RoutingCache(this.#layerKinds, 0, () => {
      this.disposedProbeCaches += 1;
    });
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

function request(
  sampling: NormalizedGenerationRequest["sampling"] = { maxTokens: 2, temperature: 0 },
): NormalizedGenerationRequest {
  return {
    id: "route-test",
    model: "tiny",
    input: { kind: "text", text: "hi" },
    sampling,
    stream: false,
    protocol: "openai.completions",
  };
}

function options(
  model: CausalLM,
  overrides: Partial<TransformersGenerationEngineOptions> = {},
): TransformersGenerationEngineOptions {
  return {
    model,
    tokenizer: new RoutingTokenizer(),
    ...overrides,
  };
}

describe("generation routing", () => {
  test("memoizes cache layer kinds and disposes the probe cache", () => {
    using model = new RoutingModel({ layerKinds: ["full"], modelType: "llama" });
    const engineOptions = options(model);

    expect(staticBatchIneligibilityReason(request(), engineOptions)).toBe("eligible");
    expect(canUseStaticBatchGeneration(request(), engineOptions)).toBe(true);
    expect(continuousBatchIneligibilityReason(request(), engineOptions)).toBe("eligible");
    expect(routeDecisionForRequest(request(), engineOptions)).toEqual({
      route: "single",
      eligible: false,
      reason: "max_batch_size",
    });
    expect(routeDecisionForRequest(request(), { ...engineOptions, maxBatchSize: 2 })).toEqual({
      route: "continuous",
      eligible: true,
      reason: "eligible",
    });
    expect(model.cacheCreations).toBe(1);
    expect(model.disposedProbeCaches).toBe(1);
  });

  test("rejects sliding cache shape from CacheLayerKind instead of config fields", () => {
    using model = new RoutingModel({
      layerKinds: ["sliding"],
      modelType: "llama",
      rawConfig: { slidingWindow: undefined },
    });
    const engineOptions = options(model);

    expect(staticBatchIneligibilityReason(request(), engineOptions)).toBe("sliding_window_cache");
    expect(continuousBatchIneligibilityReason(request(), engineOptions)).toBe(
      "sliding_window_cache",
    );
  });

  test("keeps media, streaming, and repetition-penalty exclusions explicit", () => {
    using model = new RoutingModel({ layerKinds: ["full"], modelType: "llama" });
    using penalizedModel = new RoutingModel({
      layerKinds: ["full"],
      modelType: "llama",
      generationDefaults: { repetitionPenalty: 1.1 },
    });
    const mediaRequest: NormalizedGenerationRequest = {
      ...request(),
      input: {
        kind: "content",
        messages: [{ role: "user", content: [{ kind: "text", text: "hi" }] }],
      },
    };
    const streamRequest: NormalizedGenerationRequest = { ...request(), stream: true };
    const penalizedOptions = options(penalizedModel);
    const engineOptions = options(model);

    expect(staticBatchIneligibilityReason(mediaRequest, engineOptions)).toBe("media_input");
    expect(continuousBatchIneligibilityReason(mediaRequest, engineOptions)).toBe("media_input");
    expect(staticBatchIneligibilityReason(streamRequest, engineOptions)).toBe("streaming");
    expect(staticBatchIneligibilityReason(request(), penalizedOptions)).toBe("repetition_penalty");
  });

  test("requires model-owned batch cache for linear-recurrent routing", () => {
    using model = new RoutingModel({
      layerKinds: ["linear-recurrent", "full"],
      modelType: "future_linear",
    });
    const engineOptions = options(model, { maxBatchSize: 2 });

    expect(staticBatchIneligibilityReason(request(), engineOptions)).toBe("unsupported_model_type");
    expect(continuousBatchIneligibilityReason(request(), engineOptions)).toBe(
      "unsupported_model_type",
    );
  });

  test("accepts linear-recurrent routing without Qwen identifiers", () => {
    using model = new RoutingModel({
      family: "llama",
      layerKinds: ["linear-recurrent", "full"],
      modelType: "future_linear",
      hasBatchCache: true,
    });
    const engineOptions = options(model, { maxBatchSize: 2 });

    expect(staticBatchIneligibilityReason(request(), engineOptions)).toBe("eligible");
    expect(continuousBatchIneligibilityReason(request(), engineOptions)).toBe("eligible");
    expect(routeDecisionForRequest(request(), engineOptions)).toEqual({
      route: "continuous",
      eligible: true,
      reason: "eligible",
    });
  });

  test("accepts Gemma layer-pattern routing from CacheLayerKind instead of attention strings", () => {
    using model = new RoutingModel({
      family: "gemma",
      layerKinds: ["sliding"],
      modelType: "gemma4_text",
      rawConfig: { layerTypes: ["not_a_cache_contract"] },
    });
    const engineOptions = options(model, { maxBatchSize: 2 });

    expect(staticBatchIneligibilityReason(request(), engineOptions)).toBe("eligible");
    expect(continuousBatchIneligibilityReason(request(), engineOptions)).toBe("eligible");
    expect(routeDecisionForRequest(request(), engineOptions)).toEqual({
      route: "continuous",
      eligible: true,
      reason: "eligible",
    });
  });
});
