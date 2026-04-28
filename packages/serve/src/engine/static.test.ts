import { describe, expect, test } from "bun:test";

import { array, type MxArray, type ParameterTree } from "@mlxts/core";
import type { Tokenizer } from "@mlxts/tokenizers";
import {
  type BaseModelConfig,
  type CausalLM,
  type ForwardOptions,
  KVCache,
  type TransformerCache,
} from "@mlxts/transformers";
import type { NormalizedGenerationRequest, ServeEvent } from "../types";
import { ModelExecutionLane } from "./execution-lane";
import type { TransformersGenerationEngineOptions } from "./index";
import { createStaticTransformersGeneration, runStaticBatchOnModelLane } from "./static";

class StaticTokenizer implements Tokenizer {
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
    return batch.map((tokenIds) => this.decode(tokenIds));
  }
}

class StaticModel implements CausalLM {
  readonly family: BaseModelConfig["family"] = "llama";
  readonly layerCount = 1;
  readonly config: BaseModelConfig = {
    family: "llama",
    modelType: "llama",
    rawConfig: {},
    vocabSize: 4,
    hiddenSize: 1,
    numHiddenLayers: 1,
  };

  forward(inputIds: MxArray, options?: ForwardOptions): MxArray {
    const [batchSize, sequenceLength] = inputIds.shape;
    if (batchSize === undefined || sequenceLength === undefined) {
      throw new Error("StaticModel.forward expected rank-2 token ids.");
    }
    options?.cache?.advance(sequenceLength);
    return array(
      Array.from({ length: batchSize }, () =>
        Array.from({ length: sequenceLength }, () => [0.1, 0.2, 0.9, 0.0]),
      ),
      "float32",
    );
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

class StaticOnlyModel extends StaticModel {
  override readonly family: BaseModelConfig["family"] = "gemma";
  override readonly config: BaseModelConfig & { layerTypes: string[]; slidingWindow: number } = {
    family: "gemma",
    modelType: "gemma3_text",
    rawConfig: {},
    vocabSize: 4,
    hiddenSize: 1,
    numHiddenLayers: 1,
    layerTypes: ["full_attention"],
    slidingWindow: 16,
  };
}

function request(id: string, abortSignal?: AbortSignal): NormalizedGenerationRequest {
  return {
    id,
    model: "tiny",
    input: { kind: "text", text: "hi" },
    sampling: { maxTokens: 2, temperature: 0 },
    stream: false,
    protocol: "openai.completions",
    ...(abortSignal === undefined ? {} : { abortSignal }),
  };
}

function options(model: CausalLM, events: ServeEvent[] = []): TransformersGenerationEngineOptions {
  return {
    model,
    tokenizer: new StaticTokenizer(),
    maxBatchSize: 2,
    onEvent(event) {
      events.push(event);
    },
  };
}

describe("static generation routing", () => {
  test("coalesces static-only requests through the static queue", async () => {
    using model = new StaticOnlyModel();
    const events: ServeEvent[] = [];
    const staticGeneration = createStaticTransformersGeneration(
      options(model, events),
      new ModelExecutionLane(1),
    );

    const first = staticGeneration.generate(request("first"));
    const second = staticGeneration.generate(request("second"));
    if (first === null || second === null) {
      throw new Error("Expected static-only requests to enqueue.");
    }

    const results = await Promise.all([first, second]);

    expect(results.map((result) => result.text)).toEqual(["cc", "cc"]);
    expect(events).toContainEqual({
      type: "generation_batch_start",
      mode: "static",
      model: "tiny",
      ids: ["first", "second"],
      batchSize: 2,
      maxTokens: 2,
      maxTokensByRequest: [2, 2],
    });
  });

  test("runs a static batch on the model lane and emits wait events", async () => {
    using model = new StaticModel();
    const first = new AbortController();
    const second = new AbortController();
    const events: ServeEvent[] = [];

    const results = await runStaticBatchOnModelLane(
      new ModelExecutionLane(1),
      options(model, events),
      [request("first", first.signal), request("second", second.signal)],
    );

    expect(results.map((result) => result.text)).toEqual(["cc", "cc"]);
    expect(
      events
        .filter((event) => event.type === "generation_model_lane_wait")
        .map((event) => event.id),
    ).toEqual(["first", "second"]);
    expect(events).toContainEqual({
      type: "generation_batch_start",
      mode: "static",
      model: "tiny",
      ids: ["first", "second"],
      batchSize: 2,
      maxTokens: 2,
      maxTokensByRequest: [2, 2],
    });
  });

  test("does not enqueue continuous-eligible full-cache requests as static-only work", () => {
    using model = new StaticModel();
    const staticGeneration = createStaticTransformersGeneration(
      options(model),
      new ModelExecutionLane(1),
    );

    expect(staticGeneration.generate(request("continuous"))).toBeNull();
  });
});
