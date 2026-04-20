import { describe, expect, test } from "bun:test";
import type { ParameterTree } from "@mlxts/core";
import { array, type MxArray, retainArray, zeros } from "@mlxts/core";
import type { CausalLM, TransformerCache } from "../../types";
import {
  inputTensor,
  prefillPromptCache,
  takeLastLogits,
  validatePrefillStepSize,
} from "./helpers";

class FakeCache implements TransformerCache {
  readonly layerCount = 0;
  readonly offset = 0;

  #stateArrays: MxArray[];

  constructor(stateArrays: MxArray[]) {
    this.#stateArrays = stateArrays;
  }

  updateAndFetch(): { keys: MxArray; values: MxArray } {
    throw new Error("FakeCache.updateAndFetch should not be called in generation helper tests.");
  }

  advance(): void {}

  isEmpty(): boolean {
    return this.#stateArrays.length === 0;
  }

  arrays(): MxArray[] {
    return this.#stateArrays.map((stateArray) => retainArray(stateArray));
  }

  [Symbol.dispose](): void {
    for (const array of this.#stateArrays) {
      array.free();
    }
    this.#stateArrays = [];
  }
}

class FakeModel implements CausalLM {
  readonly family = "llama";
  readonly layerCount = 0;
  readonly config = {
    family: "llama",
    modelType: "llama",
    rawConfig: {},
    vocabSize: 4,
    hiddenSize: 4,
    numHiddenLayers: 0,
  } as const;

  forwardCallCount = 0;
  #logitsFactory: () => MxArray;

  constructor(logitsFactory: () => MxArray) {
    this.#logitsFactory = logitsFactory;
  }

  forward(): MxArray {
    this.forwardCallCount += 1;
    return this.#logitsFactory();
  }

  createCache(): TransformerCache {
    return new FakeCache([]);
  }

  parameters(): ParameterTree {
    return {};
  }

  trainableParameters(): ParameterTree {
    return {};
  }

  update(): void {}

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

describe("generation helpers", () => {
  test("inputTensor wraps numeric prompts and preserves tensor prompts", () => {
    using wrapped = inputTensor([1, 2, 3]);
    expect(wrapped.dtype).toBe("int32");
    expect(wrapped.shape).toEqual([1, 3]);
    expect(wrapped.toList()).toEqual([[1, 2, 3]]);

    using existing = array([[4, 5]], "int32");
    expect(inputTensor(existing)).toBe(existing);
  });

  test("takeLastLogits slices the final sequence position on device", () => {
    using logits = array(
      [
        [
          [1, 2, 3],
          [4, 5, 6],
        ],
      ],
      "float32",
    );
    using lastLogits = takeLastLogits(logits, "generation test");
    expect(lastLogits.shape).toEqual([1, 3]);
    expect(lastLogits.toList()).toEqual([[4, 5, 6]]);
  });

  test("takeLastLogits and validatePrefillStepSize reject malformed inputs", () => {
    using badLogits = array([[1, 2, 3]], "float32");
    expect(() => takeLastLogits(badLogits, "generation test")).toThrow(
      "model logits must have shape [batch, seq, vocab]",
    );
    expect(() => validatePrefillStepSize(0, "generation test")).toThrow(
      "prefillStepSize must be a positive integer",
    );
    expect(() => validatePrefillStepSize(1.5, "generation test")).toThrow(
      "prefillStepSize must be a positive integer",
    );
  });

  test("prefillPromptCache evaluates logits directly when the cache has no state arrays", () => {
    using cache = new FakeCache([]);
    const model = new FakeModel(() =>
      array(
        [
          [
            [1, 2],
            [3, 4],
          ],
        ],
        "float32",
      ),
    );

    const tail = prefillPromptCache(model, [1, 2, 3], cache, 2);

    expect(model.forwardCallCount).toBe(1);
    expect(tail).toEqual([3]);
  });

  test("prefillPromptCache materializes cache state arrays for multi-chunk prompts", () => {
    using cacheState = zeros([1], "float32");
    using cache = new FakeCache([cacheState]);
    const model = new FakeModel(() =>
      array(
        [
          [
            [1, 2],
            [3, 4],
          ],
        ],
        "float32",
      ),
    );

    const tail = prefillPromptCache(model, [1, 2, 3, 4, 5], cache, 2);

    expect(model.forwardCallCount).toBe(2);
    expect(tail).toEqual([5]);
  });
});
