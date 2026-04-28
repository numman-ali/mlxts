import { describe, expect, test } from "bun:test";
import type { Tokenizer } from "@mlxts/tokenizers";
import type { CausalLM } from "../../types";
import { parseGenerationDefaults, resolveGenerationOptions } from "./defaults";

function fakeModel(defaults: CausalLM["config"]["generationDefaults"]): CausalLM {
  const config: CausalLM["config"] = {
    family: "llama",
    modelType: "llama",
    rawConfig: {},
    vocabSize: 8,
    hiddenSize: 4,
    numHiddenLayers: 1,
    ...(defaults === undefined ? {} : { generationDefaults: defaults }),
  };
  return {
    family: "llama",
    layerCount: 1,
    config,
    forward(): never {
      throw new Error("unused");
    },
    createCache(): never {
      throw new Error("unused");
    },
    parameters() {
      return {};
    },
    trainableParameters() {
      return {};
    },
    update() {},
    freeze() {
      return this;
    },
    unfreeze() {
      return this;
    },
    eval() {
      return this;
    },
    train() {
      return this;
    },
    [Symbol.dispose]() {},
  };
}

const tokenizer: Tokenizer = {
  vocabSize: 8,
  bosTokenId: undefined,
  eosTokenIds: [2],
  padTokenId: undefined,
  encode() {
    return [1];
  },
  encodeWithOffsets() {
    return { ids: [1] };
  },
  encodeBatch() {
    return [{ ids: [1] }];
  },
  decode() {
    return "";
  },
  decodeBatch() {
    return [""];
  },
};

describe("generation defaults", () => {
  test("parses checkpoint defaults and greedy mode", () => {
    expect(
      parseGenerationDefaults({
        do_sample: false,
        top_k: 40,
        top_p: 0.95,
        eos_token_id: [1, 4],
      }),
    ).toEqual({
      temperature: 0,
      topK: 40,
      topP: 0.95,
      eosTokenIds: [1, 4],
    });
  });

  test("parses model-native sampled checkpoint defaults", () => {
    expect(
      parseGenerationDefaults({
        do_sample: true,
        temperature: 1.0,
        top_k: 20,
        top_p: 0.95,
      }),
    ).toEqual({
      temperature: 1.0,
      topK: 20,
      topP: 0.95,
    });
  });

  test("merges model defaults with tokenizer eos ids", () => {
    const resolved = resolveGenerationOptions(
      fakeModel({
        temperature: 0.7,
        topP: 0.9,
        eosTokenIds: [7],
      }),
      tokenizer,
      { maxTokens: 3 },
    );

    expect(resolved.temperature).toBe(0.7);
    expect(resolved.topP).toBe(0.9);
    expect(resolved.eosTokenIds).toEqual([7, 2]);
  });

  test("explicit options override checkpoint defaults", () => {
    const resolved = resolveGenerationOptions(
      fakeModel({
        temperature: 0.7,
        topP: 0.9,
        eosTokenIds: [7],
      }),
      tokenizer,
      { maxTokens: 3, temperature: 0, eosTokenIds: [9] },
    );

    expect(resolved.temperature).toBe(0);
    expect(resolved.eosTokenIds).toEqual([9]);
  });
});
