import { describe, expect, test } from "bun:test";

import { array, full } from "@mlxts/core";

import { SamplerState } from "./index";

describe("sampling", () => {
  test("SamplerState uses argmax when temperature is zero or negative", () => {
    using logits = array([[0.2, 1.5, 0.1]], "float32");
    using state = new SamplerState([], {});
    using zeroTemperature = state.sampleTokenTensor(logits, { temperature: 0 });
    using negativeTemperature = state.sampleTokenTensor(logits, { temperature: -1 });

    expect(zeroTemperature.item()).toBe(1);
    expect(negativeTemperature.item()).toBe(1);
  });

  test("SamplerState applies repetition penalty on the device", () => {
    using logits = array([[4, 3, 1]], "float32");
    using state = new SamplerState([0], { repetitionPenalty: 2 });
    using token = state.sampleTokenTensor(logits, {
      temperature: 0,
      repetitionPenalty: 2,
    });

    expect(token.item()).toBe(1);
  });

  test("SamplerState applies top-k pruning on the device", () => {
    using logits = array([[1, 5, 3, 4]], "float32");
    using state = new SamplerState([], {});
    using token = state.sampleTokenTensor(logits, {
      temperature: 1,
      topK: 1,
      seed: 0,
    });

    expect(token.item()).toBe(1);
  });

  test("SamplerState combines repetition penalty with top-k filtering", () => {
    using logits = array([[4, 3, 1]], "float32");
    using state = new SamplerState([0], { repetitionPenalty: 1.2 });
    using token = state.sampleTokenTensor(logits, {
      temperature: 1,
      topK: 1,
      repetitionPenalty: 1.2,
      seed: 0,
    });

    expect(token.item()).toBe(0);
  });

  test("SamplerState keeps top-k behavior correct across vocab shapes for the same config", () => {
    using firstLogits = array([[1, 5, 3, 4]], "float32");
    using secondLogits = array([[1, 5, 3, 4, 2]], "float32");
    using state = new SamplerState([], {});
    using firstToken = state.sampleTokenTensor(firstLogits, {
      temperature: 1,
      topK: 1,
      seed: 0,
    });
    using secondToken = state.sampleTokenTensor(secondLogits, {
      temperature: 1,
      topK: 1,
      seed: 0,
    });

    expect(firstToken.item()).toBe(1);
    expect(secondToken.item()).toBe(1);
  });

  test("SamplerState handles Gemma-scale top-k vocabularies", () => {
    using logits = full([1, 262144], 0, "float32");
    using state = new SamplerState([], {});
    using token = state.sampleTokenTensor(logits, {
      temperature: 1,
      topK: 64,
      seed: 0,
    });

    expect(token.shape).toEqual([1, 1]);
    expect(Number.isInteger(token.item())).toBe(true);
  });

  test("SamplerState applies top-p pruning on the device", () => {
    using logits = array([[8, 2, 1]], "float32");
    using state = new SamplerState([], {});
    using token = state.sampleTokenTensor(logits, {
      temperature: 1,
      topP: 0.1,
      seed: 0,
    });

    expect(token.item()).toBe(0);
  });

  test("SamplerState keeps top-p behavior correct across vocab shapes for the same config", () => {
    using firstLogits = array([[8, 2, 1]], "float32");
    using secondLogits = array([[9, 2, 1, 0]], "float32");
    using state = new SamplerState([], {});
    using firstToken = state.sampleTokenTensor(firstLogits, {
      temperature: 1,
      topP: 0.1,
      seed: 0,
    });
    using secondToken = state.sampleTokenTensor(secondLogits, {
      temperature: 1,
      topP: 0.1,
      seed: 0,
    });

    expect(firstToken.item()).toBe(0);
    expect(secondToken.item()).toBe(0);
  });

  test("SamplerState applies top-p before temperature scaling", () => {
    using logits = array([[10, 9, 8]], "float32");
    using state = new SamplerState([], {});
    using token = state.sampleTokenTensor(logits, {
      temperature: 10,
      topP: 0.2,
      seed: 0,
    });

    expect(token.item()).toBe(0);
  });

  test("SamplerState applies min-p pruning on the device", () => {
    using logits = array([[4, 2, 0]], "float32");
    using state = new SamplerState([], {});
    using token = state.sampleTokenTensor(logits, {
      temperature: 1,
      minP: 0.5,
      seed: 0,
    });

    expect(token.item()).toBe(0);
  });

  test("SamplerState applies min-p before temperature scaling", () => {
    using logits = array([[10, 9, 0]], "float32");
    using state = new SamplerState([], {});
    using token = state.sampleTokenTensor(logits, {
      temperature: 10,
      minP: 0.5,
      seed: 0,
    });

    expect(token.item()).toBe(0);
  });

  test("SamplerState appends generated tokens into repetition-penalty history", () => {
    using logits = array([[4, 3, 1]], "float32");
    using state = new SamplerState([], { repetitionPenalty: 2 });
    using firstToken = array([[0]], "int32");
    state.appendToken(firstToken);
    using nextToken = state.sampleTokenTensor(logits, {
      temperature: 0,
      repetitionPenalty: 2,
    });

    expect(nextToken.item()).toBe(1);
  });
});
