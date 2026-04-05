import { describe, expect, test } from "bun:test";
import { array, fastRmsNorm } from "@mlxts/core";
import { LlamaLikeNorm } from "./norm";

const BASE_CONFIG = {
  hiddenSize: 4,
  rmsNormEps: 1e-6,
};

describe("LlamaLikeNorm", () => {
  test("matches fastRmsNorm when weight offsets are disabled", () => {
    using norm = new LlamaLikeNorm({ ...BASE_CONFIG, normWeightOffset: false });
    using hidden = array([[1, 2, 3, 4]], "float32");
    using expected = fastRmsNorm(hidden, norm.weight, { eps: BASE_CONFIG.rmsNormEps });
    using actual = norm.forward(hidden);

    expect(actual.toList()).toEqual(expected.toList());
  });

  test("applies and refreshes the cached offset weight when weight offsets are enabled", () => {
    using norm = new LlamaLikeNorm({ ...BASE_CONFIG, normWeightOffset: true });
    using hidden = array([[1, 2, 3, 4]], "float32");

    using expectedInitialWeight = array([1, 1, 1, 1], "float32");
    using expectedInitial = fastRmsNorm(hidden, expectedInitialWeight, {
      eps: BASE_CONFIG.rmsNormEps,
    });
    using actualInitial = norm.forward(hidden);
    expect(actualInitial.toList()).toEqual(expectedInitial.toList());

    const previousWeight = norm.weight;
    norm.weight = array([1, 1, 1, 1], "float32");
    previousWeight.free();

    using expectedUpdatedWeight = array([2, 2, 2, 2], "float32");
    using expectedUpdated = fastRmsNorm(hidden, expectedUpdatedWeight, {
      eps: BASE_CONFIG.rmsNormEps,
    });
    using actualUpdated = norm.forward(hidden);
    expect(actualUpdated.toList()).toEqual(expectedUpdated.toList());
  });
});
