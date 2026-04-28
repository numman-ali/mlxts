import { describe, expect, test } from "bun:test";

import { array, multiply } from "@mlxts/core";
import { PackedSwitchGLUExperts, topKFromRouterProbabilities } from "./moe";

describe("MoE infrastructure", () => {
  test("selects and normalizes top-k router probabilities", () => {
    using probabilities = array(
      [
        [0.1, 0.7, 0.2],
        [0.6, 0.3, 0.1],
      ],
      "float32",
    );
    const routing = topKFromRouterProbabilities(probabilities, 2, 3);
    try {
      expect(routing.indices.shape).toEqual([2, 2]);
      expect(routing.weights.shape).toEqual([2, 2]);
      const weights = routing.weights.toList() as number[][];
      const first = weights[0];
      const second = weights[1];
      if (first === undefined || second === undefined) {
        throw new Error("Expected two routed rows.");
      }
      expect((first[0] ?? 0) + (first[1] ?? 0)).toBeCloseTo(1);
      expect((second[0] ?? 0) + (second[1] ?? 0)).toBeCloseTo(1);
    } finally {
      routing.indices.free();
      routing.weights.free();
    }
  });

  test("runs packed SwitchGLU experts with repeated expert assignments", () => {
    using experts = new PackedSwitchGLUExperts(2, 1, 1, multiply);
    experts.gateUpProjection.free();
    experts.downProjection.free();
    experts.gateUpProjection = array(
      [
        [[1], [1]],
        [[2], [1]],
      ],
      "float32",
    );
    experts.downProjection = array([[[1]], [[1]]], "float32");

    using hidden = array([[2], [3]], "float32");
    using indices = array(
      [
        [0, 1],
        [1, 0],
      ],
      "int32",
    );
    using weights = array(
      [
        [0.25, 0.75],
        [1, 0],
      ],
      "float32",
    );
    using output = experts.forward(hidden, indices, weights);

    expect(output.toList()).toEqual([[7], [18]]);
  });
});
