/**
 * Qwen 3.5 text MLP.
 * @module
 */

import {
  add,
  formatShape,
  type MxArray,
  matmul,
  multiply,
  reshape,
  sigmoid,
  transpose,
  zeros,
} from "@mlxts/core";
import { Linear, Module, swiglu } from "@mlxts/nn";
import type { RoutedTopK } from "../../infrastructure/moe";
import { PackedSwitchGLUExperts, topKFromRouterLogits } from "../../infrastructure/moe";

import type { Qwen3_5TextConfig } from "./types";

/** Feed-forward layer variant selected by the Qwen text config. */
export type Qwen3_5TextFeedForward = Qwen3_5TextMLP | Qwen3_5TextMoE;

function requireMoeNumber(value: number | null, name: string): number {
  if (value === null) {
    throw new Error(`Qwen3_5TextMoE: ${name} is required for MoE feed-forward layers.`);
  }
  return value;
}

/** SwiGLU MLP used by Qwen 3.5 text layers. */
export class Qwen3_5TextMLP extends Module {
  gateProjection: Linear;
  upProjection: Linear;
  downProjection: Linear;
  #hiddenSize: number;

  constructor(config: Qwen3_5TextConfig) {
    super();
    this.gateProjection = new Linear(config.hiddenSize, config.intermediateSize, false);
    this.upProjection = new Linear(config.hiddenSize, config.intermediateSize, false);
    this.downProjection = new Linear(config.intermediateSize, config.hiddenSize, false);
    this.#hiddenSize = config.hiddenSize;
  }

  forward(x: MxArray): MxArray {
    const lastDimension = x.shape[x.shape.length - 1];
    if (lastDimension !== this.#hiddenSize) {
      throw new Error(
        `Qwen3_5TextMLP.forward: expected input last dimension ${this.#hiddenSize}, got ${lastDimension ?? "undefined"} for shape ${formatShape(x.shape)}.`,
      );
    }

    using gate = this.gateProjection.forward(x);
    using value = this.upProjection.forward(x);
    using activated = swiglu(gate, value);
    return this.downProjection.forward(activated);
  }
}

/** Top-k router used by Qwen 3.5/3.6 MoE text layers. */
export class Qwen3_5TextTopKRouter extends Module {
  weight: MxArray;
  #numExperts: number;
  #numExpertsPerToken: number;

  constructor(config: Qwen3_5TextConfig) {
    super();
    this.#numExperts = requireMoeNumber(config.numExperts, "numExperts");
    this.#numExpertsPerToken = requireMoeNumber(config.numExpertsPerToken, "numExpertsPerToken");
    this.weight = zeros([this.#numExperts, config.hiddenSize]);
  }

  forward(hiddenStates: MxArray): MxArray {
    using transposedWeight = transpose(this.weight);
    return matmul(hiddenStates, transposedWeight);
  }

  route(hiddenStates: MxArray): RoutedTopK {
    using logits = this.forward(hiddenStates);
    return topKFromRouterLogits(
      logits,
      this.#numExpertsPerToken,
      this.#numExperts,
      "Qwen3_5TextTopKRouter.forward",
      { precise: true },
    );
  }
}

/** Routed Qwen 3.5/3.6 MoE block with a gated shared expert. */
export class Qwen3_5TextMoE extends Module {
  gate: Qwen3_5TextTopKRouter;
  experts: PackedSwitchGLUExperts;
  sharedExpert: Qwen3_5TextMLP;
  sharedExpertGate: Linear;
  #hiddenSize: number;

  constructor(config: Qwen3_5TextConfig) {
    super();
    this.#hiddenSize = config.hiddenSize;
    this.gate = new Qwen3_5TextTopKRouter(config);
    this.experts = new PackedSwitchGLUExperts(
      requireMoeNumber(config.numExperts, "numExperts"),
      config.hiddenSize,
      requireMoeNumber(config.moeIntermediateSize, "moeIntermediateSize"),
      swiglu,
    );
    this.sharedExpert = new Qwen3_5TextMLP({
      ...config,
      intermediateSize: requireMoeNumber(
        config.sharedExpertIntermediateSize,
        "sharedExpertIntermediateSize",
      ),
      feedForwardKind: "dense",
      moeIntermediateSize: null,
      sharedExpertIntermediateSize: null,
      numExperts: null,
      numExpertsPerToken: null,
      routerAuxLossCoef: null,
    });
    this.sharedExpertGate = new Linear(config.hiddenSize, 1, false);
  }

  forward(x: MxArray): MxArray {
    const [batchSize, sequenceLength, hiddenSize] = x.shape;
    if (
      x.shape.length !== 3 ||
      batchSize === undefined ||
      sequenceLength === undefined ||
      hiddenSize !== this.#hiddenSize
    ) {
      throw new Error(
        `Qwen3_5TextMoE.forward: expected hidden states [batch, seq, ${this.#hiddenSize}], got ${formatShape(x.shape)}.`,
      );
    }

    using flatHidden = reshape(x, [batchSize * sequenceLength, this.#hiddenSize]);
    using sharedExpertOutput = this.sharedExpert.forward(flatHidden);
    const routing = this.gate.route(flatHidden);
    try {
      using expertOutput = this.experts.forward(flatHidden, routing.indices, routing.weights);
      using sharedGateLogits = this.sharedExpertGate.forward(flatHidden);
      using sharedGate = sigmoid(sharedGateLogits);
      using gatedSharedExpert = multiply(sharedGate, sharedExpertOutput);
      using combined = add(expertOutput, gatedSharedExpert);
      return reshape(combined, [batchSize, sequenceLength, this.#hiddenSize]);
    } finally {
      routing.indices.free();
      routing.weights.free();
    }
  }
}

/** Create the dense or MoE Qwen feed-forward module described by the config. */
export function createQwen3_5TextFeedForward(config: Qwen3_5TextConfig): Qwen3_5TextFeedForward {
  return config.feedForwardKind === "moe" ? new Qwen3_5TextMoE(config) : new Qwen3_5TextMLP(config);
}
