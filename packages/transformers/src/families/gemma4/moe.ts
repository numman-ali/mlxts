/**
 * Gemma 4 Mixture-of-Experts routing helpers.
 * @module
 */

import { type MxArray, multiply, ones, softmax, takeAxis } from "@mlxts/core";
import { Linear, Module } from "@mlxts/nn";
import { gegluApprox } from "../../infrastructure/gated-activations";
import type { RoutedTopK } from "../../infrastructure/moe";
import { PackedSwitchGLUExperts, topKFromRouterProbabilities } from "../../infrastructure/moe";
import { Gemma4RMSNorm } from "./norm";
import type { Gemma4TextConfig } from "./types";

function requireMoeNumber(value: number | null, name: string): number {
  if (value === null) {
    throw new Error(`Gemma4TextRouter: ${name} is required for MoE blocks.`);
  }
  return value;
}

/** Router used by Gemma 4 A4B text MoE blocks. */
export class Gemma4TextRouter extends Module {
  norm: Gemma4RMSNorm;
  proj: Linear;
  scale: MxArray;
  perExpertScale: MxArray;
  #topKExperts: number;
  #numExperts: number;
  #scalarRootSize: number;

  constructor(config: Gemma4TextConfig) {
    super();
    this.#numExperts = requireMoeNumber(config.numExperts, "numExperts");
    this.#topKExperts = requireMoeNumber(config.topKExperts, "topKExperts");
    this.#scalarRootSize = config.hiddenSize ** -0.5;
    this.norm = new Gemma4RMSNorm(config.hiddenSize, config.rmsNormEps, false);
    this.proj = new Linear(config.hiddenSize, this.#numExperts, false);
    this.scale = ones([config.hiddenSize]);
    this.perExpertScale = ones([this.#numExperts]);
  }

  forward(hiddenStates: MxArray): MxArray {
    using normalized = this.norm.forward(hiddenStates);
    using scaledByParameter = multiply(normalized, this.scale);
    using scaled = multiply(scaledByParameter, this.#scalarRootSize);
    return this.proj.forward(scaled);
  }

  route(hiddenStates: MxArray): RoutedTopK {
    using logits = this.forward(hiddenStates);
    using probabilities = softmax(logits, -1);
    const routing = topKFromRouterProbabilities(
      probabilities,
      this.#topKExperts,
      this.#numExperts,
      "Gemma4TextRouter.forward",
    );
    try {
      using expertScale = takeAxis(this.perExpertScale, routing.indices, 0);
      const weights = multiply(routing.weights, expertScale);
      routing.weights.free();
      return { indices: routing.indices, weights };
    } catch (error) {
      routing.indices.free();
      routing.weights.free();
      throw error;
    }
  }
}

/** Create Gemma 4's packed routed expert bank. */
export function createGemma4TextExperts(config: Gemma4TextConfig): PackedSwitchGLUExperts {
  return new PackedSwitchGLUExperts(
    requireMoeNumber(config.numExperts, "numExperts"),
    config.hiddenSize,
    requireMoeNumber(config.moeIntermediateSize, "moeIntermediateSize"),
    gegluApprox,
  );
}
