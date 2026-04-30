/**
 * T5 feed-forward layers.
 * @module
 */

import { geluApprox, type MxArray, multiply } from "@mlxts/core";
import { gelu, Linear, Module, relu, silu } from "@mlxts/nn";

import type { T5DenseActivation, T5EncoderConfig } from "./types";

function activate(hiddenStates: MxArray, activation: T5DenseActivation): MxArray {
  switch (activation) {
    case "relu":
      return relu(hiddenStates);
    case "gelu":
      return gelu(hiddenStates);
    case "gelu_new":
      return geluApprox(hiddenStates);
    case "silu":
      return silu(hiddenStates);
  }
}

/** Dense activation used by T5 encoder feed-forward blocks. */
export class T5DenseActivationLayer extends Module {
  wi: Linear | null;
  wi0: Linear | null;
  wi1: Linear | null;
  wo: Linear;
  #activation: T5DenseActivation;
  #isGated: boolean;

  constructor(config: T5EncoderConfig) {
    super();
    this.#activation = config.denseActivation;
    this.#isGated = config.isGatedActivation;
    this.wi = this.#isGated ? null : new Linear(config.dModel, config.dFf, false);
    this.wi0 = this.#isGated ? new Linear(config.dModel, config.dFf, false) : null;
    this.wi1 = this.#isGated ? new Linear(config.dModel, config.dFf, false) : null;
    this.wo = new Linear(config.dFf, config.dModel, false);
  }

  forward(hiddenStates: MxArray): MxArray {
    if (this.#isGated) {
      if (this.wi0 === null || this.wi1 === null) {
        throw new Error("T5DenseActivationLayer.forward: gated projections are not initialized.");
      }
      using gateProjection = this.wi0.forward(hiddenStates);
      using gate = activate(gateProjection, this.#activation);
      using value = this.wi1.forward(hiddenStates);
      using activated = multiply(gate, value);
      return this.wo.forward(activated);
    }

    if (this.wi === null) {
      throw new Error("T5DenseActivationLayer.forward: dense projection is not initialized.");
    }
    using projected = this.wi.forward(hiddenStates);
    using activated = activate(projected, this.#activation);
    return this.wo.forward(activated);
  }
}
