/**
 * CLIP feed-forward layers.
 * @module
 */

import { type MxArray, multiply, sigmoid } from "@mlxts/core";
import { gelu, Linear, Module } from "@mlxts/nn";

import type { CLIPTextConfig } from "./types";

function quickGelu(x: MxArray): MxArray {
  using scaled = multiply(x, 1.702);
  using gate = sigmoid(scaled);
  return multiply(x, gate);
}

/** Feed-forward network used by CLIP text encoder blocks. */
export class CLIPMLP extends Module {
  fc1: Linear;
  fc2: Linear;
  #hiddenAct: CLIPTextConfig["hiddenAct"];

  constructor(config: CLIPTextConfig) {
    super();
    this.fc1 = new Linear(config.hiddenSize, config.intermediateSize, true);
    this.fc2 = new Linear(config.intermediateSize, config.hiddenSize, true);
    this.#hiddenAct = config.hiddenAct;
  }

  forward(hiddenStates: MxArray): MxArray {
    using projected = this.fc1.forward(hiddenStates);
    using activated = this.#hiddenAct === "quick_gelu" ? quickGelu(projected) : gelu(projected);
    return this.fc2.forward(activated);
  }
}
