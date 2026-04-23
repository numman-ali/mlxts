/**
 * Qwen 3.5 text MLP.
 * @module
 */

import type { MxArray } from "@mlxts/core";
import { formatShape } from "@mlxts/core";
import { Linear, Module, swiglu } from "@mlxts/nn";

import type { Qwen3_5TextConfig } from "./types";

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
