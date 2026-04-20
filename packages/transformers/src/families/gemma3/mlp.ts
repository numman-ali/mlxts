/**
 * Gemma 3 text decoder MLP.
 * @module
 */

import type { MxArray } from "@mlxts/core";
import { Linear, Module } from "@mlxts/nn";

import { gegluApprox } from "../../infrastructure/gated-activations";
import type { Gemma3TextConfig } from "./types";

/** Gated GELU MLP used by Gemma 3 text layers. */
export class Gemma3MLP extends Module {
  gateProjection: Linear;
  upProjection: Linear;
  downProjection: Linear;

  constructor(config: Pick<Gemma3TextConfig, "hiddenSize" | "intermediateSize">) {
    super();
    this.gateProjection = new Linear(config.hiddenSize, config.intermediateSize, false);
    this.upProjection = new Linear(config.hiddenSize, config.intermediateSize, false);
    this.downProjection = new Linear(config.intermediateSize, config.hiddenSize, false);
  }

  forward(x: MxArray): MxArray {
    using gate = this.gateProjection.forward(x);
    using value = this.upProjection.forward(x);
    using activated = gegluApprox(gate, value);
    return this.downProjection.forward(activated);
  }
}
