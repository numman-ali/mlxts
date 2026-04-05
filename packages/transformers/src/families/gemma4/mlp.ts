/**
 * Gemma 4 dense MLP.
 * @module
 */

import type { MxArray } from "@mlxts/core";
import { geluApprox, multiply } from "@mlxts/core";
import { Linear, Module } from "@mlxts/nn";
import type { Gemma4TextConfig } from "./types";

function layerUsesDoubleWideMlp(config: Gemma4TextConfig, layerIndex: number): boolean {
  const firstSharedLayerIndex = config.numHiddenLayers - config.numKvSharedLayers;
  return (
    config.useDoubleWideMLP && firstSharedLayerIndex > 0 && layerIndex >= firstSharedLayerIndex
  );
}

/** Gated GELU MLP used by Gemma 4 dense text layers. */
export class Gemma4TextMLP extends Module {
  gateProjection: Linear;
  upProjection: Linear;
  downProjection: Linear;

  constructor(config: Gemma4TextConfig, layerIndex: number) {
    super();
    const intermediateSize = layerUsesDoubleWideMlp(config, layerIndex)
      ? config.intermediateSize * 2
      : config.intermediateSize;
    this.gateProjection = new Linear(config.hiddenSize, intermediateSize, false);
    this.upProjection = new Linear(config.hiddenSize, intermediateSize, false);
    this.downProjection = new Linear(intermediateSize, config.hiddenSize, false);
  }

  forward(x: MxArray): MxArray {
    using gate = this.gateProjection.forward(x);
    using value = this.upProjection.forward(x);
    using geluResult = geluApprox(gate);
    using activated = multiply(geluResult, value);
    return this.downProjection.forward(activated);
  }
}
