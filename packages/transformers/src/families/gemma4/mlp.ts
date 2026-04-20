/**
 * Gemma 4 dense MLP.
 * @module
 */

import { formatShape, type MxArray } from "@mlxts/core";
import { Linear, Module } from "@mlxts/nn";
import { runMlp } from "./runtime/mlp";
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
  #hiddenSize: number;

  constructor(config: Gemma4TextConfig, layerIndex: number) {
    super();
    const intermediateSize = layerUsesDoubleWideMlp(config, layerIndex)
      ? config.intermediateSize * 2
      : config.intermediateSize;
    this.#hiddenSize = config.hiddenSize;
    this.gateProjection = new Linear(config.hiddenSize, intermediateSize, false);
    this.upProjection = new Linear(config.hiddenSize, intermediateSize, false);
    this.downProjection = new Linear(intermediateSize, config.hiddenSize, false);
  }

  forward(x: MxArray): MxArray {
    const lastDimension = x.shape[x.shape.length - 1];
    if (lastDimension !== this.#hiddenSize) {
      throw new Error(
        `Gemma4TextMLP.forward: expected input last dimension ${this.#hiddenSize}, got ${lastDimension ?? "undefined"} for shape ${formatShape(x.shape)}.`,
      );
    }

    return runMlp(
      x,
      this.gateProjection.weight,
      this.upProjection.weight,
      this.downProjection.weight,
    );
  }
}
