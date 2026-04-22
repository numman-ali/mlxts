/**
 * Gemma 4 dense MLP.
 * @module
 */

import { formatShape, type MxArray } from "@mlxts/core";
import { Linear, LoRALinear, Module, QuantizedLinear } from "@mlxts/nn";

import { gegluApprox } from "../../infrastructure/gated-activations";
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

    const gateProjection = this.runtimeProjection("gateProjection");
    const upProjection = this.runtimeProjection("upProjection");
    const downProjection = this.runtimeProjection("downProjection");

    if (
      gateProjection instanceof Linear &&
      upProjection instanceof Linear &&
      downProjection instanceof Linear
    ) {
      return runMlp(x, gateProjection.weight, upProjection.weight, downProjection.weight);
    }

    using gate = gateProjection.forward(x);
    using value = upProjection.forward(x);
    using activated = gegluApprox(gate, value);
    return downProjection.forward(activated);
  }

  private runtimeProjection(
    key: "gateProjection" | "upProjection" | "downProjection",
  ): Linear | QuantizedLinear | LoRALinear {
    const projection: unknown = Reflect.get(this, key);
    if (
      projection instanceof Linear ||
      projection instanceof QuantizedLinear ||
      projection instanceof LoRALinear
    ) {
      return projection;
    }
    throw new Error(`Gemma4TextMLP.forward: ${key} is not a supported linear module.`);
  }
}
