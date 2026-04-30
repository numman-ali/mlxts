/**
 * CLIP encoder block.
 * @module
 */

import { add, type MxArray } from "@mlxts/core";
import { LayerNorm, Module } from "@mlxts/nn";

import { CLIPAttention } from "./attention";
import { CLIPMLP } from "./mlp";
import type { CLIPTextConfig } from "./types";

/** Transformer encoder block used by CLIP text models. */
export class CLIPEncoderBlock extends Module {
  layerNorm1: LayerNorm;
  layerNorm2: LayerNorm;
  selfAttention: CLIPAttention;
  mlp: CLIPMLP;

  constructor(config: CLIPTextConfig) {
    super();
    this.layerNorm1 = new LayerNorm(config.hiddenSize, config.layerNormEps);
    this.layerNorm2 = new LayerNorm(config.hiddenSize, config.layerNormEps);
    this.selfAttention = new CLIPAttention(config);
    this.mlp = new CLIPMLP(config);
  }

  forward(hiddenStates: MxArray): MxArray {
    return this.run(hiddenStates);
  }

  run(hiddenStates: MxArray): MxArray {
    using normalizedAttentionInput = this.layerNorm1.forward(hiddenStates);
    using attentionOutput = this.selfAttention.run(normalizedAttentionInput);
    const attended = add(hiddenStates, attentionOutput);
    using residual = attended;
    using normalizedMlpInput = this.layerNorm2.forward(residual);
    using mlpOutput = this.mlp.forward(normalizedMlpInput);
    return add(residual, mlpOutput);
  }
}
