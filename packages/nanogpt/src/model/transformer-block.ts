/**
 * Transformer block — the repeating unit of GPT.
 *
 * Uses pre-norm architecture (GPT-2 style):
 *   x = x + attention(layerNorm1(x))
 *   x = x + mlp(layerNorm2(x))
 *
 * @module
 */

import type { MxArray } from "@mlxts/core";
import { add } from "@mlxts/core";
import { checkpoint as checkpointModule, LayerNorm, Module } from "@mlxts/nn";

import type { GPTConfig } from "../config";
import { CausalSelfAttention } from "./causal-self-attention";
import { MLP } from "./mlp";

/** Single transformer block with pre-norm residual connections. */
export class TransformerBlock extends Module {
  layerNorm1: LayerNorm;
  attention: CausalSelfAttention;
  layerNorm2: LayerNorm;
  mlp: MLP;
  #checkpointedForward: ((x: MxArray) => MxArray) | null;

  constructor(config: GPTConfig) {
    super();
    this.layerNorm1 = new LayerNorm(config.nEmbd);
    this.attention = new CausalSelfAttention(config);
    this.layerNorm2 = new LayerNorm(config.nEmbd);
    this.mlp = new MLP(config);
    this.#checkpointedForward =
      config.gradientCheckpointing === true
        ? checkpointModule(this, (x) => this.forwardUnchecked(x))
        : null;
  }

  /**
   * Pre-norm transformer block with residual connections.
   *
   * @param x - Input tensor of shape [batch, sequence, nEmbd].
   * @returns Output tensor of same shape.
   */
  forward(x: MxArray): MxArray {
    if (this.isTraining && this.#checkpointedForward !== null) {
      return this.#checkpointedForward(x);
    }
    return this.forwardUnchecked(x);
  }

  private forwardUnchecked(x: MxArray): MxArray {
    // Attention with residual: x = x + attention(ln1(x))
    using norm1 = this.layerNorm1.forward(x);
    using attnOut = this.attention.forward(norm1);
    using afterAttn = add(x, attnOut);

    // MLP with residual: x = x + mlp(ln2(x))
    using norm2 = this.layerNorm2.forward(afterAttn);
    using mlpOut = this.mlp.forward(norm2);
    return add(afterAttn, mlpOut);
  }
}
