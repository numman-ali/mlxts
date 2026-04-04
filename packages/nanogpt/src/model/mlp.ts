/**
 * GPT feed-forward network (MLP block).
 *
 * Two linear layers with GELU activation and 4x expansion,
 * following the GPT-2 architecture.
 *
 * @module
 */

import type { MxArray } from "mlx-ts";
import { Dropout, gelu, Linear, Module } from "mlx-ts";
import type { GPTConfig } from "../config";

/** Feed-forward MLP block: expand → GELU → contract → dropout. */
export class MLP extends Module {
  expandProjection: Linear;
  contractProjection: Linear;
  dropout: Dropout;

  constructor(config: GPTConfig) {
    super();
    this.expandProjection = new Linear(config.nEmbd, 4 * config.nEmbd);
    this.contractProjection = new Linear(4 * config.nEmbd, config.nEmbd);
    this.dropout = new Dropout(config.dropout);
  }

  /**
   * @param x - Input tensor of shape [batch, sequence, nEmbd].
   * @returns Output tensor of same shape.
   */
  forward(x: MxArray): MxArray {
    using expanded = this.expandProjection.forward(x);
    using activated = gelu(expanded);
    using contracted = this.contractProjection.forward(activated);
    return this.dropout.forward(contracted);
  }
}
