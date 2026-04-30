/**
 * CLIP self-attention.
 * @module
 */

import {
  formatShape,
  type MxArray,
  reshape,
  scaledDotProductAttention,
  transpose,
} from "@mlxts/core";
import { Linear, Module } from "@mlxts/nn";

import type { CLIPTextConfig } from "./types";

/** Causal multi-head self-attention for CLIP text encoders. */
export class CLIPAttention extends Module {
  qProj: Linear;
  kProj: Linear;
  vProj: Linear;
  outProj: Linear;
  #hiddenSize: number;
  #numHeads: number;
  #headDim: number;

  constructor(config: CLIPTextConfig) {
    super();
    this.#hiddenSize = config.hiddenSize;
    this.#numHeads = config.numAttentionHeads;
    this.#headDim = config.headDim;
    this.qProj = new Linear(config.hiddenSize, config.hiddenSize, true);
    this.kProj = new Linear(config.hiddenSize, config.hiddenSize, true);
    this.vProj = new Linear(config.hiddenSize, config.hiddenSize, true);
    this.outProj = new Linear(config.hiddenSize, config.hiddenSize, true);
  }

  forward(hiddenStates: MxArray): MxArray {
    return this.run(hiddenStates);
  }

  run(hiddenStates: MxArray): MxArray {
    const [batch, sequenceLength, hiddenSize] = hiddenStates.shape;
    if (
      batch === undefined ||
      sequenceLength === undefined ||
      hiddenSize !== this.#hiddenSize ||
      hiddenStates.shape.length !== 3
    ) {
      throw new Error(
        `CLIPAttention.run: expected [batch, seq, ${this.#hiddenSize}], got ${formatShape(hiddenStates.shape)}.`,
      );
    }

    using queryProjection = this.qProj.forward(hiddenStates);
    using keyProjection = this.kProj.forward(hiddenStates);
    using valueProjection = this.vProj.forward(hiddenStates);
    using queryView = reshape(queryProjection, [
      batch,
      sequenceLength,
      this.#numHeads,
      this.#headDim,
    ]);
    using keyView = reshape(keyProjection, [batch, sequenceLength, this.#numHeads, this.#headDim]);
    using valueView = reshape(valueProjection, [
      batch,
      sequenceLength,
      this.#numHeads,
      this.#headDim,
    ]);
    using queries = transpose(queryView, [0, 2, 1, 3]);
    using keys = transpose(keyView, [0, 2, 1, 3]);
    using values = transpose(valueView, [0, 2, 1, 3]);
    using attentionOutput = scaledDotProductAttention(queries, keys, values, {
      scale: this.#headDim ** -0.5,
      maskMode: "causal",
    });
    using transposedOutput = transpose(attentionOutput, [0, 2, 1, 3]);
    using mergedOutput = reshape(transposedOutput, [batch, sequenceLength, this.#hiddenSize]);
    return this.outProj.forward(mergedOutput);
  }
}
