/**
 * Whisper attention layers.
 * @module
 */

import {
  formatShape,
  type MxArray,
  multiply,
  reshape,
  scaledDotProductAttention,
  transpose,
} from "@mlxts/core";
import { Linear, Module } from "@mlxts/nn";

import type { WhisperConfig } from "./types";

/** Multi-head attention used by Whisper encoder and decoder blocks. */
export class WhisperAttention extends Module {
  qProj: Linear;
  kProj: Linear;
  vProj: Linear;
  outProj: Linear;
  #hiddenSize: number;
  #numHeads: number;
  #headDim: number;
  #scale: number;

  constructor(hiddenSize: number, numHeads: number, headDim: number) {
    super();
    this.#hiddenSize = hiddenSize;
    this.#numHeads = numHeads;
    this.#headDim = headDim;
    this.#scale = headDim ** -0.25;
    this.qProj = new Linear(hiddenSize, hiddenSize, true);
    this.kProj = new Linear(hiddenSize, hiddenSize, false);
    this.vProj = new Linear(hiddenSize, hiddenSize, true);
    this.outProj = new Linear(hiddenSize, hiddenSize, true);
  }

  static encoder(config: WhisperConfig): WhisperAttention {
    return new WhisperAttention(config.dModel, config.encoderAttentionHeads, config.encoderHeadDim);
  }

  static decoder(config: WhisperConfig): WhisperAttention {
    return new WhisperAttention(config.dModel, config.decoderAttentionHeads, config.decoderHeadDim);
  }

  forward(hiddenStates: MxArray): MxArray {
    return this.run(hiddenStates);
  }

  run(
    hiddenStates: MxArray,
    options: {
      keyValueStates?: MxArray;
      causal?: boolean;
    } = {},
  ): MxArray {
    const [batch, queryLength, hiddenSize] = hiddenStates.shape;
    if (
      batch === undefined ||
      queryLength === undefined ||
      hiddenSize !== this.#hiddenSize ||
      hiddenStates.shape.length !== 3
    ) {
      throw new Error(
        `WhisperAttention.run: expected [batch, seq, ${this.#hiddenSize}], got ${formatShape(hiddenStates.shape)}.`,
      );
    }

    const keyValueStates = options.keyValueStates ?? hiddenStates;
    const keyValueLength = this.assertKeyValueStates(keyValueStates, batch);

    using queryProjection = this.qProj.forward(hiddenStates);
    using keyProjection = this.kProj.forward(keyValueStates);
    using valueProjection = this.vProj.forward(keyValueStates);
    using queryView = reshape(queryProjection, [batch, queryLength, this.#numHeads, this.#headDim]);
    using keyView = reshape(keyProjection, [batch, keyValueLength, this.#numHeads, this.#headDim]);
    using valueView = reshape(valueProjection, [
      batch,
      keyValueLength,
      this.#numHeads,
      this.#headDim,
    ]);
    using queryHeads = transpose(queryView, [0, 2, 1, 3]);
    using keyHeads = transpose(keyView, [0, 2, 1, 3]);
    using valueHeads = transpose(valueView, [0, 2, 1, 3]);
    using scaledQueries = multiply(queryHeads, this.#scale);
    using scaledKeys = multiply(keyHeads, this.#scale);
    using attentionOutput = scaledDotProductAttention(scaledQueries, scaledKeys, valueHeads, {
      scale: 1.0,
      maskMode: options.causal === true ? "causal" : "",
    });
    using transposedOutput = transpose(attentionOutput, [0, 2, 1, 3]);
    using mergedOutput = reshape(transposedOutput, [batch, queryLength, this.#hiddenSize]);
    return this.outProj.forward(mergedOutput);
  }

  private assertKeyValueStates(keyValueStates: MxArray, batch: number): number {
    const [keyBatch, keyValueLength, hiddenSize] = keyValueStates.shape;
    if (
      keyBatch !== batch ||
      keyValueLength === undefined ||
      hiddenSize !== this.#hiddenSize ||
      keyValueStates.shape.length !== 3
    ) {
      throw new Error(
        `WhisperAttention.run: expected key/value states [${batch}, seq, ${this.#hiddenSize}], got ${formatShape(keyValueStates.shape)}.`,
      );
    }
    return keyValueLength;
  }
}
