/**
 * T5 encoder self-attention.
 * @module
 */

import {
  array,
  expandDims,
  formatShape,
  type MxArray,
  reshape,
  scaledDotProductAttention,
  transpose,
} from "@mlxts/core";
import { Embedding, Linear, Module } from "@mlxts/nn";

import type { T5EncoderConfig } from "./types";

export function t5RelativePositionBucket(
  relativePosition: number,
  bidirectional: boolean,
  numBuckets: number,
  maxDistance: number,
): number {
  let buckets = 0;
  let availableBuckets = numBuckets;
  let distance = relativePosition;
  if (bidirectional) {
    availableBuckets = Math.floor(availableBuckets / 2);
    if (distance > 0) {
      buckets += availableBuckets;
    }
    distance = Math.abs(distance);
  } else {
    distance = Math.max(0, -distance);
  }

  const maxExact = Math.floor(availableBuckets / 2);
  if (distance < maxExact) {
    return buckets + distance;
  }

  const scaled =
    maxExact +
    Math.floor(
      (Math.log(distance / maxExact) / Math.log(maxDistance / maxExact)) *
        (availableBuckets - maxExact),
    );
  return buckets + Math.min(scaled, availableBuckets - 1);
}

function relativePositionBuckets(
  queryLength: number,
  keyLength: number,
  numBuckets: number,
  maxDistance: number,
): number[][] {
  return Array.from({ length: queryLength }, (_, query) =>
    Array.from({ length: keyLength }, (_, key) =>
      t5RelativePositionBucket(key - query, true, numBuckets, maxDistance),
    ),
  );
}

/** Multi-head self-attention used by T5 encoder layers. */
export class T5Attention extends Module {
  q: Linear;
  k: Linear;
  v: Linear;
  o: Linear;
  relativeAttentionBias: Embedding | null;
  #dModel: number;
  #dKv: number;
  #numHeads: number;
  #innerDim: number;
  #relativeAttentionNumBuckets: number;
  #relativeAttentionMaxDistance: number;

  constructor(config: T5EncoderConfig, hasRelativeAttentionBias: boolean) {
    super();
    this.#dModel = config.dModel;
    this.#dKv = config.dKv;
    this.#numHeads = config.numHeads;
    this.#innerDim = config.innerDim;
    this.#relativeAttentionNumBuckets = config.relativeAttentionNumBuckets;
    this.#relativeAttentionMaxDistance = config.relativeAttentionMaxDistance;
    this.q = new Linear(config.dModel, config.innerDim, false);
    this.k = new Linear(config.dModel, config.innerDim, false);
    this.v = new Linear(config.dModel, config.innerDim, false);
    this.o = new Linear(config.innerDim, config.dModel, false);
    this.relativeAttentionBias = hasRelativeAttentionBias
      ? new Embedding(config.relativeAttentionNumBuckets, config.numHeads)
      : null;
  }

  forward(hiddenStates: MxArray, positionBias: MxArray): MxArray {
    return this.run(hiddenStates, positionBias);
  }

  positionBias(queryLength: number, keyLength: number): MxArray {
    if (this.relativeAttentionBias === null) {
      throw new Error("T5Attention.positionBias: relative attention bias is not owned here.");
    }

    using bucketIds = array(
      relativePositionBuckets(
        queryLength,
        keyLength,
        this.#relativeAttentionNumBuckets,
        this.#relativeAttentionMaxDistance,
      ),
      "int32",
    );
    using bucketEmbeddings = this.relativeAttentionBias.forward(bucketIds);
    using transposed = transpose(bucketEmbeddings, [2, 0, 1]);
    return expandDims(transposed, 0);
  }

  run(hiddenStates: MxArray, positionBias: MxArray): MxArray {
    const [batch, sequenceLength, hiddenSize] = hiddenStates.shape;
    if (
      batch === undefined ||
      sequenceLength === undefined ||
      hiddenSize !== this.#dModel ||
      hiddenStates.shape.length !== 3
    ) {
      throw new Error(
        `T5Attention.run: expected [batch, seq, ${this.#dModel}], got ${formatShape(hiddenStates.shape)}.`,
      );
    }

    using queryProjection = this.q.forward(hiddenStates);
    using keyProjection = this.k.forward(hiddenStates);
    using valueProjection = this.v.forward(hiddenStates);
    using queryView = reshape(queryProjection, [batch, sequenceLength, this.#numHeads, this.#dKv]);
    using keyView = reshape(keyProjection, [batch, sequenceLength, this.#numHeads, this.#dKv]);
    using valueView = reshape(valueProjection, [batch, sequenceLength, this.#numHeads, this.#dKv]);
    using queries = transpose(queryView, [0, 2, 1, 3]);
    using keys = transpose(keyView, [0, 2, 1, 3]);
    using values = transpose(valueView, [0, 2, 1, 3]);
    using attentionOutput = scaledDotProductAttention(queries, keys, values, {
      scale: 1.0,
      maskArray: positionBias,
    });
    using transposedOutput = transpose(attentionOutput, [0, 2, 1, 3]);
    using mergedOutput = reshape(transposedOutput, [batch, sequenceLength, this.#innerDim]);
    return this.o.forward(mergedOutput);
  }
}
