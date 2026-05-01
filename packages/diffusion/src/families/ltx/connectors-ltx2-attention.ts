import {
  expandDims,
  type MxArray,
  multiply,
  reshape,
  retainArray,
  scaledDotProductAttention,
  sigmoid,
  transpose,
} from "@mlxts/core";
import { Linear, Module, RMSNorm } from "@mlxts/nn";

import type { Ltx2RopeType } from "./config";
import { applyLtx2ConnectorRotary } from "./connectors-ltx2-rotary";
import type { LtxRotaryEmbeddings } from "./embeddings";
import { assertSequence3d } from "./tensor-utils";

type Ltx2ConnectorAttentionProjection = {
  queries: MxArray;
  keys: MxArray;
  values: MxArray;
  gateLogits: MxArray | null;
};

function disposeProjection(projection: Ltx2ConnectorAttentionProjection): void {
  projection.queries.free();
  projection.keys.free();
  projection.values.free();
  projection.gateLogits?.free();
}

/** LTX-2 attention used by text connector blocks. */
export class Ltx2ConnectorAttention extends Module {
  normQ: RMSNorm;
  normK: RMSNorm;
  toQ: Linear;
  toK: Linear;
  toV: Linear;
  toOut: Linear;
  toGateLogits: Linear | null;
  #ropeType: Ltx2RopeType;
  #heads: number;
  #headDim: number;
  #hiddenSize: number;

  constructor(options: {
    hiddenSize: number;
    heads: number;
    headDim: number;
    attentionBias?: boolean;
    attentionOutBias?: boolean;
    gatedAttention?: boolean;
    ropeType: Ltx2RopeType;
  }) {
    super();
    if (options.hiddenSize !== options.heads * options.headDim) {
      throw new Error("Ltx2ConnectorAttention: hiddenSize must equal heads * headDim.");
    }
    this.normQ = new RMSNorm(options.hiddenSize, 1e-6);
    this.normK = new RMSNorm(options.hiddenSize, 1e-6);
    this.toQ = new Linear(options.hiddenSize, options.hiddenSize, options.attentionBias ?? true);
    this.toK = new Linear(options.hiddenSize, options.hiddenSize, options.attentionBias ?? true);
    this.toV = new Linear(options.hiddenSize, options.hiddenSize, options.attentionBias ?? true);
    this.toOut = new Linear(
      options.hiddenSize,
      options.hiddenSize,
      options.attentionOutBias ?? true,
    );
    this.toGateLogits =
      options.gatedAttention === true ? new Linear(options.hiddenSize, options.heads) : null;
    this.#ropeType = options.ropeType;
    this.#heads = options.heads;
    this.#headDim = options.headDim;
    this.#hiddenSize = options.hiddenSize;
  }

  forward(hiddenStates: MxArray): MxArray {
    return this.run(hiddenStates);
  }

  /** Run self-attention over connector text embeddings. */
  run(
    hiddenStates: MxArray,
    options: { mask?: MxArray; rotary?: LtxRotaryEmbeddings } = {},
  ): MxArray {
    const shape = assertSequence3d(hiddenStates, "Ltx2ConnectorAttention.run hiddenStates");
    if (shape.channels !== this.#hiddenSize) {
      throw new Error("Ltx2ConnectorAttention.run: hidden size mismatch.");
    }
    const projection = this.#project(hiddenStates, options.rotary);
    try {
      const attended = scaledDotProductAttention(
        projection.queries,
        projection.keys,
        projection.values,
        options.mask === undefined
          ? { scale: this.#headDim ** -0.5 }
          : { scale: this.#headDim ** -0.5, maskMode: "array", maskArray: options.mask },
      );
      try {
        const gated = this.#applyGate(attended, projection.gateLogits);
        try {
          using sequenceFirst = transpose(gated, [0, 2, 1, 3]);
          using merged = reshape(sequenceFirst, [shape.batch, shape.length, this.#hiddenSize]);
          return this.toOut.forward(merged);
        } finally {
          gated.free();
        }
      } finally {
        attended.free();
      }
    } finally {
      disposeProjection(projection);
    }
  }

  #applyGate(attended: MxArray, gateLogits: MxArray | null): MxArray {
    if (gateLogits === null) {
      return retainArray(attended);
    }
    using gate = sigmoid(gateLogits);
    using doubled = multiply(gate, 2);
    using headsLast = transpose(doubled, [0, 2, 1]);
    using gateScale = expandDims(headsLast, -1);
    return multiply(attended, gateScale);
  }

  #project(hiddenStates: MxArray, rotary?: LtxRotaryEmbeddings): Ltx2ConnectorAttentionProjection {
    const shape = assertSequence3d(hiddenStates, "Ltx2ConnectorAttention.project hiddenStates");
    using queryProjection = this.toQ.forward(hiddenStates);
    using keyProjection = this.toK.forward(hiddenStates);
    using valueProjection = this.toV.forward(hiddenStates);
    using normalizedQueries = this.normQ.forward(queryProjection);
    using normalizedKeys = this.normK.forward(keyProjection);
    const queries =
      rotary === undefined
        ? retainArray(normalizedQueries)
        : applyLtx2ConnectorRotary(normalizedQueries, rotary, this.#ropeType, this.#heads);
    const keys =
      rotary === undefined
        ? retainArray(normalizedKeys)
        : applyLtx2ConnectorRotary(normalizedKeys, rotary, this.#ropeType, this.#heads);
    try {
      using queryHeads = reshape(queries, [shape.batch, shape.length, this.#heads, this.#headDim]);
      using keyHeads = reshape(keys, [shape.batch, shape.length, this.#heads, this.#headDim]);
      using valueHeads = reshape(valueProjection, [
        shape.batch,
        shape.length,
        this.#heads,
        this.#headDim,
      ]);
      const gateLogits =
        this.toGateLogits === null ? null : this.toGateLogits.forward(hiddenStates);
      return {
        queries: transpose(queryHeads, [0, 2, 1, 3]),
        keys: transpose(keyHeads, [0, 2, 1, 3]),
        values: transpose(valueHeads, [0, 2, 1, 3]),
        gateLogits,
      };
    } finally {
      queries.free();
      keys.free();
    }
  }
}
