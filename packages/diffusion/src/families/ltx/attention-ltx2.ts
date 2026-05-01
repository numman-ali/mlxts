import {
  expandDims,
  fastRmsNorm,
  formatShape,
  type MxArray,
  multiply,
  reshape,
  retainArray,
  scaledDotProductAttention,
  sigmoid,
  transpose,
} from "@mlxts/core";
import { Linear, Module } from "@mlxts/nn";

import type { Ltx2RopeType } from "./config";
import { applyLtx2ConnectorRotary } from "./connectors-ltx2-rotary";
import type { LtxRotaryEmbeddings } from "./embeddings";
import { assertSequence3d } from "./tensor-utils";

export type Ltx2AttentionOptions = {
  encoderHiddenStates?: MxArray;
  attentionMask?: MxArray;
  queryRotaryEmbeddings?: LtxRotaryEmbeddings;
  keyRotaryEmbeddings?: LtxRotaryEmbeddings;
};

type Ltx2AttentionProjection = {
  queries: MxArray;
  keys: MxArray;
  values: MxArray;
  gateLogits: MxArray | null;
};

function disposeProjection(projection: Ltx2AttentionProjection): void {
  projection.queries.free();
  projection.keys.free();
  projection.values.free();
  projection.gateLogits?.free();
}

/** LTX-2 attention with optional cross-source keys, RoPE, and per-head gates. */
export class Ltx2Attention extends Module {
  toQ: Linear;
  toK: Linear;
  toV: Linear;
  toOut: Linear;
  toGateLogits: Linear | null;
  #queryDim: number;
  #crossAttentionDim: number;
  #heads: number;
  #headDim: number;
  #innerDim: number;
  #ropeType: Ltx2RopeType;
  #normEps: number;

  constructor(options: {
    queryDim: number;
    heads: number;
    headDim: number;
    crossAttentionDim?: number;
    attentionBias: boolean;
    attentionOutBias: boolean;
    ropeType: Ltx2RopeType;
    normEps: number;
    gatedAttention: boolean;
  }) {
    super();
    this.#queryDim = options.queryDim;
    this.#crossAttentionDim = options.crossAttentionDim ?? options.queryDim;
    this.#heads = options.heads;
    this.#headDim = options.headDim;
    this.#innerDim = options.heads * options.headDim;
    this.#ropeType = options.ropeType;
    this.#normEps = options.normEps;
    this.toQ = new Linear(options.queryDim, this.#innerDim, options.attentionBias);
    this.toK = new Linear(this.#crossAttentionDim, this.#innerDim, options.attentionBias);
    this.toV = new Linear(this.#crossAttentionDim, this.#innerDim, options.attentionBias);
    this.toOut = new Linear(this.#innerDim, options.queryDim, options.attentionOutBias);
    this.toGateLogits = options.gatedAttention ? new Linear(options.queryDim, options.heads) : null;
  }

  forward(hiddenStates: MxArray): MxArray {
    return this.run(hiddenStates);
  }

  /** Run non-causal LTX-2 attention over query and optional context streams. */
  run(hiddenStates: MxArray, options: Ltx2AttentionOptions = {}): MxArray {
    const shape = assertSequence3d(hiddenStates, "Ltx2Attention.run hiddenStates");
    if (shape.channels !== this.#queryDim) {
      throw new Error(
        `Ltx2Attention.run: hiddenStates channels must be ${this.#queryDim}, got ${shape.channels}.`,
      );
    }
    const context = options.encoderHiddenStates ?? hiddenStates;
    const contextShape = assertSequence3d(context, "Ltx2Attention.run encoderHiddenStates");
    if (contextShape.batch !== shape.batch || contextShape.channels !== this.#crossAttentionDim) {
      throw new Error(
        `Ltx2Attention.run: encoderHiddenStates must have shape [${shape.batch}, length, ${this.#crossAttentionDim}], got ${formatShape(
          context.shape,
        )}.`,
      );
    }
    const projection = this.#project(hiddenStates, context, options);
    try {
      const attended = scaledDotProductAttention(
        projection.queries,
        projection.keys,
        projection.values,
        options.attentionMask === undefined
          ? { scale: this.#headDim ** -0.5 }
          : { scale: this.#headDim ** -0.5, maskMode: "array", maskArray: options.attentionMask },
      );
      try {
        const gated = this.#applyGate(attended, projection.gateLogits);
        try {
          using sequenceFirst = transpose(gated, [0, 2, 1, 3]);
          using merged = reshape(sequenceFirst, [shape.batch, shape.length, this.#innerDim]);
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
    using headsFirst = transpose(doubled, [0, 2, 1]);
    using scale = expandDims(headsFirst, -1);
    return multiply(attended, scale);
  }

  #project(
    querySource: MxArray,
    keyValueSource: MxArray,
    options: Ltx2AttentionOptions,
  ): Ltx2AttentionProjection {
    const queryShape = assertSequence3d(querySource, "Ltx2Attention.project querySource");
    const keyValueShape = assertSequence3d(keyValueSource, "Ltx2Attention.project keyValueSource");
    using queryProjection = this.toQ.forward(querySource);
    using keyProjection = this.toK.forward(keyValueSource);
    using valueProjection = this.toV.forward(keyValueSource);
    using normalizedQueries = fastRmsNorm(queryProjection, undefined, { eps: this.#normEps });
    using normalizedKeys = fastRmsNorm(keyProjection, undefined, { eps: this.#normEps });
    const queries =
      options.queryRotaryEmbeddings === undefined
        ? retainArray(normalizedQueries)
        : applyLtx2ConnectorRotary(
            normalizedQueries,
            options.queryRotaryEmbeddings,
            this.#ropeType,
            this.#heads,
          );
    const keyRotary = options.keyRotaryEmbeddings ?? options.queryRotaryEmbeddings;
    const keys =
      keyRotary === undefined
        ? retainArray(normalizedKeys)
        : applyLtx2ConnectorRotary(normalizedKeys, keyRotary, this.#ropeType, this.#heads);
    try {
      using queryHeads = reshape(queries, [
        queryShape.batch,
        queryShape.length,
        this.#heads,
        this.#headDim,
      ]);
      using keyHeads = reshape(keys, [
        keyValueShape.batch,
        keyValueShape.length,
        this.#heads,
        this.#headDim,
      ]);
      using valueHeads = reshape(valueProjection, [
        keyValueShape.batch,
        keyValueShape.length,
        this.#heads,
        this.#headDim,
      ]);
      const gateLogits = this.toGateLogits === null ? null : this.toGateLogits.forward(querySource);
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
