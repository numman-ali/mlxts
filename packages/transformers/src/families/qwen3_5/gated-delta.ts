/**
 * Qwen 3.5 linear attention (GatedDeltaNet).
 * @module
 */

import {
  add,
  asType,
  compile,
  concatenate,
  contiguous,
  exp,
  expandDims,
  fastRmsNorm,
  formatShape,
  greater,
  log,
  type MxArray,
  multiply,
  ones,
  reshape,
  retainArray,
  sigmoid,
  slice,
  where,
  zeros,
} from "@mlxts/core";
import { Conv1d, fuseQuantizedLinears, Linear, Module, QuantizedLinear, silu } from "@mlxts/nn";

import { Qwen3_5TextBatchCache } from "./batch-cache";
import type { Qwen3_5TextCache } from "./cache";
import { gatedDeltaSequenceFromKeyHeads } from "./gated-delta-recurrence";
import { Qwen3_5RMSNormGated } from "./norm";
import type { Qwen3_5TextConfig } from "./types";

export { gatedDeltaSequence, gatedDeltaSequenceFromKeyHeads } from "./gated-delta-recurrence";

function takeLastAxisRange(x: MxArray, start: number, end: number): MxArray {
  const rank = x.shape.length;
  if (rank < 3) {
    throw new Error(
      `Qwen3_5GatedDeltaNet: expected rank >= 3 projection tensor, got ${formatShape(x.shape)}.`,
    );
  }
  const startIndices = Array(rank).fill(0);
  const stopIndices = [...x.shape];
  startIndices[rank - 1] = start;
  stopIndices[rank - 1] = end;
  return slice(x, startIndices, stopIndices);
}

function stableSoftplus(x: MxArray): MxArray {
  using expX = exp(x);
  using expPlusOne = add(expX, 1);
  using slowPath = log(expPlusOne);
  using mask = greater(x, 20);
  return where(mask, x, slowPath);
}

function applySequenceMask(x: MxArray, mask: MxArray | null): MxArray {
  if (mask === null) {
    return retainArray(x);
  }
  using featureMask = expandDims(mask, 2);
  using zero = zeros([...x.shape], x.dtype);
  return where(featureMask, x, zero);
}

const decayFactorsTransform = compile(
  (a: MxArray, aLog: MxArray, dtBias: MxArray) => {
    using floatA = asType(a, "float32");
    using floatDtBias = asType(dtBias, "float32");
    using shifted = add(floatA, floatDtBias);
    using softplus = stableSoftplus(shifted);
    using floatALog = asType(aLog, "float32");
    using expALog = exp(floatALog);
    using scaled = multiply(expALog, softplus);
    using negated = multiply(scaled, -1);
    return exp(negated);
  },
  { shapeless: true },
);

function decayFactors(a: MxArray, aLog: MxArray, dtBias: MxArray): MxArray {
  return decayFactorsTransform(a, aLog, dtBias);
}

type FusedProjectionBAState = {
  module: QuantizedLinear;
  bWeight: MxArray;
  bScales: MxArray;
  bBiases: MxArray | null;
  aWeight: MxArray;
  aScales: MxArray;
  aBiases: MxArray | null;
};

/** Linear-attention token mixer used by Qwen 3.5 text layers. */
export class Qwen3_5GatedDeltaNet extends Module {
  inProjectionQkv: Linear | QuantizedLinear;
  inProjectionZ: Linear | QuantizedLinear;
  inProjectionB: Linear | QuantizedLinear;
  inProjectionA: Linear | QuantizedLinear;
  conv1d: Conv1d;
  dtBias: MxArray;
  aLog: MxArray;
  norm: Qwen3_5RMSNormGated;
  outProjection: Linear;
  #hiddenSize: number;
  #numKeyHeads: number;
  #numValueHeads: number;
  #keyHeadDim: number;
  #valueHeadDim: number;
  #keyDim: number;
  #valueDim: number;
  #convKernelSize: number;
  #convDim: number;
  #fusedProjectionBA: FusedProjectionBAState | null = null;

  constructor(config: Qwen3_5TextConfig) {
    super();
    this.#hiddenSize = config.hiddenSize;
    this.#numKeyHeads = config.linearNumKeyHeads;
    this.#numValueHeads = config.linearNumValueHeads;
    this.#keyHeadDim = config.linearKeyHeadDim;
    this.#valueHeadDim = config.linearValueHeadDim;
    this.#keyDim = this.#numKeyHeads * this.#keyHeadDim;
    this.#valueDim = this.#numValueHeads * this.#valueHeadDim;
    this.#convKernelSize = config.linearConvKernelDim;
    this.#convDim = this.#keyDim * 2 + this.#valueDim;

    if (this.#numValueHeads % this.#numKeyHeads !== 0) {
      throw new Error(
        `Qwen3_5GatedDeltaNet: linearNumValueHeads ${this.#numValueHeads} must be divisible by linearNumKeyHeads ${this.#numKeyHeads}.`,
      );
    }

    this.inProjectionQkv = new Linear(config.hiddenSize, this.#convDim, false);
    this.inProjectionZ = new Linear(config.hiddenSize, this.#valueDim, false);
    this.inProjectionB = new Linear(config.hiddenSize, this.#numValueHeads, false);
    this.inProjectionA = new Linear(config.hiddenSize, this.#numValueHeads, false);
    this.conv1d = new Conv1d(
      this.#convDim,
      this.#convDim,
      this.#convKernelSize,
      1,
      0,
      1,
      this.#convDim,
      false,
    );
    this.dtBias = ones([this.#numValueHeads]);
    this.aLog = ones([this.#numValueHeads]);
    this.norm = new Qwen3_5RMSNormGated(this.#valueHeadDim, config.rmsNormEps);
    this.outProjection = new Linear(this.#valueDim, config.hiddenSize, false);
  }

  forward(x: MxArray): MxArray {
    return this.run(x, 0);
  }

  run(x: MxArray, layerIndex: number, cache?: Qwen3_5TextCache | Qwen3_5TextBatchCache): MxArray {
    const [batchSize, sequenceLength, hiddenSize] = x.shape;
    if (
      batchSize === undefined ||
      sequenceLength === undefined ||
      hiddenSize === undefined ||
      x.shape.length !== 3
    ) {
      throw new Error(
        `Qwen3_5GatedDeltaNet.forward: expected rank-3 input, got ${formatShape(x.shape)}.`,
      );
    }
    if (hiddenSize !== this.#hiddenSize) {
      throw new Error(
        `Qwen3_5GatedDeltaNet.forward: expected hidden size ${this.#hiddenSize}, got ${hiddenSize}.`,
      );
    }

    const linearState = cache?.linearState(layerIndex);
    const linearMask =
      cache instanceof Qwen3_5TextBatchCache ? cache.linearAttentionMask(sequenceLength) : null;
    const convStateLength = Math.max(0, this.#convKernelSize - 1);

    try {
      using rawProjectedQkv = this.inProjectionQkv.forward(x);
      using projectedQkv = applySequenceMask(rawProjectedQkv, linearMask);
      using projectedZ = this.inProjectionZ.forward(x);
      const projectedBA = this.projectBA(x);
      using z = reshape(projectedZ, [
        batchSize,
        sequenceLength,
        this.#numValueHeads,
        this.#valueHeadDim,
      ]);

      let createdConvState: MxArray | null = null;
      const activeConvState =
        linearState?.convState ??
        (() => {
          if (convStateLength === 0) {
            return null;
          }
          createdConvState = zeros([batchSize, convStateLength, this.#convDim], x.dtype);
          return createdConvState;
        })();

      try {
        using convInput =
          activeConvState === null
            ? retainArray(projectedQkv)
            : concatenate([activeConvState, projectedQkv], 1);
        using rawConvOutput = this.conv1d.forward(convInput);
        using convOutput = silu(rawConvOutput);
        using querySlice = takeLastAxisRange(convOutput, 0, this.#keyDim);
        using keySlice = takeLastAxisRange(convOutput, this.#keyDim, this.#keyDim * 2);
        using valueSlice = takeLastAxisRange(convOutput, this.#keyDim * 2, this.#convDim);
        using queries = reshape(querySlice, [
          batchSize,
          sequenceLength,
          this.#numKeyHeads,
          this.#keyHeadDim,
        ]);
        using keys = reshape(keySlice, [
          batchSize,
          sequenceLength,
          this.#numKeyHeads,
          this.#keyHeadDim,
        ]);
        using values = reshape(valueSlice, [
          batchSize,
          sequenceLength,
          this.#numValueHeads,
          this.#valueHeadDim,
        ]);
        using normalizedQueries = fastRmsNorm(queries, undefined, { eps: 1e-6 });
        using normalizedKeys = fastRmsNorm(keys, undefined, { eps: 1e-6 });
        using scaledQueries = multiply(normalizedQueries, this.#keyHeadDim ** -1);
        using scaledKeys = multiply(normalizedKeys, this.#keyHeadDim ** -0.5);
        using beta = sigmoid(projectedBA.b);
        using g = decayFactors(projectedBA.a, this.aLog, this.dtBias);

        let createdRecurrentState: MxArray | null = null;
        const initialState =
          linearState?.recurrentState ??
          (() => {
            createdRecurrentState = zeros(
              [batchSize, this.#numValueHeads, this.#valueHeadDim, this.#keyHeadDim],
              "float32",
            );
            return createdRecurrentState;
          })();

        try {
          const next = gatedDeltaSequenceFromKeyHeads(
            scaledQueries,
            scaledKeys,
            values,
            g,
            beta,
            initialState,
            linearMask ?? undefined,
          );
          try {
            using recurrentOutput = asType(next.output, x.dtype);
            using gatedOutput = this.norm.forward(recurrentOutput, z);
            using mergedOutput = reshape(gatedOutput, [batchSize, sequenceLength, this.#valueDim]);
            if (cache !== undefined) {
              let nextConvState: MxArray | null = null;
              if (convStateLength > 0) {
                using nextConvStateView = slice(
                  convInput,
                  [0, sequenceLength, 0],
                  [batchSize, sequenceLength + convStateLength, this.#convDim],
                );
                nextConvState = contiguous(nextConvStateView);
              }
              cache.updateLinearState(layerIndex, nextConvState, next.state);
              nextConvState?.free();
            }
            return this.outProjection.forward(mergedOutput);
          } finally {
            next.output.free();
            next.state.free();
          }
        } finally {
          createdRecurrentState?.free();
        }
      } finally {
        projectedBA.b.free();
        projectedBA.a.free();
        createdConvState?.free();
      }
    } finally {
      linearMask?.free();
    }
  }

  private fusedProjectionBA(): QuantizedLinear | null {
    if (this.isTraining) {
      return null;
    }
    if (
      !(this.inProjectionB instanceof QuantizedLinear) ||
      !(this.inProjectionA instanceof QuantizedLinear)
    ) {
      this.disposeFusedProjectionBA();
      return null;
    }

    const cached = this.#fusedProjectionBA;
    if (
      cached !== null &&
      cached.bWeight === this.inProjectionB.weight &&
      cached.bScales === this.inProjectionB.scales &&
      cached.bBiases === this.inProjectionB.biases &&
      cached.aWeight === this.inProjectionA.weight &&
      cached.aScales === this.inProjectionA.scales &&
      cached.aBiases === this.inProjectionA.biases
    ) {
      return cached.module;
    }

    this.disposeFusedProjectionBA();
    const fused = fuseQuantizedLinears([this.inProjectionB, this.inProjectionA]);
    if (fused === null) {
      return null;
    }
    this.#fusedProjectionBA = {
      module: fused,
      bWeight: this.inProjectionB.weight,
      bScales: this.inProjectionB.scales,
      bBiases: this.inProjectionB.biases,
      aWeight: this.inProjectionA.weight,
      aScales: this.inProjectionA.scales,
      aBiases: this.inProjectionA.biases,
    };
    return fused;
  }

  private projectBA(x: MxArray): { b: MxArray; a: MxArray } {
    const fused = this.fusedProjectionBA();
    if (fused === null) {
      return {
        b: this.inProjectionB.forward(x),
        a: this.inProjectionA.forward(x),
      };
    }

    using projectedBA = fused.forward(x);
    return {
      b: takeLastAxisRange(projectedBA, 0, this.#numValueHeads),
      a: takeLastAxisRange(projectedBA, this.#numValueHeads, this.#numValueHeads * 2),
    };
  }

  private disposeFusedProjectionBA(): void {
    this.#fusedProjectionBA?.module[Symbol.dispose]();
    this.#fusedProjectionBA = null;
  }

  override [Symbol.dispose](): void {
    this.disposeFusedProjectionBA();
    super[Symbol.dispose]();
  }
}
