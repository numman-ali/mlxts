/**
 * Qwen 3.5 linear attention (GatedDeltaNet).
 * @module
 */

import {
  add,
  asType,
  concatenate,
  exp,
  fastRmsNorm,
  formatShape,
  greater,
  log,
  type MxArray,
  multiply,
  ones,
  repeat,
  reshape,
  retainArray,
  sigmoid,
  slice,
  stack,
  subtract,
  sum,
  where,
  zeros,
} from "@mlxts/core";
import { Conv1d, Linear, Module, silu } from "@mlxts/nn";

import type { Qwen3_5TextCache } from "./cache";
import { Qwen3_5RMSNormGated } from "./norm";
import type { Qwen3_5TextConfig } from "./types";

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

function sliceSequenceStep(x: MxArray, step: number): MxArray {
  const [batchSize, sequenceLength] = x.shape;
  if (
    batchSize === undefined ||
    sequenceLength === undefined ||
    x.shape.length < 2 ||
    step < 0 ||
    step >= sequenceLength
  ) {
    throw new Error(
      `Qwen3_5GatedDeltaNet: cannot take sequence step ${step} from shape ${formatShape(x.shape)}.`,
    );
  }

  const rank = x.shape.length;
  const startIndices = Array(rank).fill(0);
  const stopIndices = [...x.shape];
  startIndices[1] = step;
  stopIndices[1] = step + 1;
  using stepView = slice(x, startIndices, stopIndices);
  const targetShape = [batchSize, ...x.shape.slice(2)];
  return reshape(stepView, targetShape);
}

function repeatHeads(x: MxArray, repeatFactor: number): MxArray {
  return repeatFactor === 1 ? x : repeat(x, repeatFactor, 2);
}

function stableSoftplus(x: MxArray): MxArray {
  using expX = exp(x);
  using expPlusOne = add(expX, 1);
  using slowPath = log(expPlusOne);
  using mask = greater(x, 20);
  return where(mask, x, slowPath);
}

function decayFactors(a: MxArray, aLog: MxArray, dtBias: MxArray): MxArray {
  using floatA = asType(a, "float32");
  using floatDtBias = asType(dtBias, "float32");
  using shifted = add(floatA, floatDtBias);
  using softplus = stableSoftplus(shifted);
  using floatALog = asType(aLog, "float32");
  using expALog = exp(floatALog);
  using scaled = multiply(expALog, softplus);
  using negated = multiply(scaled, -1);
  return exp(negated);
}

function gatedDeltaStep(
  q: MxArray,
  k: MxArray,
  v: MxArray,
  g: MxArray,
  beta: MxArray,
  state: MxArray,
): { output: MxArray; state: MxArray } {
  const keyHeadDim = k.shape[2] ?? 0;
  const queryHeadDim = q.shape[2] ?? 0;
  using decay = reshape(g, [...g.shape, 1, 1]);
  using decayedState = multiply(state, decay);
  using keyView = reshape(k, [k.shape[0] ?? 0, k.shape[1] ?? 0, 1, keyHeadDim]);
  using decayedStateTimesKey = multiply(decayedState, keyView);
  using kvMemory = sum(decayedStateTimesKey, 3);
  using deltaBase = subtract(v, kvMemory);
  using betaView = reshape(beta, [...beta.shape, 1]);
  using delta = multiply(deltaBase, betaView);
  using updateKeyView = reshape(k, [k.shape[0] ?? 0, k.shape[1] ?? 0, 1, keyHeadDim]);
  using deltaView = reshape(delta, [...delta.shape, 1]);
  using update = multiply(updateKeyView, deltaView);
  using nextState = add(decayedState, update);
  using queryView = reshape(q, [q.shape[0] ?? 0, q.shape[1] ?? 0, 1, queryHeadDim]);
  using nextStateTimesQuery = multiply(nextState, queryView);
  const output = sum(nextStateTimesQuery, 3);
  return {
    output,
    state: retainArray(nextState),
  };
}

export function gatedDeltaSequence(
  q: MxArray,
  k: MxArray,
  v: MxArray,
  g: MxArray,
  beta: MxArray,
  initialState: MxArray,
): { output: MxArray; state: MxArray } {
  const sequenceLength = q.shape[1];
  if (sequenceLength === undefined) {
    throw new Error("Qwen3_5GatedDeltaNet: q is missing a sequence dimension.");
  }

  const outputs: MxArray[] = [];
  let state: MxArray | null = retainArray(initialState);
  try {
    for (let step = 0; step < sequenceLength; step += 1) {
      if (state === null) {
        throw new Error("Qwen3_5GatedDeltaNet: recurrent state was unexpectedly released.");
      }
      using qStep = sliceSequenceStep(q, step);
      using kStep = sliceSequenceStep(k, step);
      using vStep = sliceSequenceStep(v, step);
      using gStep = sliceSequenceStep(g, step);
      using betaStep = sliceSequenceStep(beta, step);
      const next = gatedDeltaStep(qStep, kStep, vStep, gStep, betaStep, state);
      state.free();
      state = next.state;
      outputs.push(next.output);
    }

    const output = stack(outputs, 1);
    if (state === null) {
      throw new Error("Qwen3_5GatedDeltaNet: recurrent state was unexpectedly released.");
    }
    const finalState = retainArray(state);
    state.free();
    state = null;
    return {
      output,
      state: finalState,
    };
  } catch (error) {
    state?.free();
    throw error;
  } finally {
    for (const output of outputs) {
      output.free();
    }
  }
}

/** Linear-attention token mixer used by Qwen 3.5 text layers. */
export class Qwen3_5GatedDeltaNet extends Module {
  inProjectionQkv: Linear;
  inProjectionZ: Linear;
  inProjectionB: Linear;
  inProjectionA: Linear;
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

  run(x: MxArray, layerIndex: number, cache?: Qwen3_5TextCache): MxArray {
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
    const convStateLength = Math.max(0, this.#convKernelSize - 1);

    using projectedQkv = this.inProjectionQkv.forward(x);
    using projectedZ = this.inProjectionZ.forward(x);
    using projectedB = this.inProjectionB.forward(x);
    using projectedA = this.inProjectionA.forward(x);
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
      const repeatFactor = this.#numValueHeads / this.#numKeyHeads;
      using repeatedQueries = repeatHeads(normalizedQueries, repeatFactor);
      using repeatedKeys = repeatHeads(normalizedKeys, repeatFactor);
      using scaledQueries = multiply(repeatedQueries, this.#keyHeadDim ** -1);
      using scaledKeys = multiply(repeatedKeys, this.#keyHeadDim ** -0.5);
      using floatQueries = asType(scaledQueries, "float32");
      using floatKeys = asType(scaledKeys, "float32");
      using floatValues = asType(values, "float32");
      using beta = sigmoid(projectedB);
      using g = decayFactors(projectedA, this.aLog, this.dtBias);
      using floatBeta = asType(beta, "float32");
      using floatG = asType(g, "float32");

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
        const next = gatedDeltaSequence(
          floatQueries,
          floatKeys,
          floatValues,
          floatG,
          floatBeta,
          initialState,
        );
        try {
          using recurrentOutput = asType(next.output, x.dtype);
          using gatedOutput = this.norm.forward(recurrentOutput, z);
          using mergedOutput = reshape(gatedOutput, [batchSize, sequenceLength, this.#valueDim]);
          if (cache !== undefined) {
            let nextConvState: MxArray | null = null;
            if (convStateLength > 0) {
              nextConvState = slice(
                convInput,
                [0, sequenceLength, 0],
                [batchSize, sequenceLength + convStateLength, this.#convDim],
              );
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
      createdConvState?.free();
    }
  }
}
