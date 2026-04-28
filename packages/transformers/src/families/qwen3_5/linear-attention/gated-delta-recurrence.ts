/**
 * Qwen 3.5 gated-delta recurrence helpers.
 * @module
 */

import {
  add,
  asType,
  expandDims,
  fast,
  formatShape,
  isMetalAvailable,
  type MxArray,
  multiply,
  repeat,
  reshape,
  retainArray,
  slice,
  stack,
  subtract,
  sum,
  where,
  zeros,
} from "@mlxts/core";

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

let metalAvailable: boolean | null = null;

function canUseMetal(): boolean {
  metalAvailable ??= isMetalAvailable();
  return metalAvailable;
}

function gatedDeltaStep(
  q: MxArray,
  k: MxArray,
  v: MxArray,
  g: MxArray,
  beta: MxArray,
  state: MxArray,
  mask?: MxArray,
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
  let output = sum(nextStateTimesQuery, 3);
  let retainedState = retainArray(nextState);

  try {
    if (mask !== undefined) {
      using maskHeads = expandDims(mask, 1);
      using outputMask = expandDims(maskHeads, 2);
      using zeroOutput = zeros([...output.shape], output.dtype);
      const maskedOutput = where(outputMask, output, zeroOutput);
      output.free();
      output = maskedOutput;

      using stateMask = expandDims(outputMask, 3);
      const maskedState = where(stateMask, nextState, state);
      retainedState.free();
      retainedState = maskedState;
    }
  } catch (error) {
    output.free();
    retainedState.free();
    throw error;
  }

  return {
    output,
    state: retainedState,
  };
}

/** Compute the gated-delta recurrent sequence with ordinary MLX ops. */
export function gatedDeltaSequence(
  q: MxArray,
  k: MxArray,
  v: MxArray,
  g: MxArray,
  beta: MxArray,
  initialState: MxArray,
  mask?: MxArray,
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
      using maskStep = mask === undefined ? undefined : sliceSequenceStep(mask, step);
      const next = gatedDeltaStep(qStep, kStep, vStep, gStep, betaStep, state, maskStep);
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

function canUseNativeGatedDelta(q: MxArray, v: MxArray, initialState: MxArray): boolean {
  const keyHeads = q.shape[2];
  const keyHeadDim = q.shape[3];
  const valueHeads = v.shape[2];
  return (
    canUseMetal() &&
    keyHeads !== undefined &&
    keyHeadDim !== undefined &&
    valueHeads !== undefined &&
    keyHeads > 0 &&
    valueHeads > 0 &&
    valueHeads % keyHeads === 0 &&
    keyHeadDim % 32 === 0 &&
    initialState.dtype === "float32"
  );
}

/** Run native gated-delta from key-head q/k inputs, falling back to the TS oracle. */
export function gatedDeltaSequenceFromKeyHeads(
  q: MxArray,
  k: MxArray,
  v: MxArray,
  g: MxArray,
  beta: MxArray,
  initialState: MxArray,
  mask?: MxArray,
): { output: MxArray; state: MxArray } {
  if (canUseNativeGatedDelta(q, v, initialState)) {
    return mask === undefined
      ? fast.qwenGatedDeltaUpdate(q, k, v, g, beta, initialState)
      : fast.qwenGatedDeltaUpdate(q, k, v, g, beta, initialState, { mask });
  }

  const keyHeads = q.shape[2];
  const valueHeads = v.shape[2];
  if (keyHeads === undefined || valueHeads === undefined || valueHeads % keyHeads !== 0) {
    throw new Error(
      `Qwen3_5GatedDeltaNet: value heads ${valueHeads ?? "undefined"} must be divisible by key heads ${keyHeads ?? "undefined"}.`,
    );
  }
  const repeatFactor = valueHeads / keyHeads;
  using repeatedQueries = repeatHeads(q, repeatFactor);
  using repeatedKeys = repeatHeads(k, repeatFactor);
  using floatQueries = asType(repeatedQueries, "float32");
  using floatKeys = asType(repeatedKeys, "float32");
  using floatValues = asType(v, "float32");
  using floatG = asType(g, "float32");
  using floatBeta = asType(beta, "float32");
  return gatedDeltaSequence(
    floatQueries,
    floatKeys,
    floatValues,
    floatG,
    floatBeta,
    initialState,
    mask,
  );
}
