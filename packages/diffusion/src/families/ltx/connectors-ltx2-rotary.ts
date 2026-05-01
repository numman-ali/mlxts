import {
  add,
  asType,
  MxArray,
  multiply,
  reshape,
  retainArray,
  squeeze,
  stack,
  subtract,
  transpose,
} from "@mlxts/core";

import type { Ltx2RopeType } from "./config";
import type { LtxRotaryEmbeddings } from "./embeddings";
import { assertSequence3d, selectLastAxis, sliceAxis } from "./tensor-utils";

/** Create LTX-2 connector RoPE tensors for a 1D text sequence. */
export function createLtx2ConnectorRotaryEmbeddings(options: {
  batch: number;
  length: number;
  dim: number;
  theta: number;
  baseSequenceLength: number;
  ropeType: Ltx2RopeType;
  heads: number;
}): LtxRotaryEmbeddings {
  const frequencyCount = Math.floor(options.dim / 2);
  if (frequencyCount <= 0) {
    throw new Error("LTX2ConnectorRotary: dim must be at least 2.");
  }
  if (options.ropeType === "split") {
    return splitConnectorPositions(options);
  }
  const padding = options.dim % 2;
  const cosValues = new Float32Array(options.batch * options.length * options.dim);
  const sinValues = new Float32Array(cosValues.length);
  for (let row = 0; row < options.batch; row += 1) {
    for (let token = 0; token < options.length; token += 1) {
      const position = token / options.baseSequenceLength;
      const outputOffset = (row * options.length + token) * options.dim;
      for (let index = 0; index < padding; index += 1) {
        cosValues[outputOffset + index] = 1;
      }
      for (let index = 0; index < frequencyCount; index += 1) {
        const unit = frequencyCount === 1 ? 0 : index / (frequencyCount - 1);
        const frequency = options.theta ** unit * (Math.PI / 2);
        const angle = (position * 2 - 1) * frequency;
        const offset = outputOffset + padding + index * 2;
        const cosValue = Math.cos(angle);
        const sinValue = Math.sin(angle);
        cosValues[offset] = cosValue;
        cosValues[offset + 1] = cosValue;
        sinValues[offset] = sinValue;
        sinValues[offset + 1] = sinValue;
      }
    }
  }
  return {
    cos: MxArray.fromData(cosValues, [options.batch, options.length, options.dim], "float32"),
    sin: MxArray.fromData(sinValues, [options.batch, options.length, options.dim], "float32"),
  };
}

function splitConnectorPositions(options: {
  batch: number;
  length: number;
  dim: number;
  theta: number;
  baseSequenceLength: number;
  heads: number;
}): LtxRotaryEmbeddings {
  const expectedFreqs = Math.floor(options.dim / 2);
  if (expectedFreqs % options.heads !== 0) {
    throw new Error("LTX2ConnectorRotary: split RoPE dim must divide evenly across heads.");
  }
  const headFreqs = expectedFreqs / options.heads;
  const cosValues = new Float32Array(options.batch * options.heads * options.length * headFreqs);
  const sinValues = new Float32Array(cosValues.length);
  for (let row = 0; row < options.batch; row += 1) {
    for (let token = 0; token < options.length; token += 1) {
      const position = token / options.baseSequenceLength;
      const flat = new Float32Array(expectedFreqs);
      const flatSin = new Float32Array(expectedFreqs);
      for (let index = 0; index < expectedFreqs; index += 1) {
        const unit = expectedFreqs === 1 ? 0 : index / (expectedFreqs - 1);
        const frequency = options.theta ** unit * (Math.PI / 2);
        const angle = (position * 2 - 1) * frequency;
        flat[index] = Math.cos(angle);
        flatSin[index] = Math.sin(angle);
      }
      for (let head = 0; head < options.heads; head += 1) {
        const sourceOffset = head * headFreqs;
        const outputOffset = ((row * options.heads + head) * options.length + token) * headFreqs;
        cosValues.set(flat.subarray(sourceOffset, sourceOffset + headFreqs), outputOffset);
        sinValues.set(flatSin.subarray(sourceOffset, sourceOffset + headFreqs), outputOffset);
      }
    }
  }
  return {
    cos: MxArray.fromData(
      cosValues,
      [options.batch, options.heads, options.length, headFreqs],
      "float32",
    ),
    sin: MxArray.fromData(
      sinValues,
      [options.batch, options.heads, options.length, headFreqs],
      "float32",
    ),
  };
}

/** Apply interleaved or split LTX-2 connector RoPE to a sequence tensor. */
export function applyLtx2ConnectorRotary(
  x: MxArray,
  embeddings: LtxRotaryEmbeddings,
  ropeType: Ltx2RopeType,
  heads: number,
): MxArray {
  return ropeType === "split"
    ? applySplitRotary(x, embeddings, heads)
    : applyInterleavedRotary(x, embeddings);
}

function applyInterleavedRotary(x: MxArray, embeddings: LtxRotaryEmbeddings): MxArray {
  const { batch, length, channels } = assertSequence3d(x, "applyLtx2ConnectorRotary");
  if (channels % 2 !== 0) {
    throw new Error("applyLtx2ConnectorRotary: hidden size must be even.");
  }
  if (
    embeddings.cos.shape.length !== 3 ||
    embeddings.sin.shape.length !== 3 ||
    embeddings.cos.shape[0] !== batch ||
    embeddings.cos.shape[1] !== length ||
    embeddings.cos.shape[2] !== channels ||
    embeddings.sin.shape[0] !== batch ||
    embeddings.sin.shape[1] !== length ||
    embeddings.sin.shape[2] !== channels
  ) {
    throw new Error("applyLtx2ConnectorRotary: RoPE shape mismatch.");
  }
  using xFloat = x.dtype === "float32" ? retainArray(x) : asType(x, "float32");
  using pairs = reshape(xFloat, [batch, length, channels / 2, 2]);
  using first = selectLastAxis(pairs, 0);
  using second = selectLastAxis(pairs, 1);
  using negSecond = multiply(second, -1);
  using rotatedPairs = stack([negSecond, first], -1);
  using rotated = reshape(rotatedPairs, [batch, length, channels]);
  using direct = multiply(xFloat, embeddings.cos);
  using shifted = multiply(rotated, embeddings.sin);
  using output = add(direct, shifted);
  return output.dtype === x.dtype ? retainArray(output) : asType(output, x.dtype);
}

function applySplitRotary(x: MxArray, embeddings: LtxRotaryEmbeddings, heads: number): MxArray {
  const { batch, length, channels } = assertSequence3d(x, "applyLtx2ConnectorSplitRotary");
  if (channels % heads !== 0 || (channels / heads) % 2 !== 0) {
    throw new Error("applyLtx2ConnectorSplitRotary: per-head hidden size must be even.");
  }
  const halfHeadDim = channels / heads / 2;
  if (
    embeddings.cos.shape.length !== 4 ||
    embeddings.sin.shape.length !== 4 ||
    embeddings.cos.shape[0] !== batch ||
    embeddings.cos.shape[1] !== heads ||
    embeddings.cos.shape[2] !== length ||
    embeddings.cos.shape[3] !== halfHeadDim ||
    embeddings.sin.shape[0] !== batch ||
    embeddings.sin.shape[1] !== heads ||
    embeddings.sin.shape[2] !== length ||
    embeddings.sin.shape[3] !== halfHeadDim
  ) {
    throw new Error("applyLtx2ConnectorSplitRotary: RoPE shape mismatch.");
  }
  using xFloat = x.dtype === "float32" ? retainArray(x) : asType(x, "float32");
  using headsLast = reshape(xFloat, [batch, length, heads, channels / heads]);
  using headsFirst = transpose(headsLast, [0, 2, 1, 3]);
  using pairs = reshape(headsFirst, [batch, heads, length, 2, halfHeadDim]);
  using firstSlice = sliceAxis(pairs, 3, 0, 1);
  using secondSlice = sliceAxis(pairs, 3, 1, 2);
  using first = squeeze(firstSlice, 3);
  using second = squeeze(secondSlice, 3);
  using firstDirect = multiply(first, embeddings.cos);
  using firstShift = multiply(second, embeddings.sin);
  using firstOut = subtract(firstDirect, firstShift);
  using secondDirect = multiply(second, embeddings.cos);
  using secondShift = multiply(first, embeddings.sin);
  using secondOut = add(secondDirect, secondShift);
  using rotatedPairs = stack([firstOut, secondOut], 3);
  using rotatedHeads = reshape(rotatedPairs, [batch, heads, length, channels / heads]);
  using sequenceFirst = transpose(rotatedHeads, [0, 2, 1, 3]);
  using output = reshape(sequenceFirst, [batch, length, channels]);
  return output.dtype === x.dtype ? retainArray(output) : asType(output, x.dtype);
}
