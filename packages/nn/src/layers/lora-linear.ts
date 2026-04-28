/**
 * LoRA adapter wrapper for dense and quantized linear layers.
 * @module
 */

import type { MxArray } from "@mlxts/core";
import { add, matmul, multiply, random, retainArray, transpose, zeros } from "@mlxts/core";
import { Module } from "../module";
import { QuantizedLinear } from "../quantized/quantized-linear";
import { Dropout } from "./dropout";
import { Linear } from "./linear";

export type LoRALinearConfig = {
  rank?: number;
  alpha?: number;
  dropout?: number;
};

function inputAndOutputDims(base: Linear | QuantizedLinear): {
  inputDims: number;
  outputDims: number;
} {
  if (base instanceof QuantizedLinear) {
    return {
      inputDims: base.inputDims,
      outputDims: base.outputDims,
    };
  }

  const outputDims = base.weight.shape[0];
  const inputDims = base.weight.shape[1];
  if (outputDims === undefined || inputDims === undefined) {
    throw new Error("LoRALinear: base linear layer is missing rank-2 weight dimensions.");
  }

  return { inputDims, outputDims };
}

function cloneLinear(linear: Linear): Linear {
  const outputDims = linear.weight.shape[0];
  const inputDims = linear.weight.shape[1];
  if (outputDims === undefined || inputDims === undefined) {
    throw new Error("LoRALinear: base linear layer is missing rank-2 weight dimensions.");
  }

  const clone = new Linear(inputDims, outputDims, linear.bias !== null);
  clone.weight.free();
  clone.weight = retainArray(linear.weight);
  if (clone.bias !== null) {
    clone.bias.free();
    clone.bias = linear.bias === null ? null : retainArray(linear.bias);
  }
  return clone;
}

/** LoRA wrapper that keeps the frozen base layer and trains only low-rank adapters. */
export class LoRALinear extends Module {
  linear: Linear | QuantizedLinear | null;
  loraA: MxArray;
  loraB: MxArray;
  dropout: Dropout;
  #rank: number;
  #alpha: number;
  #scale: number;

  constructor(base: Linear | QuantizedLinear, options: LoRALinearConfig = {}) {
    super();
    const { inputDims, outputDims } = inputAndOutputDims(base);
    const rank = options.rank ?? 8;
    if (rank <= 0) {
      throw new Error(`LoRALinear: rank must be > 0, got ${rank}`);
    }

    const initScale = 1 / Math.sqrt(inputDims);
    this.linear = base;
    this.dropout = new Dropout(options.dropout ?? 0);
    this.#rank = rank;
    this.#alpha = options.alpha ?? 16;
    this.#scale = this.#alpha / rank;
    this.loraA = random.uniform(-initScale, initScale, [inputDims, rank]);
    this.loraB = zeros([rank, outputDims], "float32");
    this.freeze(["linear"]);
  }

  get rank(): number {
    return this.#rank;
  }

  get alpha(): number {
    return this.#alpha;
  }

  get dropoutProbability(): number {
    return this.dropout.probability;
  }

  static fromBase(base: Linear | QuantizedLinear, options: LoRALinearConfig = {}): LoRALinear {
    return new LoRALinear(base, options);
  }

  forward(x: MxArray): MxArray {
    if (this.linear === null) {
      throw new Error("LoRALinear.forward: base layer has been detached from this adapter.");
    }

    using baseOutput = this.linear.forward(x);
    using dropped = this.dropout.forward(x);
    using lowRank = matmul(dropped, this.loraA);
    using delta = matmul(lowRank, this.loraB);
    using scaledDelta = multiply(delta, this.#scale);
    return add(baseOutput, scaledDelta);
  }

  merge(options: { dequantize?: boolean } = {}): Linear | QuantizedLinear {
    if (this.linear === null) {
      throw new Error("LoRALinear.merge: base layer has been detached from this adapter.");
    }

    const base = this.linear;
    const mergedDense = base instanceof QuantizedLinear ? base.toLinear() : cloneLinear(base);
    try {
      using loraBT = transpose(this.loraB);
      using loraAT = transpose(this.loraA);
      using delta = matmul(loraBT, loraAT);
      using scaledDelta = multiply(delta, this.#scale);
      using mergedWeight = add(mergedDense.weight, scaledDelta);

      mergedDense.weight.free();
      mergedDense.weight = retainArray(mergedWeight);

      if (!(base instanceof QuantizedLinear) || options.dequantize === true) {
        return mergedDense;
      }

      const quantized = QuantizedLinear.fromLinear(mergedDense, {
        groupSize: base.groupSize,
        bits: base.bits,
        mode: base.mode,
      });
      mergedDense[Symbol.dispose]();
      return quantized;
    } catch (error) {
      mergedDense[Symbol.dispose]();
      throw error;
    }
  }

  takeBase(): Linear | QuantizedLinear {
    if (this.linear === null) {
      throw new Error("LoRALinear.takeBase: base layer has already been detached.");
    }

    const base = this.linear;
    this.linear = null;
    return base;
  }
}
