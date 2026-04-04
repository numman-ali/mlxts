/**
 * Fully connected linear layer.
 *
 * Computes `x @ weight.T + bias` (when bias is enabled).
 *
 * @module
 */

import type { MxArray } from "@mlxts/core";
import { add, formatShape, matmul, random, transpose, zeros } from "@mlxts/core";
import { Module } from "./module";

/** Fully connected layer: `y = x @ weight.T + bias`. */
export class Linear extends Module {
  weight: MxArray;
  bias: MxArray | null;
  #inputDims: number;

  /**
   * @param inputDims - Number of input features. Must be > 0.
   * @param outputDims - Number of output features. Must be > 0.
   * @param hasBias - Whether to include a bias term. Defaults to true.
   */
  constructor(inputDims: number, outputDims: number, hasBias = true) {
    super();
    if (inputDims <= 0) {
      throw new Error(`Linear: inputDims must be > 0, got ${inputDims}`);
    }
    if (outputDims <= 0) {
      throw new Error(`Linear: outputDims must be > 0, got ${outputDims}`);
    }

    const scale = 1 / Math.sqrt(inputDims);
    this.#inputDims = inputDims;
    this.weight = random.uniform(-scale, scale, [outputDims, inputDims]);
    this.bias = hasBias ? zeros([outputDims]) : null;
  }

  forward(x: MxArray): MxArray {
    const lastDimension = x.shape[x.shape.length - 1];
    if (lastDimension !== this.#inputDims) {
      throw new Error(
        `Linear.forward: expected input last dimension ${this.#inputDims}, got ${lastDimension ?? "undefined"} ` +
          `for shape ${formatShape(x.shape)}.`,
      );
    }

    using wt = transpose(this.weight);
    const out = matmul(x, wt);
    if (this.bias !== null) {
      using unbiased = out;
      return add(unbiased, this.bias);
    }
    return out;
  }
}
