/**
 * Quantized linear layer backed by MLX packed-weight kernels.
 * @module
 */

import type { QuantizationMode } from "@mlxts/core";
import {
  add,
  concatenate,
  dequantize,
  formatShape,
  type MxArray,
  quantize,
  quantizedMatmul,
  retainArray,
  zeros,
} from "@mlxts/core";
import { Linear } from "../layers/linear";
import { Module } from "../module";

export type QuantizedLinearOptions = {
  bias?: boolean;
  groupSize?: number;
  bits?: number;
  mode?: QuantizationMode;
  weight?: MxArray;
  scales?: MxArray;
  quantizationBiases?: MxArray | null;
  outputBias?: MxArray | null;
};

type ResolvedQuantizationOptions = {
  groupSize: number;
  bits: number;
  mode: QuantizationMode;
};

function defaultQuantizationOptions(mode: QuantizationMode): ResolvedQuantizationOptions {
  switch (mode) {
    case "mxfp4":
      return { groupSize: 32, bits: 4, mode };
    case "mxfp8":
      return { groupSize: 32, bits: 8, mode };
    case "nvfp4":
      return { groupSize: 16, bits: 4, mode };
    default:
      return { groupSize: 64, bits: 4, mode };
  }
}

function resolveQuantizationOptions(
  options: Pick<QuantizedLinearOptions, "groupSize" | "bits" | "mode">,
): ResolvedQuantizationOptions {
  const mode = options.mode ?? "affine";
  const defaults = defaultQuantizationOptions(mode);
  return {
    groupSize: options.groupSize ?? defaults.groupSize,
    bits: options.bits ?? defaults.bits,
    mode,
  };
}

function packedColumnCount(inputDims: number, bits: number): number {
  const packedColumns = (inputDims * bits) / 32;
  if (!Number.isInteger(packedColumns) || packedColumns <= 0) {
    throw new Error(
      `QuantizedLinear: inputDims ${inputDims} and bits ${bits} do not form a valid packed layout.`,
    );
  }
  return packedColumns;
}

function quantizationGroupCount(inputDims: number, groupSize: number): number {
  if (inputDims % groupSize !== 0) {
    throw new Error(
      `QuantizedLinear: inputDims ${inputDims} must be divisible by groupSize ${groupSize}.`,
    );
  }
  return inputDims / groupSize;
}

function quantizationUsesBiases(mode: QuantizationMode): boolean {
  return mode === "affine";
}

function compatibleForFusion(left: QuantizedLinear, right: QuantizedLinear): boolean {
  return (
    left.inputDims === right.inputDims &&
    left.groupSize === right.groupSize &&
    left.bits === right.bits &&
    left.mode === right.mode &&
    left.bias === null &&
    right.bias === null &&
    (left.biases === null) === (right.biases === null)
  );
}

/** Packed-weight linear layer with a separate output bias. */
export class QuantizedLinear extends Module {
  weight: MxArray;
  scales: MxArray;
  biases: MxArray | null;
  bias: MxArray | null;
  #groupSize: number;
  #bits: number;
  #mode: QuantizationMode;
  #outputDims: number;
  #inputDims: number;

  constructor(inputDims: number, outputDims: number, options: QuantizedLinearOptions = {}) {
    super();
    if (inputDims <= 0) {
      throw new Error(`QuantizedLinear: inputDims must be > 0, got ${inputDims}`);
    }
    if (outputDims <= 0) {
      throw new Error(`QuantizedLinear: outputDims must be > 0, got ${outputDims}`);
    }

    const resolved = resolveQuantizationOptions(options);
    const packedColumns = packedColumnCount(inputDims, resolved.bits);
    const groupCount = quantizationGroupCount(inputDims, resolved.groupSize);

    this.#inputDims = inputDims;
    this.#outputDims = outputDims;
    this.#groupSize = resolved.groupSize;
    this.#bits = resolved.bits;
    this.#mode = resolved.mode;
    this.weight = options.weight ?? zeros([outputDims, packedColumns], "uint32");
    this.scales = options.scales ?? zeros([outputDims, groupCount], "float32");
    this.biases =
      options.quantizationBiases ??
      (quantizationUsesBiases(resolved.mode) ? zeros([outputDims, groupCount], "float32") : null);
    this.bias =
      options.outputBias ?? (options.bias === true ? zeros([outputDims], "float32") : null);
  }

  get inputDims(): number {
    return this.#inputDims;
  }

  get outputDims(): number {
    return this.#outputDims;
  }

  get groupSize(): number {
    return this.#groupSize;
  }

  get bits(): number {
    return this.#bits;
  }

  get mode(): QuantizationMode {
    return this.#mode;
  }

  static fromLinear(
    linear: Linear,
    options: Pick<QuantizedLinearOptions, "groupSize" | "bits" | "mode"> = {},
  ): QuantizedLinear {
    const outputDims = linear.weight.shape[0];
    const inputDims = linear.weight.shape[1];
    if (outputDims === undefined || inputDims === undefined) {
      throw new Error(
        `QuantizedLinear.fromLinear: expected rank-2 weight, got ${formatShape(linear.weight.shape)}.`,
      );
    }

    const resolved = resolveQuantizationOptions(options);
    const quantized = quantize(linear.weight, resolved);
    return new QuantizedLinear(inputDims, outputDims, {
      ...resolved,
      bias: linear.bias !== null,
      weight: quantized.weight,
      scales: quantized.scales,
      quantizationBiases: quantized.biases ?? null,
      outputBias: linear.bias === null ? null : retainArray(linear.bias),
    });
  }

  forward(x: MxArray): MxArray {
    const lastDimension = x.shape[x.shape.length - 1];
    if (lastDimension !== this.#inputDims) {
      throw new Error(
        `QuantizedLinear.forward: expected input last dimension ${this.#inputDims}, got ${lastDimension ?? "undefined"} ` +
          `for shape ${formatShape(x.shape)}.`,
      );
    }

    const out = quantizedMatmul(x, this.weight, this.scales, {
      bits: this.#bits,
      groupSize: this.#groupSize,
      mode: this.#mode,
      transpose: true,
      ...(this.biases === null ? {} : { biases: this.biases }),
    });
    if (this.bias !== null) {
      using unbiased = out;
      return add(unbiased, this.bias);
    }
    return out;
  }

  toLinear(): Linear {
    const dense = new Linear(this.#inputDims, this.#outputDims, this.bias !== null);
    dense.weight.free();
    dense.weight = dequantize(this.weight, this.scales, {
      bits: this.#bits,
      groupSize: this.#groupSize,
      mode: this.#mode,
      dtype: "float32",
      ...(this.biases === null ? {} : { biases: this.biases }),
    });

    if (dense.bias !== null) {
      dense.bias.free();
      dense.bias = this.bias === null ? null : retainArray(this.bias);
    }

    return dense;
  }
}

/** Create one packed projection whose output concatenates compatible quantized linears. */
export function fuseQuantizedLinears(linears: readonly QuantizedLinear[]): QuantizedLinear | null {
  const first = linears[0];
  if (first === undefined || linears.length < 2) {
    return null;
  }
  for (let index = 1; index < linears.length; index += 1) {
    const linear = linears[index];
    if (linear === undefined || !compatibleForFusion(first, linear)) {
      return null;
    }
  }

  let weight: MxArray | null = null;
  let scales: MxArray | null = null;
  let biases: MxArray | null = null;
  try {
    weight = concatenate(
      linears.map((linear) => linear.weight),
      0,
    );
    scales = concatenate(
      linears.map((linear) => linear.scales),
      0,
    );
    if (first.biases !== null) {
      biases = concatenate(
        linears.map((linear) => {
          if (linear.biases === null) {
            throw new Error("fuseQuantizedLinears: expected compatible quantization biases.");
          }
          return linear.biases;
        }),
        0,
      );
    }

    const fused = new QuantizedLinear(
      first.inputDims,
      linears.reduce((sum, linear) => sum + linear.outputDims, 0),
      {
        bits: first.bits,
        groupSize: first.groupSize,
        mode: first.mode,
        weight,
        scales,
        quantizationBiases: biases,
        outputBias: null,
      },
    );
    weight = null;
    scales = null;
    biases = null;
    return fused;
  } finally {
    weight?.free();
    scales?.free();
    biases?.free();
  }
}
