/**
 * Quantized embedding layer backed by MLX packed-weight primitives.
 * @module
 */

import type { QuantizationMode } from "@mlxts/core";
import {
  dequantize,
  formatShape,
  isIntegerDType,
  type MxArray,
  quantize,
  quantizedMatmul,
  takeAxis,
  zeros,
} from "@mlxts/core";
import { Embedding } from "../layers/embedding";
import { Module } from "../module";

export type QuantizedEmbeddingOptions = {
  groupSize?: number;
  bits?: number;
  mode?: QuantizationMode;
  weight?: MxArray;
  scales?: MxArray;
  quantizationBiases?: MxArray | null;
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
  options: Pick<QuantizedEmbeddingOptions, "groupSize" | "bits" | "mode">,
): ResolvedQuantizationOptions {
  const mode = options.mode ?? "affine";
  const defaults = defaultQuantizationOptions(mode);
  return {
    groupSize: options.groupSize ?? defaults.groupSize,
    bits: options.bits ?? defaults.bits,
    mode,
  };
}

function packedColumnCount(embeddingDims: number, bits: number): number {
  const packedColumns = (embeddingDims * bits) / 32;
  if (!Number.isInteger(packedColumns) || packedColumns <= 0) {
    throw new Error(
      `QuantizedEmbedding: embedding dimension ${embeddingDims} and bits ${bits} do not form a valid packed layout.`,
    );
  }
  return packedColumns;
}

function quantizationGroupCount(embeddingDims: number, groupSize: number): number {
  if (embeddingDims % groupSize !== 0) {
    throw new Error(
      `QuantizedEmbedding: embedding dimension ${embeddingDims} must be divisible by groupSize ${groupSize}.`,
    );
  }
  return embeddingDims / groupSize;
}

function quantizationUsesBiases(mode: QuantizationMode): boolean {
  return mode === "affine";
}

/** Packed-weight embedding layer that dequantizes only the selected rows. */
export class QuantizedEmbedding extends Module {
  weight: MxArray;
  scales: MxArray;
  biases: MxArray | null;
  #groupSize: number;
  #bits: number;
  #mode: QuantizationMode;
  #numEmbeddings: number;
  #embeddingDims: number;

  constructor(
    numEmbeddings: number,
    embeddingDims: number,
    options: QuantizedEmbeddingOptions = {},
  ) {
    super();
    if (numEmbeddings <= 0) {
      throw new Error(`QuantizedEmbedding: numEmbeddings must be > 0, got ${numEmbeddings}`);
    }
    if (embeddingDims <= 0) {
      throw new Error(`QuantizedEmbedding: embeddingDims must be > 0, got ${embeddingDims}`);
    }

    const resolved = resolveQuantizationOptions(options);
    const packedColumns = packedColumnCount(embeddingDims, resolved.bits);
    const groupCount = quantizationGroupCount(embeddingDims, resolved.groupSize);

    this.#numEmbeddings = numEmbeddings;
    this.#embeddingDims = embeddingDims;
    this.#groupSize = resolved.groupSize;
    this.#bits = resolved.bits;
    this.#mode = resolved.mode;
    this.weight = options.weight ?? zeros([numEmbeddings, packedColumns], "uint32");
    this.scales = options.scales ?? zeros([numEmbeddings, groupCount], "float32");
    this.biases =
      options.quantizationBiases ??
      (quantizationUsesBiases(resolved.mode)
        ? zeros([numEmbeddings, groupCount], "float32")
        : null);
  }

  get numEmbeddings(): number {
    return this.#numEmbeddings;
  }

  get embeddingDims(): number {
    return this.#embeddingDims;
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

  static fromEmbedding(
    embedding: Embedding,
    options: Pick<QuantizedEmbeddingOptions, "groupSize" | "bits" | "mode"> = {},
  ): QuantizedEmbedding {
    const numEmbeddings = embedding.weight.shape[0];
    const embeddingDims = embedding.weight.shape[1];
    if (numEmbeddings === undefined || embeddingDims === undefined) {
      throw new Error(
        `QuantizedEmbedding.fromEmbedding: expected rank-2 weight, got ${formatShape(embedding.weight.shape)}.`,
      );
    }

    const resolved = resolveQuantizationOptions(options);
    const quantized = quantize(embedding.weight, resolved);
    return new QuantizedEmbedding(numEmbeddings, embeddingDims, {
      ...resolved,
      weight: quantized.weight,
      scales: quantized.scales,
      quantizationBiases: quantized.biases ?? null,
    });
  }

  forward(indices: MxArray): MxArray {
    if (!isIntegerDType(indices.dtype)) {
      throw new Error(
        `QuantizedEmbedding.forward: indices must be integer dtype (int32, uint32, etc.), got ${indices.dtype}.\n` +
          '  Hint: use array([1, 2, 3], "int32") to create integer indices.',
      );
    }

    const selectedWeight = takeAxis(this.weight, indices, 0);
    const selectedScales = takeAxis(this.scales, indices, 0);
    const selectedBiases = this.biases === null ? null : takeAxis(this.biases, indices, 0);
    try {
      return dequantize(selectedWeight, selectedScales, {
        bits: this.#bits,
        groupSize: this.#groupSize,
        mode: this.#mode,
        ...(selectedBiases === null ? {} : { biases: selectedBiases }),
      });
    } finally {
      selectedWeight.free();
      selectedScales.free();
      selectedBiases?.free();
    }
  }

  /**
   * Use the embedding weight as a quantized linear projection.
   *
   * This preserves weight tying without materializing the full dense embedding
   * matrix, matching MLX's QuantizedEmbedding.as_linear behavior.
   */
  asLinear(x: MxArray): MxArray {
    const lastDimension = x.shape[x.shape.length - 1];
    if (lastDimension !== this.#embeddingDims) {
      throw new Error(
        `QuantizedEmbedding.asLinear: expected input last dimension ${this.#embeddingDims}, got ${lastDimension ?? "undefined"} ` +
          `for shape ${formatShape(x.shape)}.`,
      );
    }

    return quantizedMatmul(x, this.weight, this.scales, {
      bits: this.#bits,
      groupSize: this.#groupSize,
      mode: this.#mode,
      transpose: true,
      ...(this.biases === null ? {} : { biases: this.biases }),
    });
  }

  toEmbedding(): Embedding {
    const dense = new Embedding(this.#numEmbeddings, this.#embeddingDims);
    dense.weight.free();
    dense.weight = dequantize(this.weight, this.scales, {
      bits: this.#bits,
      groupSize: this.#groupSize,
      mode: this.#mode,
      ...(this.biases === null ? {} : { biases: this.biases }),
    });
    return dense;
  }
}
