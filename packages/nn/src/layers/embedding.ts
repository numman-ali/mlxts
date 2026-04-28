/**
 * Embedding layer — maps integer indices to dense vectors.
 *
 * Stores a weight matrix of shape `[numEmbeddings, embDim]` and
 * looks up rows by integer index. Also provides `asLinear()` for
 * weight tying (shared embedding + output projection).
 *
 * @module
 */

import type { MxArray } from "@mlxts/core";
import { formatShape, isIntegerDType, matmul, random, takeAxis, transpose } from "@mlxts/core";
import { Module } from "../module";

/** Embedding layer: integer indices → dense vectors. */
export class Embedding extends Module {
  weight: MxArray;
  #embeddingDims: number;
  /** Cached because tied projections otherwise rebuild an identical transpose during decode. */
  #transposedWeight: MxArray | null = null;
  #transposedWeightSource: MxArray | null = null;

  /**
   * @param numEmbeddings - Size of the vocabulary. Must be > 0.
   * @param embDim - Dimension of each embedding vector. Must be > 0.
   */
  constructor(numEmbeddings: number, embDim: number) {
    super();
    if (numEmbeddings <= 0) {
      throw new Error(`Embedding: numEmbeddings must be > 0, got ${numEmbeddings}`);
    }
    if (embDim <= 0) {
      throw new Error(`Embedding: embDim must be > 0, got ${embDim}`);
    }
    this.#embeddingDims = embDim;
    this.weight = random.normal([numEmbeddings, embDim]);
  }

  /**
   * Look up embeddings by integer indices.
   *
   * @param indices - Integer tensor of any shape. Must be integer dtype.
   * @returns Tensor of shape `[...indices.shape, embDim]`.
   */
  forward(indices: MxArray): MxArray {
    if (!isIntegerDType(indices.dtype)) {
      throw new Error(
        `Embedding.forward: indices must be integer dtype (int32, uint32, etc.), got ${indices.dtype}.\n` +
          '  Hint: use array([1, 2, 3], "int32") to create integer indices.',
      );
    }
    return takeAxis(this.weight, indices, 0);
  }

  /**
   * Use the embedding weight as a linear projection (weight tying).
   *
   * Computes `x @ weight.T` — used in GPT-2 where the token embedding
   * and output projection share the same weight matrix. This is functional
   * reuse only in Phase 3; registering the same public parameter through
   * multiple fields remains deferred to Phase 4.
   */
  asLinear(x: MxArray): MxArray {
    const lastDimension = x.shape[x.shape.length - 1];
    if (lastDimension !== this.#embeddingDims) {
      throw new Error(
        `Embedding.asLinear: expected input last dimension ${this.#embeddingDims}, got ${lastDimension ?? "undefined"} ` +
          `for shape ${formatShape(x.shape)}.`,
      );
    }

    return matmul(x, this.transposedWeight());
  }

  private transposedWeight(): MxArray {
    if (this.#transposedWeight === null || this.#transposedWeightSource !== this.weight) {
      this.#transposedWeight?.free();
      this.#transposedWeight = transpose(this.weight);
      this.#transposedWeightSource = this.weight;
    }
    return this.#transposedWeight;
  }

  override [Symbol.dispose](): void {
    this.#transposedWeight?.free();
    this.#transposedWeight = null;
    this.#transposedWeightSource = null;
    super[Symbol.dispose]();
  }
}
