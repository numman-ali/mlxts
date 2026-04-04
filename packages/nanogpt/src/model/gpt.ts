/**
 * GPT model — the complete transformer language model.
 *
 * Architecture:
 *   Token Embedding + Position Embedding → Dropout
 *   → N × TransformerBlock (pre-norm with residual connections)
 *   → LayerNorm → Output Projection (weight-tied with token embedding)
 *
 * Weight tying: the output projection reuses the token embedding weight
 * via Embedding.asLinear(). The weight appears once in the parameter tree
 * but participates in two forward-pass paths. MLX autograd accumulates
 * gradients from both uses.
 *
 * @module
 */

import type { MxArray } from "mlx-ts";
import { add, arange, Dropout, Embedding, isIntegerDType, LayerNorm, Module } from "mlx-ts";
import type { GPTConfig } from "../config";
import { TransformerBlock } from "./transformer-block";

/** GPT language model. */
export class GPT extends Module {
  tokenEmbedding: Embedding;
  positionEmbedding: Embedding;
  dropout: Dropout;
  blocks: TransformerBlock[];
  layerNorm: LayerNorm;
  #blockSize: number;

  constructor(config: GPTConfig) {
    super();
    this.#blockSize = config.blockSize;
    this.tokenEmbedding = new Embedding(config.vocabSize, config.nEmbd);
    this.positionEmbedding = new Embedding(config.blockSize, config.nEmbd);
    this.dropout = new Dropout(config.dropout);
    this.blocks = Array.from({ length: config.nLayer }, () => new TransformerBlock(config));
    this.layerNorm = new LayerNorm(config.nEmbd);
  }

  /**
   * Forward pass: tokens → logits.
   *
   * @param indices - Token IDs of shape [batch, sequence]. Must be integer dtype.
   *   Sequence length must not exceed blockSize.
   * @returns Logits of shape [batch, sequence, vocabSize].
   */
  forward(indices: MxArray): MxArray {
    if (!isIntegerDType(indices.dtype)) {
      throw new Error(
        `GPT.forward: expected integer token dtype, got ${indices.dtype}.\n` +
          '  Hint: use array(tokens, "int32") to create integer token IDs.',
      );
    }
    if (indices.ndim !== 2) {
      throw new Error(
        `GPT.forward: expected rank-2 input [batch, sequence], got rank ${indices.ndim} with shape [${indices.shape}]`,
      );
    }

    const T = indices.shape[1];
    if (T === undefined || T > this.#blockSize) {
      throw new Error(`GPT.forward: sequence length ${T} exceeds blockSize ${this.#blockSize}`);
    }

    // Token + position embeddings
    using tokEmb = this.tokenEmbedding.forward(indices);
    using posIndices = arange(0, T, 1, "int32");
    using posEmb = this.positionEmbedding.forward(posIndices);
    using combined = add(tokEmb, posEmb);

    // Dropout always returns a new array (train: masked copy, eval: retained copy)
    let x = this.dropout.forward(combined);

    // Transformer blocks — each produces a new array, free the previous
    for (const block of this.blocks) {
      const next = block.forward(x);
      x.free();
      x = next;
    }

    // Final layer norm + output projection via weight tying
    try {
      using normed = this.layerNorm.forward(x);
      return this.tokenEmbedding.asLinear(normed);
    } finally {
      x.free();
    }
  }
}
