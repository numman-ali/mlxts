/**
 * Multi-head causal self-attention.
 *
 * Uses a single combined QKV projection (one matmul instead of three),
 * then splits into Q, K, V for each head. A causal mask prevents
 * attending to future positions.
 *
 * @module
 */

import type { MxArray } from "mlx-ts";
import {
  Dropout,
  Linear,
  Module,
  matmul,
  multiply,
  ones,
  reshape,
  scaledDotProductAttention,
  softmax,
  split,
  transpose,
  tril,
  where,
} from "mlx-ts";
import type { GPTConfig } from "../config";

/** Multi-head causal self-attention block. */
export class CausalSelfAttention extends Module {
  qkvProjection: Linear;
  outputProjection: Linear;
  attentionDropout: Dropout;
  residualDropout: Dropout;
  #nHead: number;
  #headDim: number;
  #manualCausalMasks = new Map<number, MxArray>();

  constructor(config: GPTConfig) {
    super();
    this.#nHead = config.nHead;
    this.#headDim = config.nEmbd / config.nHead;
    this.qkvProjection = new Linear(config.nEmbd, 3 * config.nEmbd);
    this.outputProjection = new Linear(config.nEmbd, config.nEmbd);
    this.attentionDropout = new Dropout(config.dropout);
    this.residualDropout = new Dropout(config.dropout);
  }

  /**
   * Forward pass: multi-head causal self-attention.
   *
   * @param x - Input tensor of shape [batch, sequence, nEmbd].
   * @returns Output tensor of same shape [batch, sequence, nEmbd].
   */
  forward(x: MxArray): MxArray {
    const [B, T] = x.shape;
    if (B === undefined || T === undefined) {
      throw new Error(`CausalSelfAttention: expected rank-3 input, got shape [${x.shape}]`);
    }
    const nHead = this.#nHead;
    const headDim = this.#headDim;
    const scale = Math.sqrt(1 / headDim);

    // 1. Combined QKV projection → [B, T, 3 * nEmbd]
    using qkv = this.qkvProjection.forward(x);

    // 2. Split into Q, K, V → each [B, T, nEmbd]
    const parts = split(qkv, 3, -1);
    const q = parts[0];
    const k = parts[1];
    const v = parts[2];
    if (q === undefined || k === undefined || v === undefined) {
      throw new Error("CausalSelfAttention: split did not produce 3 parts");
    }

    try {
      // 3. Reshape to [B, T, nHead, headDim] then transpose to [B, nHead, T, headDim]
      using qHeadInputs = reshape(q, [B, T, nHead, headDim]);
      using kHeadInputs = reshape(k, [B, T, nHead, headDim]);
      using vHeadInputs = reshape(v, [B, T, nHead, headDim]);
      using qHeads = transpose(qHeadInputs, [0, 2, 1, 3]);
      using kHeads = transpose(kHeadInputs, [0, 2, 1, 3]);
      using vHeads = transpose(vHeadInputs, [0, 2, 1, 3]);

      // 4. Use fused attention when it matches the configured semantics.
      using attendedOutput = this.computeAttentionOutput(qHeads, kHeads, vHeads, scale, T);

      // 5. Transpose back to [B, T, nHead, headDim] and reshape to [B, T, nEmbd]
      using transposed = transpose(attendedOutput, [0, 2, 1, 3]);
      using merged = reshape(transposed, [B, T, nHead * headDim]);

      // 6. Output projection + residual dropout
      using projected = this.outputProjection.forward(merged);
      return this.residualDropout.forward(projected);
    } finally {
      q.free();
      k.free();
      v.free();
    }
  }

  #usesAttentionWeightDropout(): boolean {
    return this.isTraining && this.attentionDropout.probability > 0;
  }

  private computeAttentionOutput(
    queries: MxArray,
    keys: MxArray,
    values: MxArray,
    scale: number,
    sequenceLength: number,
  ): MxArray {
    if (this.#usesAttentionWeightDropout()) {
      return this.manualAttention(queries, keys, values, scale, sequenceLength);
    }

    using attended = scaledDotProductAttention(queries, keys, values, {
      scale,
      maskMode: "causal",
    });
    return this.attentionDropout.forward(attended);
  }

  private manualAttention(
    queries: MxArray,
    keys: MxArray,
    values: MxArray,
    scale: number,
    sequenceLength: number,
  ): MxArray {
    using keyTranspose = transpose(keys, [0, 1, 3, 2]);
    using scores = matmul(queries, keyTranspose);
    using scaledScores = multiply(scores, scale);
    const causalMask = this.cachedManualCausalMask(sequenceLength);
    using maskedScores = where(causalMask, scaledScores, -1e9);
    using weights = softmax(maskedScores, -1);
    using droppedWeights = this.attentionDropout.forward(weights);
    return matmul(droppedWeights, values);
  }

  private cachedManualCausalMask(sequenceLength: number): MxArray {
    const existing = this.#manualCausalMasks.get(sequenceLength);
    if (existing !== undefined) {
      return existing;
    }

    using maskBase = ones([sequenceLength, sequenceLength], "bool");
    const mask = tril(maskBase);
    this.#manualCausalMasks.set(sequenceLength, mask);
    return mask;
  }

  override [Symbol.dispose](): void {
    for (const mask of this.#manualCausalMasks.values()) {
      mask.free();
    }
    this.#manualCausalMasks.clear();
    super[Symbol.dispose]();
  }
}
