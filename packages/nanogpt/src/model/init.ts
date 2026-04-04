/**
 * GPT-2 weight initialization.
 *
 * Follows the GPT-2 initialization scheme:
 * - Linear weights: normal(0, 0.02)
 * - Linear biases: zeros
 * - Embedding weights: normal(0, 0.02)
 * - Residual projections: scaled by 1/sqrt(2 * nLayer)
 * - LayerNorm: weight=ones, bias=zeros (already the default)
 *
 * @module
 */

import {
  type Embedding,
  type Linear,
  type MxArray,
  mxEval,
  random,
  treeFlatten,
  zeros,
} from "mlx-ts";
import type { GPTConfig } from "../config";
import type { GPT } from "./gpt";

const INIT_STD = 0.02;

/** Replace an MxArray parameter with a new one, freeing the old. */
function replaceParam(old: MxArray, fresh: MxArray): MxArray {
  old.free();
  return fresh;
}

/** Initialize a Linear layer's weights and biases with GPT-2 scheme. */
function initLinear(layer: Linear, std: number): void {
  layer.weight = replaceParam(
    layer.weight,
    random.normal([...layer.weight.shape], "float32", 0, std),
  );
  if (layer.bias !== null) {
    layer.bias = replaceParam(layer.bias, zeros([...layer.bias.shape]));
  }
}

/** Initialize an Embedding layer's weights. */
function initEmbedding(embedding: Embedding, std: number): void {
  embedding.weight = replaceParam(
    embedding.weight,
    random.normal([...embedding.weight.shape], "float32", 0, std),
  );
}

/**
 * Apply GPT-2 weight initialization to all model parameters.
 *
 * Residual projections (attention output and MLP contract) are
 * additionally scaled by 1/sqrt(2 * nLayer) to prevent the residual
 * stream magnitude from growing with depth.
 */
export function initializeGPT(model: GPT, config: GPTConfig): void {
  const residualStd = INIT_STD / Math.sqrt(2 * config.nLayer);

  // Embeddings
  initEmbedding(model.tokenEmbedding, INIT_STD);
  initEmbedding(model.positionEmbedding, INIT_STD);

  // Transformer blocks
  for (const block of model.blocks) {
    // Attention QKV and output
    initLinear(block.attention.qkvProjection, INIT_STD);
    initLinear(block.attention.outputProjection, residualStd);

    // MLP expand and contract
    initLinear(block.mlp.expandProjection, INIT_STD);
    initLinear(block.mlp.contractProjection, residualStd);
  }

  const allParams = treeFlatten(model.parameters()).map(([, value]) => value);
  mxEval(...allParams);
}
