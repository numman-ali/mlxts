/**
 * Fast fused operations backed by MLX's optimized kernels.
 *
 * These APIs expose higher-level fused primitives when MLX provides
 * them natively. They are especially valuable in transformer hot paths
 * where composed ops create extra graph nodes and intermediate tensors.
 *
 * @module
 */

import type { Pointer } from "bun:ffi";
import { type MxArray, readResultArray } from "./array";
import { defaultStream } from "./device";
import { checkStatus } from "./error";
import { ffi } from "./ffi/lib";
import { ptr } from "./ffi/pointer";

type S = Pointer | undefined;
const s = (stream?: S) => stream ?? defaultStream();

export type ScaledDotProductAttentionMaskMode = "" | "array" | "causal";

export type ScaledDotProductAttentionOptions = {
  scale: number;
  maskMode?: ScaledDotProductAttentionMaskMode;
  maskArray?: MxArray;
  sinks?: MxArray;
  stream?: S;
};

export type FastLayerNormOptions = {
  eps?: number;
  stream?: S;
};

const textEncoder = new TextEncoder();

function encodeMaskMode(maskMode: ScaledDotProductAttentionMaskMode): Uint8Array {
  return textEncoder.encode(`${maskMode}\0`);
}

function normalizeAttentionMaskMode(
  maskMode: ScaledDotProductAttentionMaskMode | undefined,
  maskArray: MxArray | undefined,
): ScaledDotProductAttentionMaskMode {
  const resolvedMode = maskMode ?? (maskArray === undefined ? "" : "array");
  if (resolvedMode === "array" && maskArray === undefined) {
    throw new Error(
      "fast.scaledDotProductAttention: maskMode 'array' requires maskArray to be provided.",
    );
  }
  if (resolvedMode === "causal" && maskArray !== undefined) {
    throw new Error(
      "fast.scaledDotProductAttention: maskArray cannot be provided when maskMode is 'causal'.",
    );
  }
  return resolvedMode;
}

/**
 * Fused scaled dot-product attention.
 *
 * Expects query, key, and value tensors in `[batch, heads, sequence, headDim]`
 * layout. Use `maskMode: "causal"` for standard autoregressive decoding.
 */
export function scaledDotProductAttention(
  queries: MxArray,
  keys: MxArray,
  values: MxArray,
  options: ScaledDotProductAttentionOptions,
): MxArray {
  const maskMode = normalizeAttentionMaskMode(options.maskMode, options.maskArray);
  const encodedMaskMode = encodeMaskMode(maskMode);
  return readResultArray("fast_scaled_dot_product_attention", (out) => {
    checkStatus(
      ffi.mlx_fast_scaled_dot_product_attention(
        out,
        queries._ctx,
        keys._ctx,
        values._ctx,
        options.scale,
        ptr(encodedMaskMode),
        options.maskArray?._ctx ?? null,
        options.sinks?._ctx ?? null,
        s(options.stream),
      ),
      "fast_scaled_dot_product_attention",
    );
  });
}

/**
 * Fused layer normalization.
 *
 * Normalizes across the last axis and optionally applies affine weight/bias.
 */
export function layerNorm(
  x: MxArray,
  weight?: MxArray,
  bias?: MxArray,
  options?: FastLayerNormOptions,
): MxArray {
  return readResultArray("fast_layer_norm", (out) => {
    checkStatus(
      ffi.mlx_fast_layer_norm(
        out,
        x._ctx,
        weight?._ctx ?? null,
        bias?._ctx ?? null,
        options?.eps ?? 1e-5,
        s(options?.stream),
      ),
      "fast_layer_norm",
    );
  });
}
