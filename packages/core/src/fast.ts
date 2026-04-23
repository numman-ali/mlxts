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
import { MxArray, readResultArray, readResultArrayWithMetadata } from "./array";
import { defaultStream } from "./device";
import { checkStatus } from "./error";
import { OutSlot, optionalFloat, ptr } from "./ffi";
import { ffi } from "./ffi/lib";

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

export type FastRMSNormOptions = {
  eps?: number;
  stream?: S;
};

export type FastRoPEOptions = {
  traditional?: boolean;
  base?: number;
  scale?: number;
  offset?: number | MxArray;
  freqs?: MxArray;
  stream?: S;
};

export type QwenGatedDeltaUpdateResult = {
  /** Recurrent outputs with shape `[batch, sequence, valueHeads, valueHeadDim]`. */
  output: MxArray;
  /** Updated recurrent state with shape `[batch, valueHeads, valueHeadDim, keyHeadDim]`. */
  state: MxArray;
};

const textEncoder = new TextEncoder();

function encodeCString(value: string): Uint8Array {
  return textEncoder.encode(`${value}\0`);
}

function encodeMaskMode(maskMode: ScaledDotProductAttentionMaskMode): Uint8Array {
  return encodeCString(maskMode);
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

function resolveRoPEBase(base: number | undefined, freqs: MxArray | undefined): number | undefined {
  if (freqs !== undefined && base !== undefined) {
    throw new Error("fast.rope: provide either base or freqs, not both.");
  }

  if (freqs !== undefined) {
    return undefined;
  }

  return base ?? 10000;
}

function validateRoPEInput(x: MxArray, dims: number): void {
  if (x.ndim < 3) {
    throw new Error(`fast.rope: expected input rank >= 3, got rank ${x.ndim}.`);
  }
  if (!Number.isInteger(dims) || dims <= 0) {
    throw new Error(`fast.rope: dims must be a positive integer, got ${dims}.`);
  }
  if (dims % 2 !== 0) {
    throw new Error(`fast.rope: dims must be even, got ${dims}.`);
  }

  const featureDimension = x.shape[x.shape.length - 1];
  if (featureDimension === undefined || dims > featureDimension) {
    throw new Error(
      `fast.rope: dims ${dims} must be <= the last dimension ${featureDimension ?? "undefined"}.`,
    );
  }
}

function expectRank(name: string, value: MxArray, rank: number): void {
  if (value.shape.length !== rank) {
    throw new Error(
      `fast.qwenGatedDeltaUpdate: expected ${name} rank ${rank}, got ${value.shape.length}.`,
    );
  }
}

function expectDimension(
  name: string,
  value: MxArray,
  axis: number,
  expected: number,
  label: string,
): void {
  const actual = value.shape[axis];
  if (actual !== expected) {
    throw new Error(
      `fast.qwenGatedDeltaUpdate: expected ${name} ${label} to be ${expected}, got ${actual ?? "undefined"}.`,
    );
  }
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
  const [batch, heads, sequenceLength] = queries.shape;
  const headDim = values.shape[3];
  const shape = [batch ?? 0, heads ?? 0, sequenceLength ?? 0, headDim ?? 0];
  return readResultArrayWithMetadata(
    "fast_scaled_dot_product_attention",
    { shape, dtype: queries.dtype },
    (out) => {
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
    },
  );
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

/**
 * Fused RMS normalization.
 *
 * Normalizes across the last axis using the root mean square and
 * optionally applies a learnable weight.
 */
export function rmsNorm(x: MxArray, weight?: MxArray, options?: FastRMSNormOptions): MxArray {
  return readResultArrayWithMetadata(
    "fast_rms_norm",
    { shape: x.shape, dtype: x.dtype, ndim: x.ndim, size: x.size },
    (out) => {
      checkStatus(
        ffi.mlx_fast_rms_norm(
          out,
          x._ctx,
          weight?._ctx ?? null,
          options?.eps ?? 1e-5,
          s(options?.stream),
        ),
        "fast_rms_norm",
      );
    },
  );
}

/**
 * Apply rotary positional encoding to the last axis of an attention tensor.
 *
 * The input is expected to be at least rank 3 with shape `[batch, *, sequence, dims]`.
 * Offsets can be provided either as a single number or as a per-example array.
 */
export function rope(x: MxArray, dims: number, options: FastRoPEOptions = {}): MxArray {
  validateRoPEInput(x, dims);

  const resolvedBase = resolveRoPEBase(options.base, options.freqs);
  const offset = options.offset ?? 0;
  const traditional = options.traditional ?? false;
  const scale = options.scale ?? 1.0;

  if (typeof offset === "number") {
    if (!Number.isInteger(offset) || offset < 0) {
      throw new Error(`fast.rope: numeric offset must be a non-negative integer, got ${offset}.`);
    }

    return readResultArrayWithMetadata(
      "fast_rope",
      { shape: x.shape, dtype: x.dtype, ndim: x.ndim, size: x.size },
      (out) => {
        checkStatus(
          ffi.mlx_fast_rope(
            out,
            x._ctx,
            dims,
            traditional,
            optionalFloat(resolvedBase),
            scale,
            offset,
            options.freqs?._ctx ?? null,
            s(options.stream),
          ),
          "fast_rope",
        );
      },
    );
  }

  return readResultArrayWithMetadata(
    "fast_rope_dynamic",
    { shape: x.shape, dtype: x.dtype, ndim: x.ndim, size: x.size },
    (out) => {
      checkStatus(
        ffi.mlx_fast_rope_dynamic(
          out,
          x._ctx,
          dims,
          traditional,
          optionalFloat(resolvedBase),
          scale,
          offset._ctx,
          options.freqs?._ctx ?? null,
          s(options.stream),
        ),
        "fast_rope_dynamic",
      );
    },
  );
}

/**
 * Fused Qwen gated-delta recurrent update.
 *
 * Mirrors the MLX-LM Qwen 3.5/3.6 Metal kernel for scalar gating. The key and
 * query tensors stay in `[batch, sequence, keyHeads, keyHeadDim]` layout; value
 * heads are mapped to key heads inside the kernel.
 */
export function qwenGatedDeltaUpdate(
  q: MxArray,
  k: MxArray,
  v: MxArray,
  g: MxArray,
  beta: MxArray,
  state: MxArray,
  stream?: S,
): QwenGatedDeltaUpdateResult {
  expectRank("q", q, 4);
  expectRank("k", k, 4);
  expectRank("v", v, 4);
  expectRank("g", g, 3);
  expectRank("beta", beta, 3);
  expectRank("state", state, 4);

  const [batchSize, sequenceLength, keyHeads, keyHeadDim] = q.shape;
  const valueHeads = v.shape[2];
  const valueHeadDim = v.shape[3];
  if (
    batchSize === undefined ||
    sequenceLength === undefined ||
    keyHeads === undefined ||
    keyHeadDim === undefined ||
    valueHeads === undefined ||
    valueHeadDim === undefined
  ) {
    throw new Error("fast.qwenGatedDeltaUpdate: q and v must have fully known shapes.");
  }
  if (keyHeads <= 0 || valueHeads <= 0 || valueHeads % keyHeads !== 0) {
    throw new Error(
      `fast.qwenGatedDeltaUpdate: valueHeads ${valueHeads} must be divisible by keyHeads ${keyHeads}.`,
    );
  }
  if (keyHeadDim <= 0 || keyHeadDim % 32 !== 0) {
    throw new Error(
      `fast.qwenGatedDeltaUpdate: keyHeadDim ${keyHeadDim} must be a positive multiple of 32.`,
    );
  }
  expectDimension("k", k, 0, batchSize, "batch");
  expectDimension("k", k, 1, sequenceLength, "sequence");
  expectDimension("k", k, 2, keyHeads, "keyHeads");
  expectDimension("k", k, 3, keyHeadDim, "keyHeadDim");
  expectDimension("v", v, 0, batchSize, "batch");
  expectDimension("v", v, 1, sequenceLength, "sequence");
  expectDimension("g", g, 0, batchSize, "batch");
  expectDimension("g", g, 1, sequenceLength, "sequence");
  expectDimension("g", g, 2, valueHeads, "valueHeads");
  expectDimension("beta", beta, 0, batchSize, "batch");
  expectDimension("beta", beta, 1, sequenceLength, "sequence");
  expectDimension("beta", beta, 2, valueHeads, "valueHeads");
  expectDimension("state", state, 0, batchSize, "batch");
  expectDimension("state", state, 1, valueHeads, "valueHeads");
  expectDimension("state", state, 2, valueHeadDim, "valueHeadDim");
  expectDimension("state", state, 3, keyHeadDim, "keyHeadDim");

  const outputSlot = new OutSlot();
  const stateSlot = new OutSlot();
  checkStatus(
    ffi.mlxts_qwen_gated_delta_update(
      outputSlot.prepare(),
      stateSlot.prepare(),
      q._ctx,
      k._ctx,
      v._ctx,
      g._ctx,
      beta._ctx,
      state._ctx,
      s(stream),
    ),
    "mlxts_qwen_gated_delta_update",
  );

  const output = MxArray._fromCtx(outputSlot.read("qwen gated delta output"), {
    shape: [batchSize, sequenceLength, valueHeads, valueHeadDim],
    dtype: q.dtype,
    ndim: 4,
    size: batchSize * sequenceLength * valueHeads * valueHeadDim,
  });
  try {
    const nextState = MxArray._fromCtx(stateSlot.read("qwen gated delta state"), {
      shape: state.shape,
      dtype: state.dtype,
      ndim: state.ndim,
      size: state.size,
    });
    return { output, state: nextState };
  } catch (error) {
    output.free();
    throw error;
  }
}
