/**
 * mlx-ts — TypeScript bindings for Apple's MLX framework.
 *
 * Provides GPU-accelerated tensor operations on Apple Silicon
 * via Bun FFI to the MLX C++ core.
 *
 * @module mlx-ts
 */

export type { NestedArray } from "./core/array";
// --- Core types ---
// --- Array factories ---
export { arange, array, full, MxArray, ones, zeros } from "./core/array";
export type { DeviceType } from "./core/device";
// --- Device ---
export { getDefaultDevice, setDefaultDevice } from "./core/device";
export type { DType } from "./core/dtype";
export { MxError } from "./core/error";
export type { Operand } from "./core/ops";
// --- Operations ---
export {
  abs,
  // Arithmetic
  add,
  argmax,
  argmin,
  astype,
  broadcastTo,
  concatenate,
  divide,
  // Comparison
  equal,
  erf,
  exp,
  expandDims,
  flatten,
  greater,
  greaterEqual,
  less,
  lessEqual,
  log,
  logsumexp,
  // Linear algebra
  matmul,
  max,
  maximum,
  mean,
  min,
  minimum,
  multiply,
  negative,
  notEqual,
  power,
  reciprocal,
  // Shape
  reshape,
  sigmoid,
  softmax,
  sqrt,
  square,
  squeeze,
  stack,
  stopGradient,
  subtract,
  // Reductions
  sum,
  takeAlongAxis,
  takeAxis,
  tanh,
  transpose,
  where,
} from "./core/ops";
// --- Random ---
// biome-ignore lint/performance/noReExportAll: Intentional namespace re-export — matches MLX Python's mx.random.normal() API pattern
export * as random from "./core/random";
// --- Transforms ---
export { grad, mxEval, valueAndGrad } from "./core/transforms";
// --- Neural network ---
// biome-ignore lint/performance/noReExportAll: Intentional namespace re-export — matches MLX Python's nn.Module API pattern
export * as nn from "./nn";
// Flat convenience exports
export {
  crossEntropy,
  Dropout,
  Embedding,
  gelu,
  LayerNorm,
  Linear,
  Module,
  mse,
  relu,
  silu,
} from "./nn";
// --- Optimizers ---
// biome-ignore lint/performance/noReExportAll: Intentional namespace re-export — matches MLX Python's optimizers API pattern
export * as optimizers from "./optimizers";
export { Adam, AdamW, SGD } from "./optimizers";

export const VERSION = "0.0.1";
