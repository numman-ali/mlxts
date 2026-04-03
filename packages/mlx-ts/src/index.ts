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
  tanh,
  transpose,
  where,
} from "./core/ops";
// --- Random ---
// biome-ignore lint/performance/noReExportAll: Intentional namespace re-export — matches MLX Python's mx.random.normal() API pattern
export * as random from "./core/random";
// --- Transforms ---
export { mxEval } from "./core/transforms";

export const VERSION = "0.0.1";
