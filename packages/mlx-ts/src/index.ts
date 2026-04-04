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
export {
  deviceCount,
  getDefaultDevice,
  isDeviceAvailable,
  setDefaultDevice,
  synchronize,
} from "./core/device";
export type { DType } from "./core/dtype";
export { INTEGER_DTYPES, isIntegerDType } from "./core/dtype";
export { MxError } from "./core/error";
export type {
  FastLayerNormOptions,
  ScaledDotProductAttentionMaskMode,
  ScaledDotProductAttentionOptions,
} from "./core/fast";
// --- Fast fused operations ---
// biome-ignore lint/performance/noReExportAll: Intentional namespace re-export — matches MLX Python's mx.fast API pattern
export * as fast from "./core/fast";
export { layerNorm as fastLayerNorm, scaledDotProductAttention } from "./core/fast";
export type { LoadedSafetensors } from "./core/io";
export { loadSafetensors, saveSafetensors } from "./core/io";
export type { MemoryStats } from "./core/memory";
export {
  clearMemoryCache,
  getActiveMemoryBytes,
  getCacheMemoryBytes,
  getMemoryLimitBytes,
  getMemoryStats,
  getPeakMemoryBytes,
  resetPeakMemory,
  setCacheLimitBytes,
  setMemoryLimitBytes,
  setWiredLimitBytes,
} from "./core/memory";
export { isMetalAvailable, startMetalCapture, stopMetalCapture } from "./core/metal";
export type { Operand, SoftmaxOptions } from "./core/ops";
// --- Operations ---
export {
  abs,
  // Arithmetic
  add,
  argmax,
  argmin,
  asType,
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
  split,
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
  tril,
  triu,
  where,
} from "./core/ops";
// --- Random ---
// biome-ignore lint/performance/noReExportAll: Intentional namespace re-export — matches MLX Python's mx.random.normal() API pattern
export * as random from "./core/random";
// --- Transforms ---
export type { CompileMode } from "./core/transforms";
export {
  checkpoint,
  clearCompileCache,
  compile,
  disableCompile,
  enableCompile,
  grad,
  mxAsyncEval,
  mxEval,
  setCompileMode,
  valueAndGrad,
} from "./core/transforms";
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
export type { AdamWCheckpoint } from "./optimizers";
// --- Optimizers ---
// biome-ignore lint/performance/noReExportAll: Intentional namespace re-export — matches MLX Python's optimizers API pattern
export * as optimizers from "./optimizers";
export { Adam, AdamW, SGD } from "./optimizers";
export { formatShape } from "./utils/format-shape";
// --- Tree utilities (for gradient manipulation in training loops) ---
export type { FlatEntry, ParameterTree } from "./utils/tree";
export { treeFlatten, treeLeaves, treeMap, treeUnflatten } from "./utils/tree";

export const VERSION = "0.0.1";
