/**
 * @mlxts/core — TypeScript bindings for Apple's MLX framework.
 *
 * Provides GPU-accelerated tensor operations on Apple Silicon
 * via Bun FFI to the MLX C++ core.
 *
 * @module
 */

export type { NestedArray } from "./array";
export { arange, array, full, MxArray, ones, retainArray, zeros } from "./array";
export type { DeviceType } from "./device";
export {
  createStream,
  deviceCount,
  getDefaultDevice,
  getRecommendedWorkingSetBytes,
  isDeviceAvailable,
  MxStream,
  setDefaultDevice,
  synchronize,
  withDefaultStream,
} from "./device";
export type { DType } from "./dtype";
export { INTEGER_DTYPES, isIntegerDType } from "./dtype";
export { MxError } from "./error";
export type {
  FastLayerNormOptions,
  FastRMSNormOptions,
  FastRoPEOptions,
  ScaledDotProductAttentionMaskMode,
  ScaledDotProductAttentionOptions,
} from "./fast";
// biome-ignore lint/performance/noReExportAll: Intentional namespace re-export — matches MLX Python's mx.fast API pattern
export * as fast from "./fast";
export {
  layerNorm as fastLayerNorm,
  rmsNorm as fastRmsNorm,
  rope as fastRoPE,
  scaledDotProductAttention,
} from "./fast";
export { formatShape } from "./format-shape";
export type {
  GgufMetadataValue,
  LoadedGguf,
  LoadedSafetensors,
  SafetensorTensorChunkEntry,
  SafetensorTensorEntry,
} from "./io";
export {
  iterateSafetensors,
  iterateSafetensorTensorChunks,
  loadGguf,
  loadSafetensors,
  parseGgufMetadataJson,
  saveGguf,
  saveSafetensors,
} from "./io";
export type { MemoryStats } from "./memory";
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
} from "./memory";
export { isMetalAvailable, startMetalCapture, stopMetalCapture } from "./metal";
export type { Operand, SoftmaxOptions } from "./ops";
export {
  abs,
  add,
  argmax,
  argmin,
  argpartition,
  argsort,
  asType,
  broadcastTo,
  concatenate,
  cumsum,
  divide,
  equal,
  erf,
  exp,
  expandDims,
  flatten,
  geluApprox,
  greater,
  greaterEqual,
  less,
  lessEqual,
  log,
  logsumexp,
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
  putAlongAxis,
  reciprocal,
  repeat,
  reshape,
  sigmoid,
  slice,
  sliceDynamic,
  sliceUpdate,
  sliceUpdateDynamic,
  softmax,
  sort,
  split,
  sqrt,
  square,
  squeeze,
  stack,
  stopGradient,
  subtract,
  sum,
  takeAlongAxis,
  takeAxis,
  tanh,
  tile,
  topk,
  transpose,
  tril,
  triu,
  where,
} from "./ops";
export type {
  DequantizeOptions,
  QuantizationMode,
  QuantizedMatmulOptions,
  QuantizeOptions,
  QuantizeResult,
} from "./quantization";
export { dequantize, quantize, quantizedMatmul } from "./quantization";
// biome-ignore lint/performance/noReExportAll: Intentional namespace re-export — matches MLX Python's mx.random.normal() API pattern
export * as random from "./random";
export type { CompileMode, DisposableTransform } from "./transforms";
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
} from "./transforms";
export type { FlatEntry, ParameterTree } from "./tree";
export { isParameterTree, treeFlatten, treeLeaves, treeMap, treeUnflatten } from "./tree";

export const VERSION = "0.0.1";
