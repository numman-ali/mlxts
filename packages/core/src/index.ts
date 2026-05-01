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
export type { MultiOutputFn } from "./ffi/closure-bridge";
export { formatShape } from "./format-shape";
export type {
  GgufMetadataValue,
  InspectedSafetensors,
  LoadedGguf,
  LoadedSafetensors,
  SafetensorByteChunkEntry,
  SafetensorTensorChunkEntry,
  SafetensorTensorEntry,
  SafetensorTensorInfo,
  SafetensorWriteEntry,
  SupportedSafetensorsDType,
} from "./io";
export {
  inspectSafetensors,
  iterateSafetensorByteChunks,
  iterateSafetensors,
  iterateSafetensorTensorChunks,
  loadGguf,
  loadSafetensors,
  parseGgufMetadataJson,
  saveGguf,
  saveSafetensors,
  saveSafetensorsStream,
  tensorBytes,
  toSupportedSafetensorsDType,
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
export type { GatherMmOptions, Operand, PadMode, SoftmaxOptions } from "./ops";
export {
  abs,
  add,
  argmax,
  argmin,
  argpartition,
  argsort,
  arrayAssignInPlace,
  asType,
  broadcastTo,
  concatenate,
  contiguous,
  conv1d,
  conv2d,
  conv3d,
  cos,
  cumsum,
  divide,
  equal,
  erf,
  exp,
  expandDims,
  flatten,
  gatherMm,
  geluApprox,
  greater,
  greaterEqual,
  less,
  lessEqual,
  log,
  logsumexp,
  maskedScatter,
  matmul,
  max,
  maximum,
  mean,
  min,
  minimum,
  multiply,
  negative,
  notEqual,
  pad,
  power,
  putAlongAxis,
  reciprocal,
  repeat,
  reshape,
  sigmoid,
  sin,
  slice,
  sliceDynamic,
  sliceUpdate,
  sliceUpdateDynamic,
  sliceUpdateInPlace,
  sliceViewInPlace,
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
  GatherQmmOptions,
  QuantizationMode,
  QuantizedMatmulOptions,
  QuantizeOptions,
  QuantizeResult,
} from "./quantization";
export { dequantize, gatherQmm, quantize, quantizedMatmul } from "./quantization";
// biome-ignore lint/performance/noReExportAll: Intentional namespace re-export — matches MLX Python's mx.random.normal() API pattern
export * as random from "./random";
export type { DisposableTransform } from "./transforms";
export {
  checkpoint,
  compile,
  compileMany,
  grad,
  mxAsyncEval,
  mxEval,
  valueAndGrad,
} from "./transforms";
export type { FlatEntry, ParameterTree } from "./tree";
export { isParameterTree, treeFlatten, treeLeaves, treeMap, treeUnflatten } from "./tree";

export const VERSION = "0.0.1";
