/**
 * Barrel re-export of all tensor operations.
 * @module
 */

export type { Operand } from "./arithmetic";
export {
  abs,
  add,
  cos,
  divide,
  erf,
  exp,
  geluApprox,
  log,
  log10,
  maximum,
  minimum,
  multiply,
  negative,
  power,
  reciprocal,
  sigmoid,
  sin,
  sqrt,
  square,
  subtract,
  tanh,
} from "./arithmetic";
export {
  equal,
  greater,
  greaterEqual,
  less,
  lessEqual,
  notEqual,
  where,
} from "./comparison";
export { rfft } from "./fft";
export type { GatherMmOptions } from "./linalg";
export { conv1d, conv2d, conv3d, gatherMm, matmul } from "./linalg";
export type { PadMode } from "./padding";
export { pad } from "./padding";
export type { SoftmaxOptions } from "./reduction";
export {
  argmax,
  argmin,
  argpartition,
  argsort,
  cumsum,
  logsumexp,
  max,
  mean,
  min,
  softmax,
  sort,
  sum,
  topk,
} from "./reduction";
export {
  arrayAssignInPlace,
  asType,
  broadcastTo,
  concatenate,
  contiguous,
  expandDims,
  flatten,
  maskedScatter,
  putAlongAxis,
  repeat,
  reshape,
  slice,
  sliceDynamic,
  sliceUpdate,
  sliceUpdateDynamic,
  sliceUpdateInPlace,
  sliceViewInPlace,
  split,
  squeeze,
  stack,
  stopGradient,
  takeAlongAxis,
  takeAxis,
  tile,
  transpose,
  tril,
  triu,
} from "./shape";
export { asStrided } from "./strides";
export { hanning } from "./windows";
