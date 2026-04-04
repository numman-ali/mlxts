/**
 * Evaluation and transform operations.
 *
 * @module
 */

import type { MxArray } from "./array";
import { checkStatus } from "./error";
import { ffi } from "./ffi/lib";
import { withArrayVector } from "./transforms-base";

export { grad, valueAndGrad } from "./transforms-autograd";
export type { DisposableTransform } from "./transforms-base";
export type { CompileMode } from "./transforms-compile";
export {
  checkpoint,
  clearCompileCache,
  compile,
  disableCompile,
  enableCompile,
  setCompileMode,
} from "./transforms-compile";

/**
 * Force evaluation of one or more lazy arrays.
 * After eval, the arrays' data is computed and available for reading.
 *
 * Named `mxEval` because `eval` is a reserved word in strict-mode JavaScript.
 */
export function mxEval(...arrays: MxArray[]): void {
  if (arrays.length === 0) {
    return;
  }

  withArrayVector(arrays, "eval", (vec) => {
    checkStatus(ffi.mlx_eval(vec), "eval");
  });
}

/**
 * Schedule evaluation of one or more lazy arrays without waiting for completion.
 *
 * Use this when you want to overlap host-side bookkeeping with MLX execution
 * and synchronize explicitly later.
 */
export function mxAsyncEval(...arrays: MxArray[]): void {
  if (arrays.length === 0) {
    return;
  }

  withArrayVector(arrays, "async_eval", (vec) => {
    checkStatus(ffi.mlx_async_eval(vec), "async_eval");
  });
}
