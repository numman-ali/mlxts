/**
 * Evaluation and transform operations.
 *
 * MLX operations are lazy — they build a computation graph.
 * Call mxEval() to force execution on the GPU/CPU.
 * @module
 */

import type { MxArray } from "./array";
import { checkStatus } from "./error";
import { ffi, unwrapPointer } from "./ffi";

/**
 * Force evaluation of one or more lazy arrays.
 * After eval, the arrays' data is computed and available for reading.
 *
 * Named `mxEval` because `eval` is a reserved word in strict-mode JavaScript.
 */
export function mxEval(...arrays: MxArray[]): void {
  if (arrays.length === 0) return;

  const vec = unwrapPointer(ffi.mlx_vector_array_new(), "mlx_vector_array_new");
  try {
    for (const arr of arrays) {
      checkStatus(ffi.mlx_vector_array_append_value(vec, arr._ctx), "eval_vec_append");
    }

    checkStatus(ffi.mlx_eval(vec), "eval");
  } finally {
    ffi.mlx_vector_array_free(vec);
  }
}
