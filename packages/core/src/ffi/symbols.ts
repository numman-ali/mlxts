/**
 * FFI symbol declarations for libmlxc.dylib.
 *
 * Each constant group maps to a domain in the mlx-c header structure
 * (array.h, closure.h, transforms.h, vector.h, etc.). Groups are spread
 * into a single dlopen call in lib.ts.
 *
 * Conventions (from mlx-c headers):
 * - Creation functions return `mlx_array` by value → FFIType.ptr
 * - Operations write results via `mlx_array*` (first arg) → FFIType.ptr to an 8-byte buffer
 * - All operations return int (0 = success) → FFIType.i32
 * - Property getters return values directly (size_t, const int*, etc.)
 * - `mlx_array` by value is a struct { void* ctx } — on ARM64, just a pointer
 *
 * @module
 */

import { FFIType } from "bun:ffi";

// --- Type shorthands for readability ---
const {
  ptr: P,
  i32: I32,
  f32: F32,
  f64: F64,
  bool: BOOL,
  void: VOID,
  cstring: CSTRING,
  u64_fast: U64_FAST,
} = FFIType;

// ---------------------------------------------------------------------------
// Error handling (error.h)
// ---------------------------------------------------------------------------

export const ERROR_SYMBOLS = {
  // void mlx_set_error_handler(handler, data, dtor)
  mlx_set_error_handler: { args: [P, P, P], returns: VOID },
} as const;

// ---------------------------------------------------------------------------
// Array lifecycle (array.h)
// ---------------------------------------------------------------------------

export const ARRAY_LIFECYCLE_SYMBOLS = {
  // Returns mlx_array by value (ctx pointer)
  mlx_array_new: { args: [], returns: P },
  mlx_array_new_data: { args: [P, P, I32, I32], returns: P },
  mlx_array_new_bool: { args: [BOOL], returns: P },
  mlx_array_new_int: { args: [I32], returns: P },
  mlx_array_new_float32: { args: [F32], returns: P },
  mlx_array_new_float: { args: [F32], returns: P },
  mlx_array_new_float64: { args: [F64], returns: P },

  // int mlx_array_set(mlx_array* arr, const mlx_array src)
  mlx_array_set: { args: [P, P], returns: I32 },

  // int mlx_array_free(mlx_array arr)
  mlx_array_free: { args: [P], returns: I32 },

  // int mlx_array_eval(mlx_array arr)
  mlx_array_eval: { args: [P], returns: I32 },

  // Property getters — return directly, no output pointer
  mlx_array_ndim: { args: [P], returns: U64_FAST },
  mlx_array_shape: { args: [P], returns: P },
  mlx_array_dtype: { args: [P], returns: I32 },
  mlx_array_size: { args: [P], returns: U64_FAST },
  mlx_array_itemsize: { args: [P], returns: U64_FAST },
  mlx_array_nbytes: { args: [P], returns: U64_FAST },

  // Scalar item extraction (output pointer for the value)
  mlx_array_item_bool: { args: [P, P], returns: I32 },
  mlx_array_item_uint8: { args: [P, P], returns: I32 },
  mlx_array_item_uint16: { args: [P, P], returns: I32 },
  mlx_array_item_uint32: { args: [P, P], returns: I32 },
  mlx_array_item_int8: { args: [P, P], returns: I32 },
  mlx_array_item_int16: { args: [P, P], returns: I32 },
  mlx_array_item_int32: { args: [P, P], returns: I32 },
  mlx_array_item_float32: { args: [P, P], returns: I32 },
  mlx_array_item_float64: { args: [P, P], returns: I32 },
  mlx_array_item_bfloat16: { args: [P, P], returns: I32 },

  // Data pointer access (returns pointer to raw data, array must be evaluated)
  mlx_array_data_bool: { args: [P], returns: P },
  mlx_array_data_uint8: { args: [P], returns: P },
  mlx_array_data_uint16: { args: [P], returns: P },
  mlx_array_data_uint32: { args: [P], returns: P },
  mlx_array_data_int8: { args: [P], returns: P },
  mlx_array_data_int16: { args: [P], returns: P },
  mlx_array_data_int32: { args: [P], returns: P },
  mlx_array_data_int64: { args: [P], returns: P },
  mlx_array_data_float16: { args: [P], returns: P },
  mlx_array_data_bfloat16: { args: [P], returns: P },
  mlx_array_data_float32: { args: [P], returns: P },
  mlx_array_data_float64: { args: [P], returns: P },

  // String representation
  mlx_array_tostring: { args: [P, P], returns: I32 },
} as const;

// ---------------------------------------------------------------------------
// Strings (string.h)
// ---------------------------------------------------------------------------

export const STRING_SYMBOLS = {
  mlx_string_new: { args: [], returns: P },
  mlx_string_new_data: { args: [CSTRING], returns: P },
  mlx_string_set: { args: [P, P], returns: I32 },
  mlx_string_free: { args: [P], returns: I32 },
  mlx_string_data: { args: [P], returns: CSTRING },
} as const;

// ---------------------------------------------------------------------------
// Map containers (map.h)
// ---------------------------------------------------------------------------

export const MAP_SYMBOLS = {
  mlx_map_string_to_array_new: { args: [], returns: P },
  mlx_map_string_to_array_free: { args: [P], returns: I32 },
  mlx_map_string_to_array_insert: { args: [P, CSTRING, P], returns: I32 },
  mlx_map_string_to_array_get: { args: [P, P, CSTRING], returns: I32 },
  mlx_map_string_to_string_new: { args: [], returns: P },
  mlx_map_string_to_string_free: { args: [P], returns: I32 },
  mlx_map_string_to_string_insert: { args: [P, CSTRING, CSTRING], returns: I32 },
} as const;

// ---------------------------------------------------------------------------
// Device & stream (device.h, stream.h)
// ---------------------------------------------------------------------------

export const DEVICE_SYMBOLS = {
  mlx_device_new: { args: [], returns: P },
  mlx_device_new_type: { args: [I32, I32], returns: P },
  mlx_device_free: { args: [P], returns: I32 },
  mlx_device_get_type: { args: [P, P], returns: I32 },
  mlx_device_info_get: { args: [P, P], returns: I32 },
  mlx_device_info_free: { args: [P], returns: I32 },
  mlx_device_info_has_key: { args: [P, P, CSTRING], returns: I32 },
  mlx_device_info_get_size: { args: [P, P, CSTRING], returns: I32 },
  mlx_device_is_available: { args: [P, P], returns: I32 },
  mlx_device_count: { args: [P, I32], returns: I32 },
  mlx_get_default_device: { args: [P], returns: I32 },
  mlx_set_default_device: { args: [P], returns: I32 },
} as const;

export const STREAM_SYMBOLS = {
  mlx_stream_new: { args: [], returns: P },
  mlx_stream_new_device: { args: [P], returns: P },
  mlx_stream_free: { args: [P], returns: I32 },
  mlx_default_cpu_stream_new: { args: [], returns: P },
  mlx_default_gpu_stream_new: { args: [], returns: P },
  mlx_get_default_stream: { args: [P, P], returns: I32 },
  mlx_set_default_stream: { args: [P], returns: I32 },
  mlx_synchronize: { args: [P], returns: I32 },
} as const;

// ---------------------------------------------------------------------------
// Vector containers (vector.h)
// ---------------------------------------------------------------------------

export const VECTOR_SYMBOLS = {
  mlx_vector_array_new: { args: [], returns: P },
  mlx_vector_array_new_data: { args: [P, U64_FAST], returns: P },
  mlx_vector_array_new_value: { args: [P], returns: P },
  mlx_vector_array_free: { args: [P], returns: I32 },
  mlx_vector_array_set_value: { args: [P, P], returns: I32 },
  mlx_vector_array_append_value: { args: [P, P], returns: I32 },
  mlx_vector_array_size: { args: [P], returns: U64_FAST },
  mlx_vector_array_get: { args: [P, P, U64_FAST], returns: I32 },
} as const;

// ---------------------------------------------------------------------------
// Arithmetic operations (ops.h)
// ---------------------------------------------------------------------------

export const ARITHMETIC_SYMBOLS = {
  // int mlx_op(mlx_array* res, const mlx_array a, [const mlx_array b,] const mlx_stream s)
  mlx_add: { args: [P, P, P, P], returns: I32 },
  mlx_subtract: { args: [P, P, P, P], returns: I32 },
  mlx_multiply: { args: [P, P, P, P], returns: I32 },
  mlx_divide: { args: [P, P, P, P], returns: I32 },
  mlx_power: { args: [P, P, P, P], returns: I32 },
  mlx_maximum: { args: [P, P, P, P], returns: I32 },
  mlx_minimum: { args: [P, P, P, P], returns: I32 },
  mlx_negative: { args: [P, P, P], returns: I32 },
  mlx_abs: { args: [P, P, P], returns: I32 },
  mlx_sqrt: { args: [P, P, P], returns: I32 },
  mlx_square: { args: [P, P, P], returns: I32 },
  mlx_exp: { args: [P, P, P], returns: I32 },
  mlx_log: { args: [P, P, P], returns: I32 },
  mlx_sigmoid: { args: [P, P, P], returns: I32 },
  mlx_erf: { args: [P, P, P], returns: I32 },
  mlx_reciprocal: { args: [P, P, P], returns: I32 },
  mlx_floor: { args: [P, P, P], returns: I32 },
  mlx_ceil: { args: [P, P, P], returns: I32 },
  mlx_tanh: { args: [P, P, P], returns: I32 },
  mlx_sin: { args: [P, P, P], returns: I32 },
  mlx_cos: { args: [P, P, P], returns: I32 },
} as const;

export const MLXTS_NATIVE_SYMBOLS = {
  mlxts_gelu_approx: { args: [P, P, P], returns: I32 },
  mlxts_conv2d: { args: [P, P, P, P, U64_FAST, P], returns: I32 },
  mlxts_conv3d: { args: [P, P, P, P, U64_FAST, P], returns: I32 },
  mlxts_qwen_gated_delta_update: { args: [P, P, P, P, P, P, P, P, P], returns: I32 },
  mlxts_qwen_gated_delta_update_masked: {
    args: [P, P, P, P, P, P, P, P, P, P],
    returns: I32,
  },
  mlxts_array_assign_inplace: { args: [P, P], returns: I32 },
  mlxts_slice_update_inplace: {
    args: [P, P, P, U64_FAST, P, U64_FAST, P, U64_FAST, P],
    returns: I32,
  },
  mlxts_slice_view_inplace: {
    args: [P, P, P, U64_FAST, P, U64_FAST, P, U64_FAST, P],
    returns: I32,
  },
  mlxts_load_gguf: { args: [P, P, CSTRING, P], returns: I32 },
  mlxts_save_gguf: { args: [CSTRING, P, P], returns: I32 },
  mlxts_map_string_to_array_keys: { args: [P, P], returns: I32 },
} as const;

// ---------------------------------------------------------------------------
// Reductions (ops.h)
// ---------------------------------------------------------------------------

export const REDUCTION_SYMBOLS = {
  // Full reduction: int mlx_sum(res, a, keepdims, stream)
  mlx_sum: { args: [P, P, BOOL, P], returns: I32 },
  mlx_sum_axis: { args: [P, P, I32, BOOL, P], returns: I32 },
  mlx_sum_axes: { args: [P, P, P, U64_FAST, BOOL, P], returns: I32 },
  mlx_mean: { args: [P, P, BOOL, P], returns: I32 },
  mlx_mean_axis: { args: [P, P, I32, BOOL, P], returns: I32 },
  mlx_mean_axes: { args: [P, P, P, U64_FAST, BOOL, P], returns: I32 },
  mlx_max: { args: [P, P, BOOL, P], returns: I32 },
  mlx_max_axis: { args: [P, P, I32, BOOL, P], returns: I32 },
  mlx_max_axes: { args: [P, P, P, U64_FAST, BOOL, P], returns: I32 },
  mlx_min: { args: [P, P, BOOL, P], returns: I32 },
  mlx_min_axis: { args: [P, P, I32, BOOL, P], returns: I32 },
  mlx_min_axes: { args: [P, P, P, U64_FAST, BOOL, P], returns: I32 },
  mlx_argmax: { args: [P, P, BOOL, P], returns: I32 },
  mlx_argmax_axis: { args: [P, P, I32, BOOL, P], returns: I32 },
  mlx_argmin: { args: [P, P, BOOL, P], returns: I32 },
  mlx_argmin_axis: { args: [P, P, I32, BOOL, P], returns: I32 },
  mlx_argpartition: { args: [P, P, I32, P], returns: I32 },
  mlx_argpartition_axis: { args: [P, P, I32, I32, P], returns: I32 },
  mlx_argsort: { args: [P, P, P], returns: I32 },
  mlx_argsort_axis: { args: [P, P, I32, P], returns: I32 },
  mlx_cumsum: { args: [P, P, I32, BOOL, BOOL, P], returns: I32 },
  mlx_logsumexp: { args: [P, P, BOOL, P], returns: I32 },
  mlx_logsumexp_axis: { args: [P, P, I32, BOOL, P], returns: I32 },
  mlx_logsumexp_axes: { args: [P, P, P, U64_FAST, BOOL, P], returns: I32 },
  mlx_sort: { args: [P, P, P], returns: I32 },
  mlx_sort_axis: { args: [P, P, I32, P], returns: I32 },
  mlx_topk: { args: [P, P, I32, P], returns: I32 },
  mlx_topk_axis: { args: [P, P, I32, I32, P], returns: I32 },
} as const;

// ---------------------------------------------------------------------------
// Shape operations (ops.h)
// ---------------------------------------------------------------------------

export const SHAPE_SYMBOLS = {
  // int mlx_reshape(res, a, shape, shape_num, stream)
  mlx_reshape: { args: [P, P, P, U64_FAST, P], returns: I32 },
  mlx_transpose: { args: [P, P, P], returns: I32 },
  mlx_transpose_axes: { args: [P, P, P, U64_FAST, P], returns: I32 },
  mlx_squeeze: { args: [P, P, P], returns: I32 },
  mlx_squeeze_axis: { args: [P, P, I32, P], returns: I32 },
  mlx_expand_dims: { args: [P, P, I32, P], returns: I32 },
  mlx_broadcast_to: { args: [P, P, P, U64_FAST, P], returns: I32 },
  mlx_astype: { args: [P, P, I32, P], returns: I32 },
  mlx_concatenate_axis: { args: [P, P, I32, P], returns: I32 },
  mlx_concatenate: { args: [P, P, P], returns: I32 },
  mlx_stack_axis: { args: [P, P, I32, P], returns: I32 },
  mlx_stack: { args: [P, P, P], returns: I32 },
  mlx_split: { args: [P, P, I32, I32, P], returns: I32 },
  mlx_flatten: { args: [P, P, I32, I32, P], returns: I32 },
  mlx_contiguous: { args: [P, P, BOOL, P], returns: I32 },
  mlx_stop_gradient: { args: [P, P, P], returns: I32 },
  mlx_repeat_axis: { args: [P, P, I32, I32, P], returns: I32 },
  mlx_repeat: { args: [P, P, I32, P], returns: I32 },
  mlx_tile: { args: [P, P, P, U64_FAST, P], returns: I32 },
  mlx_pad: { args: [P, P, P, U64_FAST, P, U64_FAST, P, U64_FAST, P, CSTRING, P], returns: I32 },
  // int mlx_tril(res, x, k, stream)
  mlx_tril: { args: [P, P, I32, P], returns: I32 },
  // int mlx_triu(res, x, k, stream)
  mlx_triu: { args: [P, P, I32, P], returns: I32 },
} as const;

// ---------------------------------------------------------------------------
// Linear algebra (linalg.h)
// ---------------------------------------------------------------------------

export const LINALG_SYMBOLS = {
  mlx_matmul: { args: [P, P, P, P], returns: I32 },
  mlx_conv1d: { args: [P, P, P, I32, I32, I32, I32, P], returns: I32 },
} as const;

// ---------------------------------------------------------------------------
// Comparison operations (ops.h)
// ---------------------------------------------------------------------------

export const COMPARISON_SYMBOLS = {
  mlx_equal: { args: [P, P, P, P], returns: I32 },
  mlx_not_equal: { args: [P, P, P, P], returns: I32 },
  mlx_greater: { args: [P, P, P, P], returns: I32 },
  mlx_greater_equal: { args: [P, P, P, P], returns: I32 },
  mlx_less: { args: [P, P, P, P], returns: I32 },
  mlx_less_equal: { args: [P, P, P, P], returns: I32 },
  mlx_where: { args: [P, P, P, P, P], returns: I32 },
} as const;

// ---------------------------------------------------------------------------
// Array creation (ops.h)
// ---------------------------------------------------------------------------

export const CREATION_SYMBOLS = {
  // int mlx_zeros(res, shape, shape_num, dtype, stream)
  mlx_zeros: { args: [P, P, U64_FAST, I32, P], returns: I32 },
  mlx_ones: { args: [P, P, U64_FAST, I32, P], returns: I32 },
  mlx_full: { args: [P, P, U64_FAST, P, I32, P], returns: I32 },
  mlx_arange: { args: [P, F64, F64, F64, I32, P], returns: I32 },
  mlx_softmax_axis: { args: [P, P, I32, BOOL, P], returns: I32 },
  mlx_softmax: { args: [P, P, BOOL, P], returns: I32 },
  mlx_masked_scatter: { args: [P, P, P, P, P], returns: I32 },
  mlx_take_along_axis: { args: [P, P, P, I32, P], returns: I32 },
  mlx_put_along_axis: { args: [P, P, P, P, I32, P], returns: I32 },
} as const;

// ---------------------------------------------------------------------------
// Take / gather — index selection along an axis
// ---------------------------------------------------------------------------

export const TAKE_SYMBOLS = {
  // int mlx_take_axis(res, a, indices, axis, stream)
  mlx_take_axis: { args: [P, P, P, I32, P], returns: I32 },
  mlx_gather_mm: { args: [P, P, P, P, P, BOOL, P], returns: I32 },
  mlx_slice: { args: [P, P, P, U64_FAST, P, U64_FAST, P, U64_FAST, P], returns: I32 },
  mlx_slice_dynamic: { args: [P, P, P, P, U64_FAST, P, U64_FAST, P], returns: I32 },
  mlx_slice_update: {
    args: [P, P, P, P, U64_FAST, P, U64_FAST, P, U64_FAST, P],
    returns: I32,
  },
  mlx_slice_update_dynamic: { args: [P, P, P, P, P, U64_FAST, P], returns: I32 },
} as const;

// ---------------------------------------------------------------------------
// Quantization (ops.h)
// ---------------------------------------------------------------------------

export const QUANTIZATION_SYMBOLS = {
  mlx_quantize: {
    args: [P, P, U64_FAST, U64_FAST, CSTRING, P, P],
    returns: I32,
  },
  mlx_dequantize: {
    args: [P, P, P, P, U64_FAST, U64_FAST, CSTRING, P, U64_FAST, P],
    returns: I32,
  },
  mlx_quantized_matmul: {
    args: [P, P, P, P, P, BOOL, U64_FAST, U64_FAST, CSTRING, P],
    returns: I32,
  },
  mlx_gather_qmm: {
    args: [P, P, P, P, P, P, P, BOOL, U64_FAST, U64_FAST, CSTRING, BOOL, P],
    returns: I32,
  },
} as const;

// ---------------------------------------------------------------------------
// Random (random.h)
// ---------------------------------------------------------------------------

export const RANDOM_SYMBOLS = {
  mlx_random_seed: { args: [U64_FAST], returns: I32 },
  mlx_random_key: { args: [P, U64_FAST], returns: I32 },
  // mlx_random_split(res_0, res_1, key, stream)
  mlx_random_split: { args: [P, P, P, P], returns: I32 },
  mlx_random_split_num: { args: [P, P, I32, P], returns: I32 },
  // mlx_random_normal(res, shape, shape_num, dtype, loc, scale, key, stream)
  mlx_random_normal: { args: [P, P, U64_FAST, I32, F32, F32, P, P], returns: I32 },
  // mlx_random_uniform(res, low, high, shape, shape_num, dtype, key, stream)
  mlx_random_uniform: { args: [P, P, P, P, U64_FAST, I32, P, P], returns: I32 },
  // mlx_random_bernoulli(res, p, shape, shape_num, key, stream)
  mlx_random_bernoulli: { args: [P, P, P, U64_FAST, P, P], returns: I32 },
  // mlx_random_categorical(res, logits, axis, key, stream)
  mlx_random_categorical: { args: [P, P, I32, P, P], returns: I32 },
} as const;

// ---------------------------------------------------------------------------
// Memory and profiling controls (memory.h, metal.h)
// ---------------------------------------------------------------------------

export const MEMORY_SYMBOLS = {
  mlx_clear_cache: { args: [], returns: I32 },
  mlx_get_active_memory: { args: [P], returns: I32 },
  mlx_get_cache_memory: { args: [P], returns: I32 },
  mlx_get_memory_limit: { args: [P], returns: I32 },
  mlx_get_peak_memory: { args: [P], returns: I32 },
  mlx_reset_peak_memory: { args: [], returns: I32 },
  mlx_set_cache_limit: { args: [P, U64_FAST], returns: I32 },
  mlx_set_memory_limit: { args: [P, U64_FAST], returns: I32 },
  mlx_set_wired_limit: { args: [P, U64_FAST], returns: I32 },
} as const;

export const METAL_SYMBOLS = {
  mlx_metal_is_available: { args: [P], returns: I32 },
  mlx_metal_start_capture: { args: [CSTRING], returns: I32 },
  mlx_metal_stop_capture: { args: [], returns: I32 },
} as const;

// ---------------------------------------------------------------------------
// Fast operations (fast.h)
// ---------------------------------------------------------------------------

export const FAST_SYMBOLS = {
  mlx_fast_scaled_dot_product_attention: {
    args: [P, P, P, P, F32, CSTRING, P, P, P],
    returns: I32,
  },
  mlx_fast_layer_norm: {
    args: [P, P, P, P, F32, P],
    returns: I32,
  },
  mlx_fast_rms_norm: {
    args: [P, P, P, F32, P],
    returns: I32,
  },
  mlx_fast_rope: {
    args: [P, P, I32, BOOL, U64_FAST, F32, I32, P, P],
    returns: I32,
  },
  mlx_fast_rope_dynamic: {
    args: [P, P, I32, BOOL, U64_FAST, F32, P, P, P],
    returns: I32,
  },
} as const;

// ---------------------------------------------------------------------------
// Transforms (transforms.h)
// ---------------------------------------------------------------------------

export const TRANSFORM_SYMBOLS = {
  // int mlx_eval(const mlx_vector_array outputs)
  mlx_eval: { args: [P], returns: I32 },
  mlx_async_eval: { args: [P], returns: I32 },
  mlx_checkpoint: { args: [P, P], returns: I32 },
  mlx_compile: { args: [P, P, BOOL], returns: I32 },
  mlx_detail_compile_clear_cache: { args: [], returns: I32 },
  mlx_disable_compile: { args: [], returns: I32 },
  mlx_enable_compile: { args: [], returns: I32 },
  mlx_set_compile_mode: { args: [I32], returns: I32 },
} as const;

// ---------------------------------------------------------------------------
// Closures (closure.h)
// ---------------------------------------------------------------------------

export const CLOSURE_SYMBOLS = {
  // mlx_closure mlx_closure_new_func(int (*fun)(mlx_vector_array*, const mlx_vector_array))
  mlx_closure_new_func: { args: [P], returns: P },
  // int mlx_closure_apply(mlx_vector_array* res, mlx_closure cls, const mlx_vector_array input)
  mlx_closure_apply: { args: [P, P, P], returns: I32 },
  // int mlx_closure_free(mlx_closure cls)
  mlx_closure_free: { args: [P], returns: I32 },

  // mlx_closure_value_and_grad lifecycle
  mlx_closure_value_and_grad_new: { args: [], returns: P },
  // int mlx_closure_value_and_grad_apply(res_values*, res_grads*, cls, input)
  mlx_closure_value_and_grad_apply: { args: [P, P, P, P], returns: I32 },
  mlx_closure_value_and_grad_free: { args: [P], returns: I32 },
} as const;

// ---------------------------------------------------------------------------
// Gradient transforms (transforms.h)
// ---------------------------------------------------------------------------

export const GRAD_TRANSFORM_SYMBOLS = {
  // int mlx_value_and_grad(mlx_closure_value_and_grad* res, const mlx_closure fun,
  //                        const int* argnums, size_t argnums_num)
  mlx_value_and_grad: { args: [P, P, P, U64_FAST], returns: I32 },
} as const;
