/**
 * MLX data types and their mappings to mlx-c integer constants.
 *
 * Values must match the mlx_dtype enum in mlx/c/array.h exactly.
 * @module
 */

/** Supported array element types. */
export type DType =
  | "bool"
  | "uint8"
  | "uint16"
  | "uint32"
  | "uint64"
  | "int8"
  | "int16"
  | "int32"
  | "int64"
  | "float16"
  | "float32"
  | "float64"
  | "bfloat16"
  | "complex64";

/**
 * Maps DType strings to mlx-c enum values.
 * Order matches the mlx_dtype enum in mlx/c/array.h.
 */
export const DTYPE_TO_MLX = {
  bool: 0,
  uint8: 1,
  uint16: 2,
  uint32: 3,
  uint64: 4,
  int8: 5,
  int16: 6,
  int32: 7,
  int64: 8,
  float16: 9,
  float32: 10,
  float64: 11,
  bfloat16: 12,
  complex64: 13,
} as const satisfies Record<DType, number>;

/** Maps mlx-c enum values back to DType strings. */
export const MLX_TO_DTYPE: Readonly<Record<number, DType>> = {
  0: "bool",
  1: "uint8",
  2: "uint16",
  3: "uint32",
  4: "uint64",
  5: "int8",
  6: "int16",
  7: "int32",
  8: "int64",
  9: "float16",
  10: "float32",
  11: "float64",
  12: "bfloat16",
  13: "complex64",
};

/** Byte size of each dtype. */
export const DTYPE_BYTE_SIZE = {
  bool: 1,
  uint8: 1,
  uint16: 2,
  uint32: 4,
  uint64: 8,
  int8: 1,
  int16: 2,
  int32: 4,
  int64: 8,
  float16: 2,
  float32: 4,
  float64: 8,
  bfloat16: 2,
  complex64: 8,
} as const satisfies Record<DType, number>;

/**
 * Maps dtypes to TypedArray constructors for data extraction.
 * Only includes types that have direct TypedArray equivalents.
 */
export const DTYPE_TYPED_ARRAY = {
  float32: Float32Array,
  float64: Float64Array,
  int8: Int8Array,
  int16: Int16Array,
  int32: Int32Array,
  uint8: Uint8Array,
  uint16: Uint16Array,
  uint32: Uint32Array,
} as const;

/** Integer dtypes accepted for index and target tensors. */
export const INTEGER_DTYPES = new Set<DType>([
  "int8",
  "int16",
  "int32",
  "int64",
  "uint8",
  "uint16",
  "uint32",
  "uint64",
]);

/** Whether a dtype is an integer dtype. */
export function isIntegerDType(dtype: DType): boolean {
  return INTEGER_DTYPES.has(dtype);
}
