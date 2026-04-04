import type { DType } from "./dtype";

/** Recursive array type for toList() output. */
export type NestedArray<T> = T | NestedArray<T>[];

export type NumericArrayData = readonly (number | NumericArrayData)[];
export type DirectDataDType =
  | "bool"
  | "uint8"
  | "uint16"
  | "uint32"
  | "int8"
  | "int16"
  | "int32"
  | "float32"
  | "float64";
export type DirectTypedArray =
  | Float32Array
  | Float64Array
  | Int8Array
  | Int16Array
  | Int32Array
  | Uint8Array
  | Uint16Array
  | Uint32Array;
export type SupportedTypedArray =
  | Float32Array
  | Float64Array
  | Int8Array
  | Int16Array
  | Int32Array
  | Uint8Array
  | Uint16Array
  | Uint32Array;

type NormalizedInput = {
  inferredDtype: DType;
  inferredShape: number[];
  values: number[];
};

type TypedArrayMatcher = {
  dtype: DirectDataDType;
  is: (value: unknown) => value is SupportedTypedArray;
};

export function expectPresent<T>(value: T | undefined, label: string): T {
  if (value === undefined) {
    throw new Error(`${label} was unexpectedly undefined`);
  }

  return value;
}

const TYPED_ARRAY_MATCHERS: readonly TypedArrayMatcher[] = [
  { dtype: "float32", is: (value): value is Float32Array => value instanceof Float32Array },
  { dtype: "float64", is: (value): value is Float64Array => value instanceof Float64Array },
  { dtype: "int8", is: (value): value is Int8Array => value instanceof Int8Array },
  { dtype: "int16", is: (value): value is Int16Array => value instanceof Int16Array },
  { dtype: "int32", is: (value): value is Int32Array => value instanceof Int32Array },
  { dtype: "uint8", is: (value): value is Uint8Array => value instanceof Uint8Array },
  { dtype: "uint16", is: (value): value is Uint16Array => value instanceof Uint16Array },
  { dtype: "uint32", is: (value): value is Uint32Array => value instanceof Uint32Array },
];

function isSupportedTypedArray(value: unknown): value is SupportedTypedArray {
  return TYPED_ARRAY_MATCHERS.some((matcher) => matcher.is(value));
}

function normalizeTypedArrayInput(
  data: SupportedTypedArray,
  shape?: number[],
  dtype?: DType,
): NormalizedInput {
  for (const matcher of TYPED_ARRAY_MATCHERS) {
    if (matcher.is(data)) {
      return {
        values: Array.from(data),
        inferredShape: shape ?? [data.length],
        inferredDtype: dtype ?? matcher.dtype,
      };
    }
  }

  throw new Error("normalizeTypedArrayInput: unsupported typed array");
}

export function normalizeInput(
  data: NumericArrayData | SupportedTypedArray,
  shape?: number[],
  dtype?: DType,
): NormalizedInput {
  if (isSupportedTypedArray(data)) {
    return normalizeTypedArrayInput(data, shape, dtype);
  }

  const { flat, inferredShape } = flattenArray(data);
  return {
    values: flat,
    inferredShape: shape ?? inferredShape,
    inferredDtype: dtype ?? "float32",
  };
}

export function getDirectStorageDtype(dtype: DType): DirectDataDType {
  switch (dtype) {
    case "bool":
    case "uint8":
    case "uint16":
    case "uint32":
    case "int8":
    case "int16":
    case "int32":
    case "float32":
    case "float64":
      return dtype;
    case "uint64":
    case "int64":
      return "float64";
    case "float16":
    case "bfloat16":
    case "complex64":
      return "float32";
  }
}

export function assertElementCount(elementCount: number, shape: readonly number[]): void {
  const expectedCount =
    shape.length === 0 ? 1 : shape.reduce((product, dimension) => product * dimension, 1);

  if (elementCount !== expectedCount) {
    throw new Error(
      `Data length ${elementCount} does not match shape [${shape.join(", ")}] (${expectedCount} elements)`,
    );
  }
}

export function createDataBuffer(
  values: readonly number[],
  dtype: DirectDataDType,
): DirectTypedArray {
  const source = values.length === 0 ? [0] : values;

  switch (dtype) {
    case "bool":
      return new Uint8Array(source.map((value) => (value === 0 ? 0 : 1)));
    case "uint8":
      return new Uint8Array(source);
    case "uint16":
      return new Uint16Array(source);
    case "uint32":
      return new Uint32Array(source);
    case "int8":
      return new Int8Array(source);
    case "int16":
      return new Int16Array(source);
    case "int32":
      return new Int32Array(source);
    case "float32":
      return new Float32Array(source);
    case "float64":
      return new Float64Array(source);
  }
}

function flattenArray(data: NumericArrayData): { flat: number[]; inferredShape: number[] } {
  const flat: number[] = [];
  return { flat, inferredShape: flattenInto(data, flat) };
}

function flattenInto(data: NumericArrayData | number, flat: number[]): number[] {
  if (typeof data === "number") {
    flat.push(data);
    return [];
  }

  if (data.length === 0) {
    return [0];
  }

  const firstShape = flattenInto(expectPresent(data[0], "nested array item"), flat);
  for (let index = 1; index < data.length; index++) {
    const currentShape = flattenInto(expectPresent(data[index], "nested array item"), flat);
    if (!shapesEqual(firstShape, currentShape)) {
      throw new Error("Nested array data must have a uniform shape");
    }
  }

  return [data.length, ...firstShape];
}

function shapesEqual(left: readonly number[], right: readonly number[]): boolean {
  if (left.length !== right.length) {
    return false;
  }

  for (let index = 0; index < left.length; index++) {
    if (left[index] !== right[index]) {
      return false;
    }
  }

  return true;
}

export function reshapeFlat(flat: number[], shape: readonly number[]): NestedArray<number> {
  if (shape.length === 0) {
    return expectPresent(flat[0], "scalar reshape value");
  }

  if (shape.length === 1) {
    return flat;
  }

  const outerSize = expectPresent(shape[0], "outer shape dimension");
  const innerShape = shape.slice(1);
  const innerSize = innerShape.reduce((product, dimension) => product * dimension, 1);

  const result: NestedArray<number>[] = [];
  for (let index = 0; index < outerSize; index++) {
    result.push(reshapeFlat(flat.slice(index * innerSize, (index + 1) * innerSize), innerShape));
  }
  return result;
}
