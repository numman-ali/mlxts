import { DiffusionConfigError } from "../../errors";
import { fieldName, optionalPositiveIntegerList } from "../flux2/config-parsing";

function describeConfigValue(value: unknown): string {
  if (Array.isArray(value)) {
    return "array";
  }
  return value === null ? "null" : typeof value;
}

export function optionalNonNegativeNumber(
  record: Record<string, unknown>,
  key: string,
  context: string,
  fallback: number,
): number {
  const value = record[key];
  if (value === undefined || value === null) {
    return fallback;
  }
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    throw new DiffusionConfigError(
      `${fieldName(context, key)} must be a non-negative finite number, got ${describeConfigValue(
        value,
      )}.`,
    );
  }
  return value;
}

export function optionalPositiveIntegerArray(
  record: Record<string, unknown>,
  key: string,
  context: string,
  fallback: readonly number[],
  expectedLength?: number,
): number[] {
  const values = optionalPositiveIntegerList(record, key, context, fallback);
  if (expectedLength !== undefined && values.length !== expectedLength) {
    throw new DiffusionConfigError(
      `${fieldName(context, key)} must contain ${expectedLength} positive integers.`,
    );
  }
  return values;
}

export function optionalNullablePositiveInteger(
  record: Record<string, unknown>,
  key: string,
  context: string,
  fallback: number | null,
): number | null {
  const value = record[key];
  if (value === undefined) {
    return fallback;
  }
  if (value === null) {
    return null;
  }
  if (typeof value !== "number" || !Number.isInteger(value) || value <= 0) {
    throw new DiffusionConfigError(
      `${fieldName(context, key)} must be a positive integer or null, got ${describeConfigValue(
        value,
      )}.`,
    );
  }
  return value;
}

export function optionalBooleanArray(
  record: Record<string, unknown>,
  key: string,
  context: string,
  fallback: readonly boolean[],
  expectedLength: number,
): boolean[] {
  const value = record[key] ?? fallback;
  if (typeof value === "boolean") {
    return Array.from({ length: expectedLength }, () => value);
  }
  if (!Array.isArray(value) || value.length !== expectedLength) {
    throw new DiffusionConfigError(
      `${fieldName(context, key)} must contain ${expectedLength} boolean values.`,
    );
  }
  return value.map((entry, index) => {
    if (typeof entry !== "boolean") {
      throw new DiffusionConfigError(
        `${fieldName(context, `${key}[${index}]`)} must be a boolean.`,
      );
    }
    return entry;
  });
}

export function optionalStringArray<const T extends string>(
  record: Record<string, unknown>,
  key: string,
  context: string,
  fallback: readonly T[],
  allowed: ReadonlySet<T>,
  expectedLength: number,
): T[] {
  const value = record[key] ?? fallback;
  if (!Array.isArray(value) || value.length !== expectedLength) {
    throw new DiffusionConfigError(
      `${fieldName(context, key)} must contain ${expectedLength} string values.`,
    );
  }
  return value.map((entry, index) => {
    if (typeof entry !== "string") {
      throw new DiffusionConfigError(
        `${fieldName(context, `${key}[${index}]`)}="${String(entry)}" is not supported yet.`,
      );
    }
    for (const candidate of allowed) {
      if (entry === candidate) {
        return candidate;
      }
    }
    throw new DiffusionConfigError(
      `${fieldName(context, `${key}[${index}]`)}="${entry}" is not supported yet.`,
    );
  });
}

export function optionalStringValue<const T extends string>(
  record: Record<string, unknown>,
  key: string,
  context: string,
  fallback: T,
  allowed: ReadonlySet<T>,
): T {
  const value = record[key] ?? fallback;
  if (typeof value !== "string") {
    throw new DiffusionConfigError(
      `${fieldName(context, key)}="${String(value)}" is not supported yet.`,
    );
  }
  for (const candidate of allowed) {
    if (value === candidate) {
      return candidate;
    }
  }
  throw new DiffusionConfigError(`${fieldName(context, key)}="${value}" is not supported yet.`);
}

export function optionalNullableStringValue<const T extends string>(
  record: Record<string, unknown>,
  key: string,
  context: string,
  fallback: T | null,
  allowed: ReadonlySet<T>,
): T | null {
  const value = record[key] ?? fallback;
  if (value === null) {
    return null;
  }
  if (typeof value !== "string") {
    throw new DiffusionConfigError(
      `${fieldName(context, key)}="${String(value)}" is not supported yet.`,
    );
  }
  for (const candidate of allowed) {
    if (value === candidate) {
      return candidate;
    }
  }
  throw new DiffusionConfigError(`${fieldName(context, key)}="${value}" is not supported yet.`);
}

export function optionalPositiveIntegerMatrix(
  record: Record<string, unknown>,
  key: string,
  context: string,
  fallback: readonly (readonly number[])[],
  expectedRows: number,
  expectedColumns: number,
): number[][] {
  const value = record[key] ?? fallback;
  if (!Array.isArray(value) || value.length !== expectedRows) {
    throw new DiffusionConfigError(
      `${fieldName(context, key)} must contain ${expectedRows} integer arrays.`,
    );
  }
  return value.map((row, rowIndex) => {
    if (!Array.isArray(row) || row.length !== expectedColumns) {
      throw new DiffusionConfigError(
        `${fieldName(context, `${key}[${rowIndex}]`)} must contain ${expectedColumns} integers.`,
      );
    }
    return row.map((entry, columnIndex) => {
      if (typeof entry !== "number" || !Number.isInteger(entry) || entry <= 0) {
        throw new DiffusionConfigError(
          `${fieldName(context, `${key}[${rowIndex}][${columnIndex}]`)} must be a positive integer.`,
        );
      }
      return entry;
    });
  });
}
