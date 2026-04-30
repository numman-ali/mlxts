import { DiffusionConfigError } from "../../errors";

/** Spatial size stored by Diffusers component configs. */
export type StableDiffusionSampleSize = number | readonly [number, number];

export function describeConfigValue(value: unknown): string {
  if (Array.isArray(value)) {
    return "array";
  }
  return value === null ? "null" : typeof value;
}

export function fieldName(context: string, key: string): string {
  return `${context}.${key}`;
}

export function expectRecord(value: unknown, context: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new DiffusionConfigError(`${context} must be a JSON object.`);
  }
  return Object.fromEntries(Object.entries(value));
}

function isPositiveInteger(value: unknown): value is number {
  return Number.isInteger(value) && typeof value === "number" && value > 0;
}

export function requiredPositiveInteger(
  record: Record<string, unknown>,
  key: string,
  context: string,
): number {
  const value = record[key];
  if (!isPositiveInteger(value)) {
    throw new DiffusionConfigError(
      `${fieldName(context, key)} must be a positive integer, got ${describeConfigValue(value)}.`,
    );
  }
  return value;
}

export function optionalPositiveInteger(
  record: Record<string, unknown>,
  key: string,
  context: string,
  fallback: number,
): number {
  const value = record[key];
  if (value === undefined || value === null) {
    return fallback;
  }
  if (!isPositiveInteger(value)) {
    throw new DiffusionConfigError(
      `${fieldName(context, key)} must be a positive integer, got ${describeConfigValue(value)}.`,
    );
  }
  return value;
}

export function optionalNullablePositiveInteger(
  record: Record<string, unknown>,
  key: string,
  context: string,
): number | null {
  const value = record[key];
  if (value === undefined || value === null) {
    return null;
  }
  if (!isPositiveInteger(value)) {
    throw new DiffusionConfigError(
      `${fieldName(context, key)} must be a positive integer or null, got ${describeConfigValue(
        value,
      )}.`,
    );
  }
  return value;
}

export function optionalFiniteNumber(
  record: Record<string, unknown>,
  key: string,
  context: string,
  fallback: number,
): number {
  const value = record[key];
  if (value === undefined || value === null) {
    return fallback;
  }
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new DiffusionConfigError(
      `${fieldName(context, key)} must be a finite number, got ${describeConfigValue(value)}.`,
    );
  }
  return value;
}

export function optionalBoolean(
  record: Record<string, unknown>,
  key: string,
  context: string,
  fallback: boolean,
): boolean {
  const value = record[key];
  if (value === undefined || value === null) {
    return fallback;
  }
  if (typeof value !== "boolean") {
    throw new DiffusionConfigError(
      `${fieldName(context, key)} must be a boolean, got ${describeConfigValue(value)}.`,
    );
  }
  return value;
}

export function requiredPositiveIntegerList(
  record: Record<string, unknown>,
  key: string,
  context: string,
): number[] {
  const value = record[key];
  if (!Array.isArray(value) || value.length === 0) {
    throw new DiffusionConfigError(`${fieldName(context, key)} must be a non-empty integer array.`);
  }
  return value.map((entry, index) => {
    if (!isPositiveInteger(entry)) {
      throw new DiffusionConfigError(
        `${fieldName(context, `${key}[${index}]`)} must be a positive integer.`,
      );
    }
    return entry;
  });
}

export function normalizePositiveIntegerList(
  record: Record<string, unknown>,
  key: string,
  context: string,
  expectedLength: number,
): number[] {
  const value = record[key];
  if (isPositiveInteger(value)) {
    return Array.from({ length: expectedLength }, () => value);
  }
  if (!Array.isArray(value) || value.length !== expectedLength) {
    throw new DiffusionConfigError(
      `${fieldName(context, key)} must be a positive integer or ${expectedLength}-item integer array.`,
    );
  }
  return value.map((entry, index) => {
    if (!isPositiveInteger(entry)) {
      throw new DiffusionConfigError(
        `${fieldName(context, `${key}[${index}]`)} must be a positive integer.`,
      );
    }
    return entry;
  });
}

export function normalizeOptionalPositiveIntegerList(
  record: Record<string, unknown>,
  key: string,
  context: string,
  expectedLength: number,
  fallback: number,
): number[] {
  const value = record[key];
  if (value === undefined || value === null) {
    return Array.from({ length: expectedLength }, () => fallback);
  }
  return normalizePositiveIntegerList(record, key, context, expectedLength);
}

export function optionalSampleSize(
  record: Record<string, unknown>,
  key: string,
  context: string,
): StableDiffusionSampleSize | undefined {
  const value = record[key];
  if (value === undefined || value === null) {
    return undefined;
  }
  if (isPositiveInteger(value)) {
    return value;
  }
  if (Array.isArray(value) && value.length === 2) {
    const first = value[0];
    const second = value[1];
    if (isPositiveInteger(first) && isPositiveInteger(second)) {
      return [first, second];
    }
  }
  throw new DiffusionConfigError(
    `${fieldName(context, key)} must be a positive integer or [height, width].`,
  );
}

export function optionalClassName(
  record: Record<string, unknown>,
  expectedClassName: string,
  context: string,
): void {
  const value = record._class_name;
  if (value === undefined || value === null) {
    return;
  }
  if (value !== expectedClassName) {
    throw new DiffusionConfigError(
      `${fieldName(context, "_class_name")} must be "${expectedClassName}".`,
    );
  }
}

export function rejectNonNull(record: Record<string, unknown>, key: string, context: string): void {
  const value = record[key];
  if (value !== undefined && value !== null) {
    throw new DiffusionConfigError(`${fieldName(context, key)} is not supported yet.`);
  }
}

export function rejectUnsupportedBoolean(
  record: Record<string, unknown>,
  key: string,
  context: string,
  supported: boolean,
): void {
  const value = record[key];
  if (value === undefined || value === null) {
    return;
  }
  if (typeof value !== "boolean") {
    throw new DiffusionConfigError(
      `${fieldName(context, key)} must be a boolean, got ${describeConfigValue(value)}.`,
    );
  }
  if (value !== supported) {
    throw new DiffusionConfigError(`${fieldName(context, key)}=${value} is not supported yet.`);
  }
}

export function rejectUnsupportedBooleanList(
  record: Record<string, unknown>,
  key: string,
  context: string,
  supported: boolean,
): void {
  const value = record[key];
  if (value === undefined || value === null) {
    return;
  }
  if (typeof value === "boolean") {
    if (value !== supported) {
      throw new DiffusionConfigError(`${fieldName(context, key)}=${value} is not supported yet.`);
    }
    return;
  }
  if (Array.isArray(value) && value.every((entry) => entry === supported)) {
    return;
  }
  throw new DiffusionConfigError(`${fieldName(context, key)} is not supported yet.`);
}

export function rejectUnsupportedString(
  record: Record<string, unknown>,
  key: string,
  context: string,
  supported: string,
): void {
  const value = record[key];
  if (value === undefined || value === null) {
    return;
  }
  if (value !== supported) {
    throw new DiffusionConfigError(
      `${fieldName(context, key)}="${String(value)}" is not supported yet.`,
    );
  }
}

export function requiredStringList(
  record: Record<string, unknown>,
  key: string,
  context: string,
  expectedLength: number,
): string[] {
  const value = record[key];
  if (!Array.isArray(value) || value.length !== expectedLength) {
    throw new DiffusionConfigError(
      `${fieldName(context, key)} must be a ${expectedLength}-item string array.`,
    );
  }
  return value.map((entry, index) => {
    if (typeof entry !== "string" || entry === "") {
      throw new DiffusionConfigError(
        `${fieldName(context, `${key}[${index}]`)} must be a non-empty string.`,
      );
    }
    return entry;
  });
}
