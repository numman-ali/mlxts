import { basename } from "path";

import { DiffusionConfigError } from "../../errors";
import type { DiffusionComponentName } from "../../pretrained/model-index";
import type { DiffusionSnapshotManifest } from "../../pretrained/snapshot-manifest";

function describeConfigValue(value: unknown): string {
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

export function rejectUnknownFields(
  record: Record<string, unknown>,
  knownKeys: ReadonlySet<string>,
  context: string,
): void {
  for (const key of Object.keys(record)) {
    if (!knownKeys.has(key)) {
      throw new DiffusionConfigError(`${fieldName(context, key)} is not supported yet.`);
    }
  }
}

export function optionalClassName(
  record: Record<string, unknown>,
  expected: string,
  context: string,
): void {
  const value = record._class_name;
  if (value === undefined || value === null) {
    return;
  }
  if (value !== expected) {
    throw new DiffusionConfigError(
      `${fieldName(context, "_class_name")}="${String(value)}" is not supported yet.`,
    );
  }
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
  if (typeof value !== "number" || !Number.isInteger(value) || value <= 0) {
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
  fallback: number,
): number {
  const value = record[key];
  if (value === undefined || value === null) {
    return fallback;
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

export function optionalPositiveNumber(
  record: Record<string, unknown>,
  key: string,
  context: string,
  fallback: number,
): number {
  const value = record[key];
  if (value === undefined || value === null) {
    return fallback;
  }
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    throw new DiffusionConfigError(
      `${fieldName(context, key)} must be a positive finite number, got ${describeConfigValue(
        value,
      )}.`,
    );
  }
  return value;
}

export function optionalExactString(
  record: Record<string, unknown>,
  key: string,
  context: string,
  expected: string,
): void {
  const value = record[key];
  if (value === undefined || value === null) {
    return;
  }
  if (value !== expected) {
    throw new DiffusionConfigError(
      `${fieldName(context, key)}="${String(value)}" is not supported; expected "${expected}".`,
    );
  }
}

export function optionalPositiveIntegerList(
  record: Record<string, unknown>,
  key: string,
  context: string,
  fallback?: readonly number[],
): number[] {
  const value = record[key] ?? fallback;
  if (!Array.isArray(value) || value.length === 0) {
    throw new DiffusionConfigError(`${fieldName(context, key)} must be a non-empty integer array.`);
  }
  return value.map((entry, index) => {
    if (typeof entry !== "number" || !Number.isInteger(entry) || entry <= 0) {
      throw new DiffusionConfigError(
        `${fieldName(context, `${key}[${index}]`)} must be a positive integer.`,
      );
    }
    return entry;
  });
}

export function componentConfigPath(
  manifest: DiffusionSnapshotManifest,
  componentName: DiffusionComponentName,
): string {
  const component = manifest.components.find(
    (candidate) => candidate.name === componentName && candidate.enabled,
  );
  const path = component?.metadataPaths.find((candidate) => basename(candidate) === "config.json");
  if (path === undefined) {
    throw new DiffusionConfigError(
      `${componentName}/config.json is missing from the snapshot manifest.`,
    );
  }
  return path;
}

export async function readComponentJson(
  path: string,
  context: string,
): Promise<Record<string, unknown>> {
  const file = Bun.file(path);
  let rawConfig: unknown;
  try {
    rawConfig = await file.json();
  } catch {
    throw new DiffusionConfigError(`${context} must contain valid JSON: ${path}.`);
  }
  return expectRecord(rawConfig, context);
}

export function pipelineBooleanConfig(
  manifest: DiffusionSnapshotManifest,
  key: string,
  fallback: boolean,
): boolean {
  const value = manifest.modelIndex.pipelineConfig[key];
  if (value === undefined) {
    return fallback;
  }
  if (typeof value !== "boolean") {
    throw new DiffusionConfigError(`model_index.json.${key} must be a boolean.`);
  }
  return value;
}
