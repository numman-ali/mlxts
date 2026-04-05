/**
 * Runtime config parsing helpers for supported pretrained families.
 * @module
 */

import { ConfigParseError } from "../types";

function valueDescription(value: unknown): string {
  if (Array.isArray(value)) {
    return "array";
  }
  return value === null ? "null" : typeof value;
}

function formatField(context: string, key: string): string {
  return `${context}.${key}`;
}

export function expectConfigRecord(value: unknown, context: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new ConfigParseError(`${context} must be a JSON object.`);
  }
  return Object.fromEntries(Object.entries(value));
}

export function expectString(
  record: Record<string, unknown>,
  key: string,
  context: string,
): string {
  const value = record[key];
  if (typeof value !== "string" || value === "") {
    throw new ConfigParseError(
      `${formatField(context, key)} must be a non-empty string, got ${valueDescription(value)}.`,
    );
  }
  return value;
}

export function expectInteger(
  record: Record<string, unknown>,
  key: string,
  context: string,
): number {
  const value = record[key];
  if (typeof value !== "number" || !Number.isInteger(value)) {
    throw new ConfigParseError(
      `${formatField(context, key)} must be an integer, got ${valueDescription(value)}.`,
    );
  }
  return value;
}

export function optionalInteger(
  record: Record<string, unknown>,
  key: string,
  context: string,
): number | undefined {
  const value = record[key];
  if (value === undefined) {
    return undefined;
  }
  if (typeof value !== "number" || !Number.isInteger(value)) {
    throw new ConfigParseError(
      `${formatField(context, key)} must be an integer when present, got ${valueDescription(value)}.`,
    );
  }
  return value;
}

export function optionalNumber(
  record: Record<string, unknown>,
  key: string,
  context: string,
): number | undefined {
  const value = record[key];
  if (value === undefined) {
    return undefined;
  }
  if (typeof value !== "number") {
    throw new ConfigParseError(
      `${formatField(context, key)} must be a number when present, got ${valueDescription(value)}.`,
    );
  }
  return value;
}

export function optionalBoolean(
  record: Record<string, unknown>,
  key: string,
  context: string,
): boolean | undefined {
  const value = record[key];
  if (value === undefined) {
    return undefined;
  }
  if (typeof value !== "boolean") {
    throw new ConfigParseError(
      `${formatField(context, key)} must be a boolean when present, got ${valueDescription(value)}.`,
    );
  }
  return value;
}

export function optionalString(
  record: Record<string, unknown>,
  key: string,
  context: string,
): string | undefined {
  const value = record[key];
  if (value === undefined) {
    return undefined;
  }
  if (typeof value !== "string" || value === "") {
    throw new ConfigParseError(
      `${formatField(context, key)} must be a non-empty string when present, got ${valueDescription(value)}.`,
    );
  }
  return value;
}
