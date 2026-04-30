/**
 * Shared scalar readers for serve CLI flags.
 * @module
 */

export function readStringFlag(flag: string, value: string | undefined): string {
  if (value === undefined || value.trim() === "") {
    throw new Error(`Missing value for ${flag}.`);
  }
  return value;
}

export function readIntegerFlag(
  flag: string,
  value: string | undefined,
  isValid: (value: number) => boolean,
  description: string,
): number {
  const raw = readStringFlag(flag, value);
  const parsed = Number(raw);
  if (!Number.isInteger(parsed) || !isValid(parsed)) {
    throw new Error(`Expected ${flag} to be ${description}, got "${raw}".`);
  }
  return parsed;
}

export function readNumberFlag(
  flag: string,
  value: string | undefined,
  isValid: (value: number) => boolean,
  description: string,
): number {
  const raw = readStringFlag(flag, value);
  const parsed = Number(raw);
  if (!Number.isFinite(parsed) || !isValid(parsed)) {
    throw new Error(`Expected ${flag} to be ${description}, got "${raw}".`);
  }
  return parsed;
}
