import { DiffusionConfigError } from "../errors";
import type {
  FlowMatchEulerSchedulerConfig,
  FlowMatchEulerTimeShiftType,
} from "../schedulers/flow-match-euler";

function valueDescription(value: unknown): string {
  if (Array.isArray(value)) {
    return "array";
  }
  return value === null ? "null" : typeof value;
}

function fieldName(context: string, key: string): string {
  return `${context}.${key}`;
}

function optionalNumber(
  record: Record<string, unknown>,
  key: string,
  context: string,
): number | undefined {
  const value = record[key];
  if (value === undefined || value === null) {
    return undefined;
  }
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new DiffusionConfigError(
      `${fieldName(context, key)} must be a finite number when present, got ${valueDescription(
        value,
      )}.`,
    );
  }
  return value;
}

function optionalInteger(
  record: Record<string, unknown>,
  key: string,
  context: string,
): number | undefined {
  const value = record[key];
  if (value === undefined || value === null) {
    return undefined;
  }
  if (typeof value !== "number" || !Number.isInteger(value)) {
    throw new DiffusionConfigError(
      `${fieldName(context, key)} must be an integer when present, got ${valueDescription(value)}.`,
    );
  }
  return value;
}

function optionalBoolean(
  record: Record<string, unknown>,
  key: string,
  context: string,
): boolean | undefined {
  const value = record[key];
  if (value === undefined || value === null) {
    return undefined;
  }
  if (typeof value !== "boolean") {
    throw new DiffusionConfigError(
      `${fieldName(context, key)} must be a boolean when present, got ${valueDescription(value)}.`,
    );
  }
  return value;
}

function optionalString(
  record: Record<string, unknown>,
  key: string,
  context: string,
): string | undefined {
  const value = record[key];
  if (value === undefined || value === null) {
    return undefined;
  }
  if (typeof value !== "string" || value === "") {
    throw new DiffusionConfigError(
      `${fieldName(context, key)} must be a non-empty string when present, got ${valueDescription(
        value,
      )}.`,
    );
  }
  return value;
}

function optionalTimeShiftType(
  record: Record<string, unknown>,
  key: string,
  context: string,
): FlowMatchEulerTimeShiftType | undefined {
  const value = optionalString(record, key, context);
  if (value === undefined) {
    return undefined;
  }
  if (value === "exponential" || value === "linear") {
    return value;
  }
  throw new DiffusionConfigError(
    `${fieldName(context, key)}="${value}" is not a supported FlowMatch time shift type.`,
  );
}

function expectAbsent(record: Record<string, unknown>, key: string, context: string): void {
  const value = record[key];
  if (value !== undefined && value !== null) {
    throw new DiffusionConfigError(`${fieldName(context, key)} is not supported yet.`);
  }
}

function expectBooleanNotTrue(record: Record<string, unknown>, key: string, context: string): void {
  const value = optionalBoolean(record, key, context);
  if (value === true) {
    throw new DiffusionConfigError(`${fieldName(context, key)}=true is not supported yet.`);
  }
}

function rejectUnsupportedFlowMatchFields(record: Record<string, unknown>, context: string): void {
  for (const key of [
    "beta_start",
    "beta_end",
    "beta_schedule",
    "trained_betas",
    "prediction_type",
    "rescale_betas_zero_snr",
    "sigma_min",
    "sigma_max",
  ]) {
    expectAbsent(record, key, context);
  }
  for (const key of [
    "invert_sigmas",
    "stochastic_sampling",
    "use_karras_sigmas",
    "use_exponential_sigmas",
    "use_beta_sigmas",
  ]) {
    expectBooleanNotTrue(record, key, context);
  }
}

/** Parse a Diffusers FlowMatch Euler scheduler config into package-native fields. */
export function parseFlowMatchEulerConfig(
  record: Record<string, unknown>,
  context: string,
): FlowMatchEulerSchedulerConfig {
  rejectUnsupportedFlowMatchFields(record, context);

  const numTrainTimesteps = optionalInteger(record, "num_train_timesteps", context);
  const shift = optionalNumber(record, "shift", context);
  const shiftTerminal = optionalNumber(record, "shift_terminal", context);
  const useDynamicShifting = optionalBoolean(record, "use_dynamic_shifting", context);
  const baseShift = optionalNumber(record, "base_shift", context);
  const maxShift = optionalNumber(record, "max_shift", context);
  const baseImageSeqLen = optionalInteger(record, "base_image_seq_len", context);
  const maxImageSeqLen = optionalInteger(record, "max_image_seq_len", context);
  const timeShiftType = optionalTimeShiftType(record, "time_shift_type", context);

  return {
    ...(numTrainTimesteps !== undefined ? { numTrainTimesteps } : {}),
    ...(shift !== undefined ? { shift } : {}),
    ...(shiftTerminal !== undefined ? { shiftTerminal } : {}),
    ...(useDynamicShifting !== undefined ? { useDynamicShifting } : {}),
    ...(baseShift !== undefined ? { baseShift } : {}),
    ...(maxShift !== undefined ? { maxShift } : {}),
    ...(baseImageSeqLen !== undefined ? { baseImageSeqLen } : {}),
    ...(maxImageSeqLen !== undefined ? { maxImageSeqLen } : {}),
    ...(timeShiftType !== undefined ? { timeShiftType } : {}),
  };
}
