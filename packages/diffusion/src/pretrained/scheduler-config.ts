import { isAbsolute, join, normalize } from "path";
import { DiffusionConfigError } from "../errors";
import { DDIMScheduler, type DDIMSchedulerConfig } from "../schedulers/ddim";
import {
  type EulerFinalSigmasType,
  EulerScheduler,
  type EulerSchedulerConfig,
} from "../schedulers/euler";
import {
  FlowMatchEulerScheduler,
  type FlowMatchEulerSchedulerConfig,
} from "../schedulers/flow-match-euler";
import type {
  BetaSchedule,
  DiffusionScheduleConfig,
  TimestepSpacing,
} from "../schedulers/schedule";
import { parseFlowMatchEulerConfig } from "./flow-match-scheduler-config";

/** Supported scheduler families for local diffusion checkpoint metadata. */
export type DiffusionSchedulerKind = "ddim" | "euler" | "flow-match-euler";

/** Diffusers scheduler class names with implemented scheduler math. */
export type DiffusersSchedulerClassName =
  | "DDIMScheduler"
  | "EulerDiscreteScheduler"
  | "FlowMatchEulerDiscreteScheduler";

/** Parsed scheduler metadata translated into package-native scheduler config. */
export type ParsedDiffusionSchedulerConfig =
  | {
      kind: "ddim";
      className: "DDIMScheduler";
      config: DDIMSchedulerConfig;
    }
  | {
      kind: "euler";
      className: "EulerDiscreteScheduler";
      config: EulerSchedulerConfig;
    }
  | {
      kind: "flow-match-euler";
      className: "FlowMatchEulerDiscreteScheduler";
      config: FlowMatchEulerSchedulerConfig;
    };

/** Scheduler instances currently constructible from Diffusers metadata. */
export type SupportedDiffusionScheduler = DDIMScheduler | EulerScheduler | FlowMatchEulerScheduler;

/** Options for reading a scheduler config from a local snapshot directory. */
export type DiffusionSchedulerLoadOptions = {
  subfolder?: string;
};

/** Loaded scheduler plus the metadata needed to audit how it was constructed. */
export type DiffusionSchedulerLoadResult = {
  scheduler: SupportedDiffusionScheduler;
  parsedConfig: ParsedDiffusionSchedulerConfig;
  className: DiffusersSchedulerClassName;
  configPath: string;
  rawConfig: Record<string, unknown>;
};

function valueDescription(value: unknown): string {
  if (Array.isArray(value)) {
    return "array";
  }
  return value === null ? "null" : typeof value;
}

function fieldName(context: string, key: string): string {
  return `${context}.${key}`;
}

function expectRecord(value: unknown, context: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new DiffusionConfigError(`${context} must be a JSON object.`);
  }
  return Object.fromEntries(Object.entries(value));
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

function optionalBetaSchedule(
  record: Record<string, unknown>,
  key: string,
  context: string,
): BetaSchedule | undefined {
  const value = optionalString(record, key, context);
  if (value === undefined) {
    return undefined;
  }
  if (value === "linear" || value === "scaled_linear") {
    return value;
  }
  throw new DiffusionConfigError(
    `${fieldName(context, key)}="${value}" is not supported by @mlxts/diffusion yet.`,
  );
}

function optionalTimestepSpacing(
  record: Record<string, unknown>,
  key: string,
  context: string,
): TimestepSpacing | undefined {
  const value = optionalString(record, key, context);
  if (value === undefined) {
    return undefined;
  }
  if (value === "leading" || value === "linspace" || value === "trailing") {
    return value;
  }
  throw new DiffusionConfigError(
    `${fieldName(context, key)}="${value}" is not a supported timestep spacing.`,
  );
}

function optionalEulerFinalSigmasType(
  record: Record<string, unknown>,
  key: string,
  context: string,
): EulerFinalSigmasType | undefined {
  const value = optionalString(record, key, context);
  if (value === undefined) {
    return undefined;
  }
  if (value === "zero" || value === "sigma_min") {
    return value;
  }
  throw new DiffusionConfigError(
    `${fieldName(context, key)}="${value}" is not a supported final sigma type.`,
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

function expectStringValue(
  record: Record<string, unknown>,
  key: string,
  expected: string,
  context: string,
): void {
  const value = optionalString(record, key, context);
  if (value !== undefined && value !== expected) {
    throw new DiffusionConfigError(
      `${fieldName(context, key)}="${value}" is not supported; expected "${expected}".`,
    );
  }
}

function parseBaseScheduleConfig(
  record: Record<string, unknown>,
  context: string,
): DiffusionScheduleConfig {
  const config: DiffusionScheduleConfig = {};
  const numTrainTimesteps = optionalInteger(record, "num_train_timesteps", context);
  const betaStart = optionalNumber(record, "beta_start", context);
  const betaEnd = optionalNumber(record, "beta_end", context);
  const betaSchedule = optionalBetaSchedule(record, "beta_schedule", context);

  if (numTrainTimesteps !== undefined) {
    config.numTrainTimesteps = numTrainTimesteps;
  }
  if (betaStart !== undefined) {
    config.betaStart = betaStart;
  }
  if (betaEnd !== undefined) {
    config.betaEnd = betaEnd;
  }
  if (betaSchedule !== undefined) {
    config.betaSchedule = betaSchedule;
  }
  return config;
}

function parseClassName(
  record: Record<string, unknown>,
  context: string,
): DiffusersSchedulerClassName {
  const className = optionalString(record, "_class_name", context);
  if (
    className === "DDIMScheduler" ||
    className === "EulerDiscreteScheduler" ||
    className === "FlowMatchEulerDiscreteScheduler"
  ) {
    return className;
  }
  if (className === undefined) {
    throw new DiffusionConfigError(`${fieldName(context, "_class_name")} is required.`);
  }
  throw new DiffusionConfigError(
    `${fieldName(context, "_class_name")}="${className}" is not supported yet.`,
  );
}

function normalizeSchedulerSubfolder(options: DiffusionSchedulerLoadOptions): string {
  const subfolder = options.subfolder ?? "scheduler";
  if (subfolder === "" || subfolder === ".") {
    return "";
  }
  const normalized = normalize(subfolder);
  if (isAbsolute(subfolder) || normalized === ".." || normalized.startsWith("../")) {
    throw new DiffusionConfigError("loadDiffusionSchedulerConfig: subfolder must stay relative.");
  }
  return normalized;
}

function schedulerConfigPath(
  snapshotDirectory: string,
  options: DiffusionSchedulerLoadOptions,
): string {
  const subfolder = normalizeSchedulerSubfolder(options);
  if (subfolder === "") {
    return join(snapshotDirectory, "scheduler_config.json");
  }
  return join(snapshotDirectory, subfolder, "scheduler_config.json");
}

function rejectUnsupportedCommonFields(record: Record<string, unknown>, context: string): void {
  expectAbsent(record, "trained_betas", context);
  expectStringValue(record, "prediction_type", "epsilon", context);
  expectBooleanNotTrue(record, "rescale_betas_zero_snr", context);
}

function parseDDIMConfig(record: Record<string, unknown>, context: string): DDIMSchedulerConfig {
  rejectUnsupportedCommonFields(record, context);
  expectBooleanNotTrue(record, "thresholding", context);

  const config: DDIMSchedulerConfig = parseBaseScheduleConfig(record, context);
  const setAlphaToOne = optionalBoolean(record, "set_alpha_to_one", context);
  const clipSample = optionalBoolean(record, "clip_sample", context);
  const clipSampleRange = optionalNumber(record, "clip_sample_range", context);
  const timestepSpacing = optionalTimestepSpacing(record, "timestep_spacing", context);
  const stepsOffset = optionalInteger(record, "steps_offset", context);

  if (setAlphaToOne !== undefined) {
    config.setAlphaToOne = setAlphaToOne;
  }
  if (clipSample !== undefined) {
    config.clipSample = clipSample;
  }
  if (clipSampleRange !== undefined) {
    config.clipSampleRange = clipSampleRange;
  }
  if (timestepSpacing !== undefined) {
    config.timestepSpacing = timestepSpacing;
  }
  if (stepsOffset !== undefined) {
    config.stepsOffset = stepsOffset;
  }
  return config;
}

function parseEulerConfig(record: Record<string, unknown>, context: string): EulerSchedulerConfig {
  rejectUnsupportedCommonFields(record, context);
  expectStringValue(record, "interpolation_type", "linear", context);
  expectBooleanNotTrue(record, "use_karras_sigmas", context);
  expectBooleanNotTrue(record, "use_exponential_sigmas", context);
  expectBooleanNotTrue(record, "use_beta_sigmas", context);
  expectAbsent(record, "sigma_min", context);
  expectAbsent(record, "sigma_max", context);
  expectStringValue(record, "timestep_type", "discrete", context);

  const config: EulerSchedulerConfig = parseBaseScheduleConfig(record, context);
  const timestepSpacing = optionalTimestepSpacing(record, "timestep_spacing", context);
  const stepsOffset = optionalInteger(record, "steps_offset", context);
  const finalSigmasType = optionalEulerFinalSigmasType(record, "final_sigmas_type", context);

  if (timestepSpacing !== undefined) {
    config.timestepSpacing = timestepSpacing;
  }
  if (stepsOffset !== undefined) {
    config.stepsOffset = stepsOffset;
  }
  if (finalSigmasType !== undefined) {
    config.finalSigmasType = finalSigmasType;
  }
  return config;
}

/** Parse a Diffusers scheduler config JSON payload into a supported scheduler spec. */
export function parseDiffusionSchedulerConfig(rawConfig: unknown): ParsedDiffusionSchedulerConfig {
  const context = "scheduler_config.json";
  const record = expectRecord(rawConfig, context);
  const className = parseClassName(record, context);

  if (className === "DDIMScheduler") {
    return {
      kind: "ddim",
      className,
      config: parseDDIMConfig(record, context),
    };
  }

  if (className === "FlowMatchEulerDiscreteScheduler") {
    return {
      kind: "flow-match-euler",
      className,
      config: parseFlowMatchEulerConfig(record, context),
    };
  }

  return {
    kind: "euler",
    className,
    config: parseEulerConfig(record, context),
  };
}

/** Create a supported scheduler instance from a parsed Diffusers config. */
export function createDiffusionScheduler(
  parsedConfig: Extract<ParsedDiffusionSchedulerConfig, { kind: "ddim" }>,
): DDIMScheduler;
export function createDiffusionScheduler(
  parsedConfig: Extract<ParsedDiffusionSchedulerConfig, { kind: "euler" }>,
): EulerScheduler;
export function createDiffusionScheduler(
  parsedConfig: Extract<ParsedDiffusionSchedulerConfig, { kind: "flow-match-euler" }>,
): FlowMatchEulerScheduler;
export function createDiffusionScheduler(
  parsedConfig: ParsedDiffusionSchedulerConfig,
): SupportedDiffusionScheduler;
export function createDiffusionScheduler(
  parsedConfig: ParsedDiffusionSchedulerConfig,
): SupportedDiffusionScheduler {
  if (parsedConfig.kind === "ddim") {
    return new DDIMScheduler(parsedConfig.config);
  }
  if (parsedConfig.kind === "flow-match-euler") {
    return new FlowMatchEulerScheduler(parsedConfig.config);
  }
  return new EulerScheduler(parsedConfig.config);
}

/** Load and parse `scheduler/scheduler_config.json` from a local diffusion snapshot. */
export async function loadDiffusionSchedulerConfig(
  snapshotDirectory: string,
  options: DiffusionSchedulerLoadOptions = {},
): Promise<ParsedDiffusionSchedulerConfig> {
  const loaded = await readDiffusionSchedulerConfig(snapshotDirectory, options);
  return parseDiffusionSchedulerConfig(loaded.rawConfig);
}

async function readDiffusionSchedulerConfig(
  snapshotDirectory: string,
  options: DiffusionSchedulerLoadOptions,
): Promise<{ configPath: string; rawConfig: Record<string, unknown> }> {
  const configPath = schedulerConfigPath(snapshotDirectory, options);
  const file = Bun.file(configPath);
  if (!(await file.exists())) {
    throw new DiffusionConfigError(`loadDiffusionSchedulerConfig: missing ${configPath}.`);
  }

  let rawConfig: unknown;
  try {
    rawConfig = await file.json();
  } catch {
    throw new DiffusionConfigError(
      `loadDiffusionSchedulerConfig: ${configPath} must contain valid JSON.`,
    );
  }
  return { configPath, rawConfig: expectRecord(rawConfig, "scheduler_config.json") };
}

/** Load supported scheduler metadata and an instance from a local diffusion snapshot. */
export async function loadDiffusionSchedulerFromSnapshot(
  snapshotDirectory: string,
  options: DiffusionSchedulerLoadOptions = {},
): Promise<DiffusionSchedulerLoadResult> {
  const loaded = await readDiffusionSchedulerConfig(snapshotDirectory, options);
  const parsedConfig = parseDiffusionSchedulerConfig(loaded.rawConfig);
  return {
    scheduler: createDiffusionScheduler(parsedConfig),
    parsedConfig,
    className: parsedConfig.className,
    configPath: loaded.configPath,
    rawConfig: loaded.rawConfig,
  };
}
