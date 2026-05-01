#!/usr/bin/env bun

import { readFileSync } from "fs";

import {
  type ImageProofArtifactReport,
  type ImageProofChannelStats,
  type ImageProofChecks,
  type ImageProofTensorStats,
  verifyImageProofArtifact,
} from "./artifact";

type ImageProofPipeline =
  | "stable-diffusion"
  | "stable-diffusion-xl"
  | "flux"
  | "z-image"
  | "qwen-image";

export type ImageProofExampleReport = {
  source: string;
  snapshotPath: string;
  requestedRevision?: string;
  resolvedRevision?: string;
  pipeline: ImageProofPipeline;
  prompt: string;
  outputPath: string;
  imageSize: { width: number; height: number };
  outputBytes: number;
  steps: number;
  seed: number;
  dtype: "float16" | "float32" | "bfloat16";
  promptTruncated: boolean;
  elapsedMs: number;
  artifact: ImageProofArtifactReport;
  guidanceScale?: number | null;
  trueCfgScale?: number;
  maxSequenceLength?: number;
  prompt2?: string;
  prompt2Truncated?: boolean;
  negativePrompt?: string | null;
  negativePromptTruncated?: boolean;
};

export type VerifyImageProofReportOptions = {
  reportPath: string;
  expectPipeline?: ImageProofPipeline;
  expectSource?: string;
  expectResolvedRevision?: string;
  expectSha256?: string;
};

type VerifyCommand = { kind: "help" } | { kind: "run"; options: VerifyImageProofReportOptions };

type CheckResult = {
  name: string;
  passed: boolean;
};

export type ImageProofVerificationResult = {
  status: "passed" | "failed";
  reportPath: string;
  source: string;
  pipeline: ImageProofPipeline;
  outputPath: string;
  imageSize: { width: number; height: number };
  outputBytes: number;
  artifactSha256: string;
  passedChecks: number;
  failedChecks: number;
  failedCheckNames: readonly string[];
};

class ImageProofVerifyUsageError extends Error {}
class ImageProofVerifyError extends Error {}

function quoteScalar(value: string | number | boolean | null): string {
  return typeof value === "string" ? JSON.stringify(value) : String(value);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function readRequiredRecord(
  record: Record<string, unknown>,
  field: string,
  failures: string[],
): Record<string, unknown> | null {
  const value = record[field];
  if (!isRecord(value)) {
    failures.push(`${field} must be an object`);
    return null;
  }
  return value;
}

function readRequiredString(
  record: Record<string, unknown>,
  field: string,
  failures: string[],
): string {
  const value = record[field];
  if (typeof value !== "string") {
    failures.push(`${field} must be a string`);
    return "";
  }
  return value;
}

function readOptionalString(
  record: Record<string, unknown>,
  field: string,
  failures: string[],
): string | undefined {
  const value = record[field];
  if (value === undefined) {
    return undefined;
  }
  if (typeof value !== "string") {
    failures.push(`${field} must be a string when present`);
    return undefined;
  }
  return value;
}

function readNullableString(
  record: Record<string, unknown>,
  field: string,
  failures: string[],
): string | null | undefined {
  const value = record[field];
  if (value === undefined) {
    return undefined;
  }
  if (value === null || typeof value === "string") {
    return value;
  }
  failures.push(`${field} must be a string or null when present`);
  return undefined;
}

function readRequiredNumber(
  record: Record<string, unknown>,
  field: string,
  failures: string[],
): number {
  const value = record[field];
  if (typeof value !== "number" || !Number.isFinite(value)) {
    failures.push(`${field} must be a finite number`);
    return 0;
  }
  return value;
}

function readOptionalNumber(
  record: Record<string, unknown>,
  field: string,
  failures: string[],
): number | undefined {
  const value = record[field];
  if (value === undefined) {
    return undefined;
  }
  if (typeof value !== "number" || !Number.isFinite(value)) {
    failures.push(`${field} must be a finite number when present`);
    return undefined;
  }
  return value;
}

function readRequiredBoolean(
  record: Record<string, unknown>,
  field: string,
  failures: string[],
): boolean {
  const value = record[field];
  if (typeof value !== "boolean") {
    failures.push(`${field} must be a boolean`);
    return false;
  }
  return value;
}

function readOptionalBoolean(
  record: Record<string, unknown>,
  field: string,
  failures: string[],
): boolean | undefined {
  const value = record[field];
  if (value === undefined) {
    return undefined;
  }
  if (typeof value !== "boolean") {
    failures.push(`${field} must be a boolean when present`);
    return undefined;
  }
  return value;
}

function readPipeline(
  record: Record<string, unknown>,
  field: string,
  failures: string[],
): ImageProofPipeline {
  const value = readRequiredString(record, field, failures);
  if (
    value === "stable-diffusion" ||
    value === "stable-diffusion-xl" ||
    value === "flux" ||
    value === "z-image" ||
    value === "qwen-image"
  ) {
    return value;
  }
  failures.push(`${field} must be a supported image proof pipeline`);
  return "stable-diffusion";
}

function readDType(
  record: Record<string, unknown>,
  field: string,
  failures: string[],
): "float16" | "float32" | "bfloat16" {
  const value = readRequiredString(record, field, failures);
  if (value === "float16" || value === "float32" || value === "bfloat16") {
    return value;
  }
  failures.push(`${field} must be float16, float32, or bfloat16`);
  return "float16";
}

function readChannelStats(
  record: Record<string, unknown>,
  field: string,
  failures: string[],
): ImageProofChannelStats {
  const channel = readRequiredRecord(record, field, failures);
  if (channel === null) {
    return { min: 0, max: 0, mean: 0, stddev: 0 };
  }
  return {
    min: readRequiredNumber(channel, "min", failures),
    max: readRequiredNumber(channel, "max", failures),
    mean: readRequiredNumber(channel, "mean", failures),
    stddev: readRequiredNumber(channel, "stddev", failures),
  };
}

function readTensorStats(
  record: Record<string, unknown>,
  failures: string[],
): ImageProofTensorStats {
  return {
    min: readRequiredNumber(record, "min", failures),
    max: readRequiredNumber(record, "max", failures),
    mean: readRequiredNumber(record, "mean", failures),
    nonFiniteValues: readRequiredNumber(record, "nonFiniteValues", failures),
    clippedLowValues: readRequiredNumber(record, "clippedLowValues", failures),
    clippedHighValues: readRequiredNumber(record, "clippedHighValues", failures),
    uniqueByteValues: readRequiredNumber(record, "uniqueByteValues", failures),
    red: readChannelStats(record, "red", failures),
    green: readChannelStats(record, "green", failures),
    blue: readChannelStats(record, "blue", failures),
    maxChannelStddev: readRequiredNumber(record, "maxChannelStddev", failures),
  };
}

function readArtifactChecks(record: Record<string, unknown>, failures: string[]): ImageProofChecks {
  return {
    bmpHeaderValid: readRequiredBoolean(record, "bmpHeaderValid", failures),
    dimensionsMatch: readRequiredBoolean(record, "dimensionsMatch", failures),
    byteLengthMatches: readRequiredBoolean(record, "byteLengthMatches", failures),
    sha256Present: readRequiredBoolean(record, "sha256Present", failures),
    finiteTensor: readRequiredBoolean(record, "finiteTensor", failures),
    tensorHasDynamicRange: readRequiredBoolean(record, "tensorHasDynamicRange", failures),
    tensorHasChannelVariance: readRequiredBoolean(record, "tensorHasChannelVariance", failures),
    bmpHasMultipleByteValues: readRequiredBoolean(record, "bmpHasMultipleByteValues", failures),
  };
}

function readArtifactReport(
  record: Record<string, unknown>,
  failures: string[],
): ImageProofArtifactReport {
  const tensorRecord = readRequiredRecord(record, "tensor", failures);
  const checksRecord = readRequiredRecord(record, "checks", failures);
  const format = readRequiredString(record, "format", failures);
  const bitsPerPixel = readRequiredNumber(record, "bitsPerPixel", failures);
  const status = readRequiredString(record, "status", failures);
  if (format !== "bmp") {
    failures.push("artifact.format must be bmp");
  }
  if (bitsPerPixel !== 24) {
    failures.push("artifact.bitsPerPixel must be 24");
  }
  if (status !== "passed" && status !== "failed") {
    failures.push("artifact.status must be passed or failed");
  }
  return {
    path: readRequiredString(record, "path", failures),
    format: "bmp",
    width: readRequiredNumber(record, "width", failures),
    height: readRequiredNumber(record, "height", failures),
    bitsPerPixel: 24,
    rowStride: readRequiredNumber(record, "rowStride", failures),
    pixelBytes: readRequiredNumber(record, "pixelBytes", failures),
    bytes: readRequiredNumber(record, "bytes", failures),
    sha256: readRequiredString(record, "sha256", failures),
    tensor:
      tensorRecord === null
        ? {
            min: 0,
            max: 0,
            mean: 0,
            nonFiniteValues: 0,
            clippedLowValues: 0,
            clippedHighValues: 0,
            uniqueByteValues: 0,
            red: { min: 0, max: 0, mean: 0, stddev: 0 },
            green: { min: 0, max: 0, mean: 0, stddev: 0 },
            blue: { min: 0, max: 0, mean: 0, stddev: 0 },
            maxChannelStddev: 0,
          }
        : readTensorStats(tensorRecord, failures),
    checks:
      checksRecord === null
        ? {
            bmpHeaderValid: false,
            dimensionsMatch: false,
            byteLengthMatches: false,
            sha256Present: false,
            finiteTensor: false,
            tensorHasDynamicRange: false,
            tensorHasChannelVariance: false,
            bmpHasMultipleByteValues: false,
          }
        : readArtifactChecks(checksRecord, failures),
    status: status === "failed" ? "failed" : "passed",
  };
}

function readImageSize(
  record: Record<string, unknown>,
  failures: string[],
): { width: number; height: number } {
  const imageSize = readRequiredRecord(record, "imageSize", failures);
  if (imageSize === null) {
    return { width: 0, height: 0 };
  }
  return {
    width: readRequiredNumber(imageSize, "width", failures),
    height: readRequiredNumber(imageSize, "height", failures),
  };
}

function readImageProofReport(value: unknown): ImageProofExampleReport {
  const failures: string[] = [];
  if (!isRecord(value)) {
    throw new ImageProofVerifyError("report JSON must be an object");
  }
  const artifactRecord = readRequiredRecord(value, "artifact", failures);
  const pipeline = readPipeline(value, "pipeline", failures);
  const report: ImageProofExampleReport = {
    source: readRequiredString(value, "source", failures),
    snapshotPath: readRequiredString(value, "snapshotPath", failures),
    pipeline,
    prompt: readRequiredString(value, "prompt", failures),
    outputPath: readRequiredString(value, "outputPath", failures),
    imageSize: readImageSize(value, failures),
    outputBytes: readRequiredNumber(value, "outputBytes", failures),
    steps: readRequiredNumber(value, "steps", failures),
    seed: readRequiredNumber(value, "seed", failures),
    dtype: readDType(value, "dtype", failures),
    promptTruncated: readRequiredBoolean(value, "promptTruncated", failures),
    elapsedMs: readRequiredNumber(value, "elapsedMs", failures),
    artifact:
      artifactRecord === null
        ? readArtifactReport({}, failures)
        : readArtifactReport(artifactRecord, failures),
  };
  const requestedRevision = readOptionalString(value, "requestedRevision", failures);
  const resolvedRevision = readOptionalString(value, "resolvedRevision", failures);
  const guidanceScale = readOptionalNumber(value, "guidanceScale", failures);
  const trueCfgScale = readOptionalNumber(value, "trueCfgScale", failures);
  const maxSequenceLength = readOptionalNumber(value, "maxSequenceLength", failures);
  const prompt2 = readOptionalString(value, "prompt2", failures);
  const prompt2Truncated = readOptionalBoolean(value, "prompt2Truncated", failures);
  const negativePrompt = readNullableString(value, "negativePrompt", failures);
  const negativePromptTruncated = readOptionalBoolean(value, "negativePromptTruncated", failures);
  if (requestedRevision !== undefined) {
    report.requestedRevision = requestedRevision;
  }
  if (resolvedRevision !== undefined) {
    report.resolvedRevision = resolvedRevision;
  }
  if (guidanceScale !== undefined) {
    report.guidanceScale = guidanceScale;
  } else if (value.guidanceScale === null) {
    report.guidanceScale = null;
  }
  if (trueCfgScale !== undefined) {
    report.trueCfgScale = trueCfgScale;
  }
  if (maxSequenceLength !== undefined) {
    report.maxSequenceLength = maxSequenceLength;
  }
  if (prompt2 !== undefined) {
    report.prompt2 = prompt2;
  }
  if (prompt2Truncated !== undefined) {
    report.prompt2Truncated = prompt2Truncated;
  }
  if (negativePrompt !== undefined) {
    report.negativePrompt = negativePrompt;
  }
  if (negativePromptTruncated !== undefined) {
    report.negativePromptTruncated = negativePromptTruncated;
  }
  validateFamilyFields(report, failures);
  if (failures.length > 0) {
    throw new ImageProofVerifyError(failures.join("; "));
  }
  return report;
}

function requireFamilyField(passed: boolean, message: string, failures: string[]): void {
  if (!passed) {
    failures.push(message);
  }
}

function validateStableDiffusionFields(report: ImageProofExampleReport, failures: string[]): void {
  requireFamilyField(
    typeof report.guidanceScale === "number",
    "guidanceScale is required for Stable Diffusion reports",
    failures,
  );
  requireFamilyField(
    typeof report.negativePrompt === "string",
    "negativePrompt is required for Stable Diffusion reports",
    failures,
  );
  requireFamilyField(
    typeof report.negativePromptTruncated === "boolean",
    "negativePromptTruncated is required for Stable Diffusion reports",
    failures,
  );
}

function validateFluxFields(report: ImageProofExampleReport, failures: string[]): void {
  requireFamilyField(
    typeof report.guidanceScale === "number" || report.guidanceScale === null,
    "guidanceScale must be a number or null for FLUX reports",
    failures,
  );
  requireFamilyField(
    typeof report.maxSequenceLength === "number",
    "maxSequenceLength is required for FLUX reports",
    failures,
  );
  requireFamilyField(
    typeof report.prompt2 === "string",
    "prompt2 is required for FLUX reports",
    failures,
  );
  requireFamilyField(
    typeof report.prompt2Truncated === "boolean",
    "prompt2Truncated is required for FLUX reports",
    failures,
  );
}

function validateZImageFields(report: ImageProofExampleReport, failures: string[]): void {
  requireFamilyField(
    report.guidanceScale === 0,
    "guidanceScale must be 0 for current Z-Image proof reports",
    failures,
  );
  requireFamilyField(
    typeof report.maxSequenceLength === "number",
    "maxSequenceLength is required for Z-Image reports",
    failures,
  );
}

function validateQwenImageFields(report: ImageProofExampleReport, failures: string[]): void {
  requireFamilyField(
    typeof report.trueCfgScale === "number",
    "trueCfgScale is required for Qwen-Image reports",
    failures,
  );
  requireFamilyField(
    report.negativePrompt === null || typeof report.negativePrompt === "string",
    "negativePrompt is required for Qwen-Image reports",
    failures,
  );
  requireFamilyField(
    typeof report.negativePromptTruncated === "boolean",
    "negativePromptTruncated is required for Qwen-Image reports",
    failures,
  );
  requireFamilyField(
    typeof report.maxSequenceLength === "number",
    "maxSequenceLength is required for Qwen-Image reports",
    failures,
  );
}

function validateFamilyFields(report: ImageProofExampleReport, failures: string[]): void {
  if (report.pipeline === "stable-diffusion" || report.pipeline === "stable-diffusion-xl") {
    validateStableDiffusionFields(report, failures);
    return;
  }
  if (report.pipeline === "flux") {
    validateFluxFields(report, failures);
    return;
  }
  if (report.pipeline === "z-image") {
    validateZImageFields(report, failures);
    return;
  }
  validateQwenImageFields(report, failures);
}

function artifactCheckResults(checks: ImageProofChecks): CheckResult[] {
  return Object.entries(checks).map(([name, passed]) => ({
    name: `artifact.${name}`,
    passed,
  }));
}

function commonCheckResults(
  report: ImageProofExampleReport,
  options: VerifyImageProofReportOptions,
): CheckResult[] {
  return [
    {
      name: "report.output_path_matches_artifact",
      passed: report.outputPath === report.artifact.path,
    },
    {
      name: "report.output_bytes_match_artifact",
      passed: report.outputBytes === report.artifact.bytes,
    },
    {
      name: "report.width_matches_artifact",
      passed: report.imageSize.width === report.artifact.width,
    },
    {
      name: "report.height_matches_artifact",
      passed: report.imageSize.height === report.artifact.height,
    },
    {
      name: "report.steps_positive",
      passed: Number.isInteger(report.steps) && report.steps > 0,
    },
    {
      name: "report.seed_non_negative",
      passed: Number.isInteger(report.seed) && report.seed >= 0,
    },
    {
      name: "expect.pipeline",
      passed: options.expectPipeline === undefined || report.pipeline === options.expectPipeline,
    },
    {
      name: "expect.source",
      passed: options.expectSource === undefined || report.source === options.expectSource,
    },
    {
      name: "expect.resolved_revision",
      passed:
        options.expectResolvedRevision === undefined ||
        report.resolvedRevision === options.expectResolvedRevision,
    },
    {
      name: "expect.sha256",
      passed: options.expectSha256 === undefined || report.artifact.sha256 === options.expectSha256,
    },
  ];
}

export function verifyImageProofReport(
  report: ImageProofExampleReport,
  options: VerifyImageProofReportOptions,
): ImageProofVerificationResult {
  const artifactChecks = verifyImageProofArtifact(report.artifact);
  const checks = [...artifactCheckResults(artifactChecks), ...commonCheckResults(report, options)];
  const failedCheckNames = checks.filter((check) => !check.passed).map((check) => check.name);
  return {
    status: failedCheckNames.length === 0 ? "passed" : "failed",
    reportPath: options.reportPath,
    source: report.source,
    pipeline: report.pipeline,
    outputPath: report.outputPath,
    imageSize: report.imageSize,
    outputBytes: report.outputBytes,
    artifactSha256: report.artifact.sha256,
    passedChecks: checks.length - failedCheckNames.length,
    failedChecks: failedCheckNames.length,
    failedCheckNames,
  };
}

export function loadImageProofReport(path: string): ImageProofExampleReport {
  const parsed: unknown = JSON.parse(readFileSync(path, "utf8"));
  return readImageProofReport(parsed);
}

function readStringFlag(flag: string, value: string | undefined): string {
  if (value === undefined || value.trim() === "" || value.startsWith("--")) {
    throw new ImageProofVerifyUsageError(`Missing value for ${flag}.`);
  }
  return value;
}

function readPipelineFlag(value: string | undefined): ImageProofPipeline {
  const pipeline = readStringFlag("--expect-pipeline", value);
  if (
    pipeline === "stable-diffusion" ||
    pipeline === "stable-diffusion-xl" ||
    pipeline === "flux" ||
    pipeline === "z-image" ||
    pipeline === "qwen-image"
  ) {
    return pipeline;
  }
  throw new ImageProofVerifyUsageError("--expect-pipeline must name a supported image pipeline.");
}

export function parseVerifyCommand(argv: readonly string[]): VerifyCommand {
  if (argv.some((arg) => arg === "--help" || arg === "-h")) {
    return { kind: "help" };
  }
  const reportPath = argv[0];
  if (reportPath === undefined || reportPath.trim() === "" || reportPath.startsWith("--")) {
    throw new ImageProofVerifyUsageError("Missing report JSON path.");
  }
  const options: VerifyImageProofReportOptions = { reportPath };
  for (let index = 1; index < argv.length; index += 1) {
    const arg = argv[index];
    switch (arg) {
      case "--expect-pipeline":
        options.expectPipeline = readPipelineFlag(argv[index + 1]);
        index += 1;
        break;
      case "--expect-source":
        options.expectSource = readStringFlag(arg, argv[index + 1]);
        index += 1;
        break;
      case "--expect-resolved-revision":
        options.expectResolvedRevision = readStringFlag(arg, argv[index + 1]);
        index += 1;
        break;
      case "--expect-sha256":
        options.expectSha256 = readStringFlag(arg, argv[index + 1]);
        index += 1;
        break;
      default:
        throw new ImageProofVerifyUsageError(
          arg === undefined ? "Missing argument." : `Unknown argument: ${arg}`,
        );
    }
  }
  return { kind: "run", options };
}

export function formatUsage(): string {
  return [
    "description: Verify one Phase 10 image proof JSON report and its BMP artifact",
    "usage[2]:",
    "  bun run examples/image-proof/verify-report.ts .tmp/image-proof/report.json",
    "  bun run examples/image-proof/verify-report.ts .tmp/image-proof/report.json --expect-pipeline qwen-image",
    "arguments[1]{name,description}:",
    '  "report-json","Path to JSON emitted by a Phase 10 image proof command"',
    "options[5]{flag,description}:",
    '  "--expect-pipeline <name>","Require stable-diffusion, stable-diffusion-xl, flux, z-image, or qwen-image"',
    '  "--expect-source <source>","Require the report source field"',
    '  "--expect-resolved-revision <rev>","Require the report resolvedRevision field"',
    '  "--expect-sha256 <hex>","Require the BMP artifact SHA-256"',
    '  "--help","Show this help"',
    "exit_codes[3]{code,meaning}:",
    '  0,"report and artifact passed verification"',
    '  1,"report or artifact failed verification"',
    '  2,"usage error"',
  ].join("\n");
}

export function formatVerificationSuccess(result: ImageProofVerificationResult): string {
  return [
    "image_proof_report:",
    `  status: ${quoteScalar(result.status)}`,
    `  report_path: ${quoteScalar(result.reportPath)}`,
    `  source: ${quoteScalar(result.source)}`,
    `  pipeline: ${quoteScalar(result.pipeline)}`,
    `  output_path: ${quoteScalar(result.outputPath)}`,
    `  image_size: ${quoteScalar(`${result.imageSize.width}x${result.imageSize.height}`)}`,
    `  output_bytes: ${result.outputBytes}`,
    `  artifact_sha256: ${quoteScalar(result.artifactSha256)}`,
    `  passed_checks: ${result.passedChecks}`,
    `  failed_checks: ${result.failedChecks}`,
    ...(result.failedCheckNames.length === 0
      ? []
      : [
          `failed_check_names[${result.failedCheckNames.length}]:`,
          ...result.failedCheckNames.map((name) => `  ${quoteScalar(name)}`),
        ]),
  ].join("\n");
}

function formatError(message: string, code: "usage" | "validation"): string {
  return [
    "error:",
    `  code: ${quoteScalar(code)}`,
    `  message: ${quoteScalar(message)}`,
    "help[1]:",
    '  "Run `bun run examples/image-proof/verify-report.ts --help` for options"',
  ].join("\n");
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export async function runImageProofVerifyCommand(
  argv: readonly string[],
  runtime: { stdout?: (text: string) => void } = {},
): Promise<number> {
  const stdout = runtime.stdout ?? console.log;
  let command: VerifyCommand;
  try {
    command = parseVerifyCommand(argv);
  } catch (error) {
    stdout(formatError(errorMessage(error), "usage"));
    return error instanceof ImageProofVerifyUsageError ? 2 : 1;
  }
  if (command.kind === "help") {
    stdout(formatUsage());
    return 0;
  }
  try {
    const report = loadImageProofReport(command.options.reportPath);
    const result = verifyImageProofReport(report, command.options);
    stdout(formatVerificationSuccess(result));
    return result.status === "passed" ? 0 : 1;
  } catch (error) {
    stdout(formatError(errorMessage(error), "validation"));
    return 1;
  }
}

if (import.meta.main) {
  const exitCode = await runImageProofVerifyCommand(Bun.argv.slice(2));
  process.exit(exitCode);
}
