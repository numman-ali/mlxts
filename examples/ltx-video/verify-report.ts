#!/usr/bin/env bun

import { readFileSync } from "fs";

import { type ImageProofArtifactReport, verifyImageProofArtifact } from "../image-proof/artifact";
import type { Ltx2AudioWavChecks, Ltx2AudioWavWriteResult } from "./audio-output";

type LtxVideoProofPipeline = "ltx-video" | "ltx2";
type LtxVideoProofDType = "float16" | "float32" | "bfloat16";

type MediaSize = { width: number; height: number };
type VideoSize = { width: number; height: number; frames: number; channels: number };
type AudioSize = {
  sampleRate: number;
  channels: number;
  samples: number;
  durationSeconds: number;
};

export type LtxVideoProofReport = {
  source: string;
  snapshotPath: string;
  requestedRevision?: string;
  resolvedRevision?: string;
  pipeline: LtxVideoProofPipeline;
  prompt: string;
  negativePrompt: string | null;
  outputPath: string;
  audioOutputPath?: string;
  imageSize: MediaSize;
  videoSize: VideoSize;
  latentSize: VideoSize;
  outputBytes: number;
  artifact: ImageProofArtifactReport;
  audioSize?: AudioSize;
  audioOutputBytes?: number;
  audioArtifact?: Ltx2AudioWavWriteResult;
  steps: number;
  guidanceScale: number;
  audioGuidanceScale?: number;
  maxSequenceLength: number;
  requestedFrames: number;
  decodedFrames: number;
  frameRate: number;
  seed: number;
  dtype: LtxVideoProofDType;
  promptTruncated: boolean;
  negativePromptTruncated: boolean;
  elapsedMs: number;
};

export type VerifyLtxVideoReportOptions = {
  reportPath: string;
  expectPipeline?: LtxVideoProofPipeline;
  expectSource?: string;
  expectResolvedRevision?: string;
  expectPreviewSha256?: string;
  expectAudioSha256?: string;
};

type VerifyCommand = { kind: "help" } | { kind: "run"; options: VerifyLtxVideoReportOptions };

type CheckResult = {
  name: string;
  passed: boolean;
};

type AudioVerificationChecks = Ltx2AudioWavChecks & {
  fileReadable: boolean;
  formatMatches: boolean;
  channelsMatch: boolean;
  samplesMatch: boolean;
  bitsPerSampleMatches: boolean;
  sha256Matches: boolean;
  durationMatches: boolean;
};

export type LtxVideoVerificationResult = {
  status: "passed" | "failed";
  reportPath: string;
  source: string;
  pipeline: LtxVideoProofPipeline;
  outputPath: string;
  imageSize: MediaSize;
  outputBytes: number;
  previewSha256: string;
  audioOutputPath?: string;
  audioOutputBytes?: number;
  audioSha256?: string;
  audioSampleRate?: number;
  passedChecks: number;
  failedChecks: number;
  failedCheckNames: readonly string[];
};

class LtxVideoVerifyUsageError extends Error {}
class LtxVideoVerifyError extends Error {}

function quoteScalar(value: string | number | boolean | null): string {
  return typeof value === "string" ? JSON.stringify(value) : String(value);
}

function sha256Hex(bytes: Uint8Array): string {
  return new Bun.CryptoHasher("sha256").update(bytes).digest("hex");
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
): string | null {
  const value = record[field];
  if (value === null || typeof value === "string") {
    return value;
  }
  failures.push(`${field} must be a string or null`);
  return null;
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

function readPipeline(
  record: Record<string, unknown>,
  field: string,
  failures: string[],
): LtxVideoProofPipeline {
  const value = readRequiredString(record, field, failures);
  if (value === "ltx-video" || value === "ltx2") {
    return value;
  }
  failures.push(`${field} must be ltx-video or ltx2`);
  return "ltx-video";
}

function readDType(
  record: Record<string, unknown>,
  field: string,
  failures: string[],
): LtxVideoProofDType {
  const value = readRequiredString(record, field, failures);
  if (value === "float16" || value === "float32" || value === "bfloat16") {
    return value;
  }
  failures.push(`${field} must be float16, float32, or bfloat16`);
  return "float16";
}

function readImageSize(record: Record<string, unknown>, failures: string[]): MediaSize {
  const imageSize = readRequiredRecord(record, "imageSize", failures);
  if (imageSize === null) {
    return { width: 0, height: 0 };
  }
  return {
    width: readRequiredNumber(imageSize, "width", failures),
    height: readRequiredNumber(imageSize, "height", failures),
  };
}

function readVideoSize(
  record: Record<string, unknown>,
  field: string,
  failures: string[],
): VideoSize {
  const size = readRequiredRecord(record, field, failures);
  if (size === null) {
    return { width: 0, height: 0, frames: 0, channels: 0 };
  }
  return {
    width: readRequiredNumber(size, "width", failures),
    height: readRequiredNumber(size, "height", failures),
    frames: readRequiredNumber(size, "frames", failures),
    channels: readRequiredNumber(size, "channels", failures),
  };
}

function readAudioSize(record: Record<string, unknown>, failures: string[]): AudioSize | undefined {
  const value = record.audioSize;
  if (value === undefined) {
    return undefined;
  }
  if (!isRecord(value)) {
    failures.push("audioSize must be an object when present");
    return undefined;
  }
  return {
    sampleRate: readRequiredNumber(value, "sampleRate", failures),
    channels: readRequiredNumber(value, "channels", failures),
    samples: readRequiredNumber(value, "samples", failures),
    durationSeconds: readRequiredNumber(value, "durationSeconds", failures),
  };
}

function isImageProofArtifactReport(value: unknown): value is ImageProofArtifactReport {
  return (
    isRecord(value) &&
    typeof value.path === "string" &&
    value.format === "bmp" &&
    typeof value.width === "number" &&
    typeof value.height === "number" &&
    value.bitsPerPixel === 24 &&
    typeof value.rowStride === "number" &&
    typeof value.pixelBytes === "number" &&
    typeof value.bytes === "number" &&
    typeof value.sha256 === "string" &&
    isRecord(value.tensor) &&
    isRecord(value.checks) &&
    (value.status === "passed" || value.status === "failed")
  );
}

function isAudioChecks(value: unknown): value is Ltx2AudioWavChecks {
  return (
    isRecord(value) &&
    typeof value.riffHeaderValid === "boolean" &&
    typeof value.byteLengthMatches === "boolean" &&
    typeof value.sampleRateMatches === "boolean" &&
    typeof value.sha256Present === "boolean" &&
    typeof value.finiteTensor === "boolean" &&
    typeof value.waveformHasSamples === "boolean"
  );
}

function isAudioArtifact(value: unknown): value is Ltx2AudioWavWriteResult {
  return (
    isRecord(value) &&
    typeof value.path === "string" &&
    value.format === "wav" &&
    typeof value.sampleRate === "number" &&
    typeof value.channels === "number" &&
    typeof value.samples === "number" &&
    typeof value.durationSeconds === "number" &&
    value.bitsPerSample === 16 &&
    typeof value.bytes === "number" &&
    typeof value.sha256 === "string" &&
    typeof value.peakAbs === "number" &&
    typeof value.meanAbs === "number" &&
    isAudioChecks(value.checks) &&
    (value.status === "passed" || value.status === "failed")
  );
}

function readArtifactReport(
  record: Record<string, unknown>,
  failures: string[],
): ImageProofArtifactReport {
  const artifact = record.artifact;
  if (isImageProofArtifactReport(artifact)) {
    return artifact;
  }
  failures.push("artifact must be a BMP proof artifact report");
  return {
    path: "",
    format: "bmp",
    width: 0,
    height: 0,
    bitsPerPixel: 24,
    rowStride: 0,
    pixelBytes: 0,
    bytes: 0,
    sha256: "",
    tensor: {
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
    },
    checks: {
      bmpHeaderValid: false,
      dimensionsMatch: false,
      byteLengthMatches: false,
      sha256Present: false,
      finiteTensor: false,
      tensorHasDynamicRange: false,
      tensorHasChannelVariance: false,
      bmpHasMultipleByteValues: false,
    },
    status: "failed",
  };
}

function readAudioArtifact(
  record: Record<string, unknown>,
  failures: string[],
): Ltx2AudioWavWriteResult | undefined {
  const artifact = record.audioArtifact;
  if (artifact === undefined) {
    return undefined;
  }
  if (isAudioArtifact(artifact)) {
    return artifact;
  }
  failures.push("audioArtifact must be a WAV proof artifact report when present");
  return undefined;
}

function readLtxVideoProofReport(value: unknown): LtxVideoProofReport {
  const failures: string[] = [];
  if (!isRecord(value)) {
    throw new LtxVideoVerifyError("report JSON must be an object");
  }
  const pipeline = readPipeline(value, "pipeline", failures);
  const report: LtxVideoProofReport = {
    source: readRequiredString(value, "source", failures),
    snapshotPath: readRequiredString(value, "snapshotPath", failures),
    pipeline,
    prompt: readRequiredString(value, "prompt", failures),
    negativePrompt: readNullableString(value, "negativePrompt", failures),
    outputPath: readRequiredString(value, "outputPath", failures),
    imageSize: readImageSize(value, failures),
    videoSize: readVideoSize(value, "videoSize", failures),
    latentSize: readVideoSize(value, "latentSize", failures),
    outputBytes: readRequiredNumber(value, "outputBytes", failures),
    artifact: readArtifactReport(value, failures),
    steps: readRequiredNumber(value, "steps", failures),
    guidanceScale: readRequiredNumber(value, "guidanceScale", failures),
    maxSequenceLength: readRequiredNumber(value, "maxSequenceLength", failures),
    requestedFrames: readRequiredNumber(value, "requestedFrames", failures),
    decodedFrames: readRequiredNumber(value, "decodedFrames", failures),
    frameRate: readRequiredNumber(value, "frameRate", failures),
    seed: readRequiredNumber(value, "seed", failures),
    dtype: readDType(value, "dtype", failures),
    promptTruncated: readRequiredBoolean(value, "promptTruncated", failures),
    negativePromptTruncated: readRequiredBoolean(value, "negativePromptTruncated", failures),
    elapsedMs: readRequiredNumber(value, "elapsedMs", failures),
  };
  const requestedRevision = readOptionalString(value, "requestedRevision", failures);
  const resolvedRevision = readOptionalString(value, "resolvedRevision", failures);
  const audioOutputPath = readOptionalString(value, "audioOutputPath", failures);
  const audioOutputBytes = readOptionalNumber(value, "audioOutputBytes", failures);
  const audioGuidanceScale = readOptionalNumber(value, "audioGuidanceScale", failures);
  const audioSize = readAudioSize(value, failures);
  const audioArtifact = readAudioArtifact(value, failures);
  if (requestedRevision !== undefined) {
    report.requestedRevision = requestedRevision;
  }
  if (resolvedRevision !== undefined) {
    report.resolvedRevision = resolvedRevision;
  }
  if (audioOutputPath !== undefined) {
    report.audioOutputPath = audioOutputPath;
  }
  if (audioOutputBytes !== undefined) {
    report.audioOutputBytes = audioOutputBytes;
  }
  if (audioGuidanceScale !== undefined) {
    report.audioGuidanceScale = audioGuidanceScale;
  }
  if (audioSize !== undefined) {
    report.audioSize = audioSize;
  }
  if (audioArtifact !== undefined) {
    report.audioArtifact = audioArtifact;
  }
  validateFamilyFields(report, failures);
  if (failures.length > 0) {
    throw new LtxVideoVerifyError(failures.join("; "));
  }
  return report;
}

function requireFamilyField(passed: boolean, message: string, failures: string[]): void {
  if (!passed) {
    failures.push(message);
  }
}

function validateFamilyFields(report: LtxVideoProofReport, failures: string[]): void {
  requireFamilyField(report.steps > 0, "steps must be positive", failures);
  requireFamilyField(report.maxSequenceLength > 0, "maxSequenceLength must be positive", failures);
  requireFamilyField(report.requestedFrames > 0, "requestedFrames must be positive", failures);
  requireFamilyField(report.decodedFrames > 0, "decodedFrames must be positive", failures);
  requireFamilyField(report.frameRate > 0, "frameRate must be positive", failures);
  requireFamilyField(report.seed >= 0, "seed must be non-negative", failures);
  if (report.pipeline === "ltx-video") {
    requireFamilyField(
      report.audioArtifact === undefined,
      "audioArtifact is only valid for ltx2 reports",
      failures,
    );
    return;
  }
  requireFamilyField(
    typeof report.audioOutputPath === "string",
    "audioOutputPath is required for ltx2 reports",
    failures,
  );
  requireFamilyField(
    typeof report.audioOutputBytes === "number",
    "audioOutputBytes is required for ltx2 reports",
    failures,
  );
  requireFamilyField(
    typeof report.audioGuidanceScale === "number",
    "audioGuidanceScale is required for ltx2 reports",
    failures,
  );
  requireFamilyField(
    report.audioSize !== undefined,
    "audioSize is required for ltx2 reports",
    failures,
  );
  requireFamilyField(
    report.audioArtifact !== undefined,
    "audioArtifact is required for ltx2 reports",
    failures,
  );
}

function imageCheckResults(artifact: ImageProofArtifactReport): CheckResult[] {
  try {
    const checks = verifyImageProofArtifact(artifact);
    return [
      { name: "artifact.file_readable", passed: true },
      ...Object.entries(checks).map(([name, passed]) => ({
        name: `artifact.${name}`,
        passed,
      })),
    ];
  } catch {
    return [
      { name: "artifact.file_readable", passed: false },
      ...Object.keys(artifact.checks).map((name) => ({ name: `artifact.${name}`, passed: false })),
    ];
  }
}

function verifyAudioArtifact(artifact: Ltx2AudioWavWriteResult): AudioVerificationChecks {
  let bytes: Uint8Array;
  try {
    bytes = readFileSync(artifact.path);
  } catch {
    return {
      fileReadable: false,
      formatMatches: artifact.format === "wav",
      riffHeaderValid: false,
      byteLengthMatches: false,
      sampleRateMatches: false,
      channelsMatch: false,
      samplesMatch: false,
      bitsPerSampleMatches: false,
      sha256Present: artifact.sha256.length === 64,
      sha256Matches: false,
      finiteTensor: Number.isFinite(artifact.peakAbs) && Number.isFinite(artifact.meanAbs),
      waveformHasSamples: artifact.channels > 0 && artifact.samples > 0,
      durationMatches: false,
    };
  }
  if (bytes.byteLength < 44) {
    return {
      fileReadable: true,
      formatMatches: artifact.format === "wav",
      riffHeaderValid: false,
      byteLengthMatches: false,
      sampleRateMatches: false,
      channelsMatch: false,
      samplesMatch: false,
      bitsPerSampleMatches: false,
      sha256Present: artifact.sha256.length === 64,
      sha256Matches: sha256Hex(bytes) === artifact.sha256 && artifact.sha256.length === 64,
      finiteTensor: Number.isFinite(artifact.peakAbs) && Number.isFinite(artifact.meanAbs),
      waveformHasSamples: artifact.channels > 0 && artifact.samples > 0,
      durationMatches: false,
    };
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const riff = String.fromCharCode(
    view.getUint8(0),
    view.getUint8(1),
    view.getUint8(2),
    view.getUint8(3),
  );
  const wave = String.fromCharCode(
    view.getUint8(8),
    view.getUint8(9),
    view.getUint8(10),
    view.getUint8(11),
  );
  const channels = view.getUint16(22, true);
  const sampleRate = view.getUint32(24, true);
  const bitsPerSample = view.getUint16(34, true);
  const payloadBytes = view.getUint32(40, true);
  const expectedPayloadBytes = artifact.channels * artifact.samples * 2;
  const duration = artifact.sampleRate === 0 ? Number.NaN : artifact.samples / artifact.sampleRate;
  return {
    fileReadable: true,
    formatMatches: artifact.format === "wav",
    riffHeaderValid: riff === "RIFF" && wave === "WAVE" && bitsPerSample === artifact.bitsPerSample,
    byteLengthMatches:
      bytes.byteLength === artifact.bytes &&
      bytes.byteLength === 44 + expectedPayloadBytes &&
      payloadBytes === expectedPayloadBytes,
    sampleRateMatches: sampleRate === artifact.sampleRate,
    channelsMatch: channels === artifact.channels,
    samplesMatch: payloadBytes === expectedPayloadBytes,
    bitsPerSampleMatches: bitsPerSample === artifact.bitsPerSample,
    sha256Present: artifact.sha256.length === 64,
    sha256Matches: sha256Hex(bytes) === artifact.sha256 && artifact.sha256.length === 64,
    finiteTensor: Number.isFinite(artifact.peakAbs) && Number.isFinite(artifact.meanAbs),
    waveformHasSamples: artifact.channels > 0 && artifact.samples > 0,
    durationMatches: Math.abs(artifact.durationSeconds - duration) <= 0.000001,
  };
}

function audioCheckResults(report: LtxVideoProofReport): CheckResult[] {
  const artifact = report.audioArtifact;
  const audioSize = report.audioSize;
  if (report.pipeline === "ltx-video") {
    return [];
  }
  if (artifact === undefined || audioSize === undefined) {
    return [{ name: "audio.required_artifact_present", passed: false }];
  }
  const verification = verifyAudioArtifact(artifact);
  return [
    ...Object.entries(verification).map(([name, passed]) => ({
      name: `audio_artifact.${name}`,
      passed,
    })),
    {
      name: "audio.output_path_matches_artifact",
      passed: report.audioOutputPath === artifact.path,
    },
    {
      name: "audio.output_bytes_match_artifact",
      passed: report.audioOutputBytes === artifact.bytes,
    },
    {
      name: "audio.sample_rate_matches_report",
      passed: audioSize.sampleRate === artifact.sampleRate,
    },
    {
      name: "audio.channels_match_report",
      passed: audioSize.channels === artifact.channels,
    },
    {
      name: "audio.samples_match_report",
      passed: audioSize.samples === artifact.samples,
    },
    {
      name: "audio.duration_matches_report",
      passed: Math.abs(audioSize.durationSeconds - artifact.durationSeconds) <= 0.000001,
    },
  ];
}

function commonCheckResults(
  report: LtxVideoProofReport,
  options: VerifyLtxVideoReportOptions,
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
    { name: "report.video_width_positive", passed: report.videoSize.width > 0 },
    { name: "report.video_height_positive", passed: report.videoSize.height > 0 },
    {
      name: "report.decoded_frames_match",
      passed: report.decodedFrames === report.videoSize.frames,
    },
    { name: "report.latent_channels_positive", passed: report.latentSize.channels > 0 },
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
      name: "expect.preview_sha256",
      passed:
        options.expectPreviewSha256 === undefined ||
        report.artifact.sha256 === options.expectPreviewSha256,
    },
    {
      name: "expect.audio_sha256",
      passed:
        options.expectAudioSha256 === undefined ||
        report.audioArtifact?.sha256 === options.expectAudioSha256,
    },
  ];
}

export function verifyLtxVideoProofReport(
  report: LtxVideoProofReport,
  options: VerifyLtxVideoReportOptions,
): LtxVideoVerificationResult {
  const checks = [
    ...imageCheckResults(report.artifact),
    ...audioCheckResults(report),
    ...commonCheckResults(report, options),
  ];
  const failedCheckNames = checks.filter((check) => !check.passed).map((check) => check.name);
  const result: LtxVideoVerificationResult = {
    status: failedCheckNames.length === 0 ? "passed" : "failed",
    reportPath: options.reportPath,
    source: report.source,
    pipeline: report.pipeline,
    outputPath: report.outputPath,
    imageSize: report.imageSize,
    outputBytes: report.outputBytes,
    previewSha256: report.artifact.sha256,
    passedChecks: checks.length - failedCheckNames.length,
    failedChecks: failedCheckNames.length,
    failedCheckNames,
  };
  if (report.audioArtifact !== undefined) {
    result.audioOutputPath = report.audioArtifact.path;
    result.audioOutputBytes = report.audioArtifact.bytes;
    result.audioSha256 = report.audioArtifact.sha256;
    result.audioSampleRate = report.audioArtifact.sampleRate;
  }
  return result;
}

export function loadLtxVideoProofReport(path: string): LtxVideoProofReport {
  const parsed: unknown = JSON.parse(readFileSync(path, "utf8"));
  return readLtxVideoProofReport(parsed);
}

function readStringFlag(flag: string, value: string | undefined): string {
  if (value === undefined || value.trim() === "" || value.startsWith("--")) {
    throw new LtxVideoVerifyUsageError(`Missing value for ${flag}.`);
  }
  return value;
}

function readPipelineFlag(value: string | undefined): LtxVideoProofPipeline {
  const pipeline = readStringFlag("--expect-pipeline", value);
  if (pipeline === "ltx-video" || pipeline === "ltx2") {
    return pipeline;
  }
  throw new LtxVideoVerifyUsageError("--expect-pipeline must be ltx-video or ltx2.");
}

export function parseVerifyCommand(argv: readonly string[]): VerifyCommand {
  if (argv.some((arg) => arg === "--help" || arg === "-h")) {
    return { kind: "help" };
  }
  const reportPath = argv[0];
  if (reportPath === undefined || reportPath.trim() === "" || reportPath.startsWith("--")) {
    throw new LtxVideoVerifyUsageError("Missing report JSON path.");
  }
  const options: VerifyLtxVideoReportOptions = { reportPath };
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
      case "--expect-preview-sha256":
        options.expectPreviewSha256 = readStringFlag(arg, argv[index + 1]);
        index += 1;
        break;
      case "--expect-audio-sha256":
        options.expectAudioSha256 = readStringFlag(arg, argv[index + 1]);
        index += 1;
        break;
      default:
        throw new LtxVideoVerifyUsageError(
          arg === undefined ? "Missing argument." : `Unknown argument: ${arg}`,
        );
    }
  }
  return { kind: "run", options };
}

export function formatUsage(): string {
  return [
    "description: Verify one LTX-Video or LTX-2 proof JSON report and media artifacts",
    "usage[2]:",
    "  bun run examples/ltx-video/verify-report.ts .tmp/ltx-video/report.json",
    "  bun run examples/ltx-video/verify-report.ts .tmp/ltx-video/report.json --expect-pipeline ltx2",
    "arguments[1]{name,description}:",
    '  "report-json","Path to JSON emitted by examples/ltx-video/index.ts --json"',
    "options[6]{flag,description}:",
    '  "--expect-pipeline <name>","Require ltx-video or ltx2"',
    '  "--expect-source <source>","Require the report source field"',
    '  "--expect-resolved-revision <rev>","Require the report resolvedRevision field"',
    '  "--expect-preview-sha256 <hex>","Require the BMP preview artifact SHA-256"',
    '  "--expect-audio-sha256 <hex>","Require the WAV audio artifact SHA-256"',
    '  "--help","Show this help"',
    "exit_codes[3]{code,meaning}:",
    '  0,"report and artifacts passed verification"',
    '  1,"report or artifact failed verification"',
    '  2,"usage error"',
  ].join("\n");
}

export function formatVerificationSuccess(result: LtxVideoVerificationResult): string {
  return [
    "ltx_video_report:",
    `  status: ${quoteScalar(result.status)}`,
    `  report_path: ${quoteScalar(result.reportPath)}`,
    `  source: ${quoteScalar(result.source)}`,
    `  pipeline: ${quoteScalar(result.pipeline)}`,
    `  output_path: ${quoteScalar(result.outputPath)}`,
    `  preview_size: ${quoteScalar(`${result.imageSize.width}x${result.imageSize.height}`)}`,
    `  output_bytes: ${result.outputBytes}`,
    `  preview_sha256: ${quoteScalar(result.previewSha256)}`,
    ...(result.audioOutputPath === undefined
      ? []
      : [
          `  audio_output_path: ${quoteScalar(result.audioOutputPath)}`,
          `  audio_output_bytes: ${result.audioOutputBytes ?? 0}`,
          `  audio_sample_rate: ${result.audioSampleRate ?? 0}`,
          `  audio_sha256: ${quoteScalar(result.audioSha256 ?? "")}`,
        ]),
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
    '  "Run `bun run examples/ltx-video/verify-report.ts --help` for options"',
  ].join("\n");
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export async function runLtxVideoVerifyCommand(
  argv: readonly string[],
  runtime: { stdout?: (text: string) => void } = {},
): Promise<number> {
  const stdout = runtime.stdout ?? console.log;
  let command: VerifyCommand;
  try {
    command = parseVerifyCommand(argv);
  } catch (error) {
    stdout(formatError(errorMessage(error), "usage"));
    return error instanceof LtxVideoVerifyUsageError ? 2 : 1;
  }
  if (command.kind === "help") {
    stdout(formatUsage());
    return 0;
  }
  try {
    const report = loadLtxVideoProofReport(command.options.reportPath);
    const result = verifyLtxVideoProofReport(report, command.options);
    stdout(formatVerificationSuccess(result));
    return result.status === "passed" ? 0 : 1;
  } catch (error) {
    stdout(formatError(errorMessage(error), "validation"));
    return 1;
  }
}

if (import.meta.main) {
  const exitCode = await runLtxVideoVerifyCommand(Bun.argv.slice(2));
  process.exit(exitCode);
}
