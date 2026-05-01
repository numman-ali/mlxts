import type { MxArray } from "@mlxts/core";
import { mkdirSync, readFileSync, writeFileSync } from "fs";
import { dirname } from "path";

export type ImageProofChannelStats = {
  min: number;
  max: number;
  mean: number;
  stddev: number;
};

export type ImageProofTensorStats = {
  min: number;
  max: number;
  mean: number;
  nonFiniteValues: number;
  clippedLowValues: number;
  clippedHighValues: number;
  uniqueByteValues: number;
  red: ImageProofChannelStats;
  green: ImageProofChannelStats;
  blue: ImageProofChannelStats;
  maxChannelStddev: number;
};

export type ImageProofChecks = {
  bmpHeaderValid: boolean;
  dimensionsMatch: boolean;
  byteLengthMatches: boolean;
  sha256Present: boolean;
  finiteTensor: boolean;
  tensorHasDynamicRange: boolean;
  tensorHasChannelVariance: boolean;
  bmpHasMultipleByteValues: boolean;
};

export type ImageProofArtifactReport = {
  path: string;
  format: "bmp";
  width: number;
  height: number;
  bitsPerPixel: 24;
  rowStride: number;
  pixelBytes: number;
  bytes: number;
  sha256: string;
  tensor: ImageProofTensorStats;
  checks: ImageProofChecks;
  status: "passed" | "failed";
};

export type ImageProofWriteOptions = {
  label: string;
};

type ImageShape = {
  height: number;
  width: number;
  channels: number;
};

type ImageProofBytes = {
  bytes: Uint8Array;
  width: number;
  height: number;
  rowStride: number;
  pixelBytes: number;
  tensor: ImageProofTensorStats;
};

type RunningChannelStats = {
  min: number;
  max: number;
  total: number;
  totalSquares: number;
};

type TensorStatsAccumulator = {
  min: number;
  max: number;
  total: number;
  finiteValues: number;
  nonFiniteValues: number;
  clippedLowValues: number;
  clippedHighValues: number;
  channels: [RunningChannelStats, RunningChannelStats, RunningChannelStats];
};

const BMP_HEADER_BYTES = 54;

function roundMetric(value: number): number {
  if (!Number.isFinite(value)) {
    return value;
  }
  return Math.round(value * 1_000_000) / 1_000_000;
}

function clampByte(value: number): number {
  if (!Number.isFinite(value) || value <= 0) {
    return 0;
  }
  if (value >= 1) {
    return 255;
  }
  return Math.round(value * 255);
}

function assertSingleRgbImage(image: MxArray, label: string): ImageShape {
  const [batch, height, width, channels] = image.shape;
  if (
    image.shape.length !== 4 ||
    batch !== 1 ||
    height === undefined ||
    width === undefined ||
    channels === undefined ||
    channels < 3
  ) {
    throw new Error(
      `${label} image output expects one NHWC RGB image, got [${image.shape.join(",")}].`,
    );
  }
  return { height, width, channels };
}

function writeAscii(view: DataView, offset: number, value: string): void {
  for (let index = 0; index < value.length; index += 1) {
    view.setUint8(offset + index, value.charCodeAt(index));
  }
}

function emptyChannelStats(): RunningChannelStats {
  return {
    min: Number.POSITIVE_INFINITY,
    max: Number.NEGATIVE_INFINITY,
    total: 0,
    totalSquares: 0,
  };
}

function finalizeChannelStats(stats: RunningChannelStats, count: number): ImageProofChannelStats {
  const mean = count === 0 ? 0 : stats.total / count;
  const variance = count === 0 ? 0 : Math.max(0, stats.totalSquares / count - mean * mean);
  return {
    min: roundMetric(stats.min),
    max: roundMetric(stats.max),
    mean: roundMetric(mean),
    stddev: roundMetric(Math.sqrt(variance)),
  };
}

function addChannelValue(stats: RunningChannelStats, value: number): void {
  if (!Number.isFinite(value)) {
    return;
  }
  stats.min = Math.min(stats.min, value);
  stats.max = Math.max(stats.max, value);
  stats.total += value;
  stats.totalSquares += value * value;
}

function emptyTensorStatsAccumulator(): TensorStatsAccumulator {
  return {
    min: Number.POSITIVE_INFINITY,
    max: Number.NEGATIVE_INFINITY,
    total: 0,
    finiteValues: 0,
    nonFiniteValues: 0,
    clippedLowValues: 0,
    clippedHighValues: 0,
    channels: [emptyChannelStats(), emptyChannelStats(), emptyChannelStats()],
  };
}

function addTensorValue(accumulator: TensorStatsAccumulator, channel: number, value: number): void {
  if (!Number.isFinite(value)) {
    accumulator.nonFiniteValues += 1;
    return;
  }
  accumulator.min = Math.min(accumulator.min, value);
  accumulator.max = Math.max(accumulator.max, value);
  accumulator.total += value;
  accumulator.finiteValues += 1;
  accumulator.clippedLowValues += value <= 0 ? 1 : 0;
  accumulator.clippedHighValues += value >= 1 ? 1 : 0;
  const channelStats = accumulator.channels[channel];
  if (channelStats !== undefined) {
    addChannelValue(channelStats, value);
  }
}

function tensorStats(
  pixels: ArrayLike<number>,
  shape: ImageShape,
  byteValues: Set<number>,
): ImageProofTensorStats {
  const accumulator = emptyTensorStatsAccumulator();
  const pixelCount = shape.height * shape.width;

  for (let pixelIndex = 0; pixelIndex < pixelCount; pixelIndex += 1) {
    const sourceOffset = pixelIndex * shape.channels;
    for (let channel = 0; channel < 3; channel += 1) {
      const value = Number(pixels[sourceOffset + channel] ?? Number.NaN);
      addTensorValue(accumulator, channel, value);
    }
  }

  const redStats = finalizeChannelStats(accumulator.channels[0], pixelCount);
  const greenStats = finalizeChannelStats(accumulator.channels[1], pixelCount);
  const blueStats = finalizeChannelStats(accumulator.channels[2], pixelCount);
  return {
    min: roundMetric(accumulator.finiteValues === 0 ? Number.NaN : accumulator.min),
    max: roundMetric(accumulator.finiteValues === 0 ? Number.NaN : accumulator.max),
    mean: roundMetric(
      accumulator.finiteValues === 0 ? Number.NaN : accumulator.total / accumulator.finiteValues,
    ),
    nonFiniteValues: accumulator.nonFiniteValues,
    clippedLowValues: accumulator.clippedLowValues,
    clippedHighValues: accumulator.clippedHighValues,
    uniqueByteValues: byteValues.size,
    red: redStats,
    green: greenStats,
    blue: blueStats,
    maxChannelStddev: Math.max(redStats.stddev, greenStats.stddev, blueStats.stddev),
  };
}

function sha256Hex(bytes: Uint8Array): string {
  return new Bun.CryptoHasher("sha256").update(bytes).digest("hex");
}

function expectedBmpByteLength(width: number, height: number): number {
  return BMP_HEADER_BYTES + Math.ceil((width * 3) / 4) * 4 * height;
}

function imageToBmpProofBytes(image: MxArray, label: string): ImageProofBytes {
  const shape = assertSingleRgbImage(image, label);
  image.eval();
  const pixels = image.toTypedArray();
  const rowStride = Math.ceil((shape.width * 3) / 4) * 4;
  const pixelBytes = rowStride * shape.height;
  const bytes = new Uint8Array(BMP_HEADER_BYTES + pixelBytes);
  const view = new DataView(bytes.buffer);
  const byteValues = new Set<number>();

  writeAscii(view, 0, "BM");
  view.setUint32(2, bytes.byteLength, true);
  view.setUint32(10, BMP_HEADER_BYTES, true);
  view.setUint32(14, 40, true);
  view.setInt32(18, shape.width, true);
  view.setInt32(22, shape.height, true);
  view.setUint16(26, 1, true);
  view.setUint16(28, 24, true);
  view.setUint32(30, 0, true);
  view.setUint32(34, pixelBytes, true);
  view.setInt32(38, 2835, true);
  view.setInt32(42, 2835, true);

  for (let y = 0; y < shape.height; y += 1) {
    const sourceY = shape.height - 1 - y;
    const rowOffset = BMP_HEADER_BYTES + y * rowStride;
    for (let x = 0; x < shape.width; x += 1) {
      const sourceOffset = (sourceY * shape.width + x) * shape.channels;
      const targetOffset = rowOffset + x * 3;
      const blue = clampByte(Number(pixels[sourceOffset + 2] ?? Number.NaN));
      const green = clampByte(Number(pixels[sourceOffset + 1] ?? Number.NaN));
      const red = clampByte(Number(pixels[sourceOffset] ?? Number.NaN));
      bytes[targetOffset] = blue;
      bytes[targetOffset + 1] = green;
      bytes[targetOffset + 2] = red;
      byteValues.add(red);
      byteValues.add(green);
      byteValues.add(blue);
    }
  }

  return {
    bytes,
    width: shape.width,
    height: shape.height,
    rowStride,
    pixelBytes,
    tensor: tensorStats(pixels, shape, byteValues),
  };
}

function checksFor(report: Omit<ImageProofArtifactReport, "checks" | "status">): ImageProofChecks {
  const bytes = readFileSync(report.path);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const magic = String.fromCharCode(view.getUint8(0), view.getUint8(1));
  return {
    bmpHeaderValid: magic === "BM" && view.getUint16(28, true) === report.bitsPerPixel,
    dimensionsMatch:
      view.getInt32(18, true) === report.width && view.getInt32(22, true) === report.height,
    byteLengthMatches: report.bytes === expectedBmpByteLength(report.width, report.height),
    sha256Present: report.sha256.length === 64,
    finiteTensor: report.tensor.nonFiniteValues === 0,
    tensorHasDynamicRange: report.tensor.max > report.tensor.min,
    tensorHasChannelVariance: report.tensor.maxChannelStddev > 0,
    bmpHasMultipleByteValues: report.tensor.uniqueByteValues > 1,
  };
}

function statusFor(checks: ImageProofChecks): "passed" | "failed" {
  return Object.values(checks).every(Boolean) ? "passed" : "failed";
}

function failedCheckNames(checks: ImageProofChecks): string[] {
  return Object.entries(checks)
    .filter(([, passed]) => !passed)
    .map(([name]) => name);
}

function bmpPayloadStats(
  bytes: Uint8Array,
  width: number,
  height: number,
  rowStride: number,
): { uniqueByteValues: number; maxChannelStddev: number; min: number; max: number } {
  const red = emptyChannelStats();
  const green = emptyChannelStats();
  const blue = emptyChannelStats();
  const byteValues = new Set<number>();
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;

  for (let y = 0; y < height; y += 1) {
    const rowOffset = BMP_HEADER_BYTES + y * rowStride;
    for (let x = 0; x < width; x += 1) {
      const offset = rowOffset + x * 3;
      const b = bytes[offset] ?? 0;
      const g = bytes[offset + 1] ?? 0;
      const r = bytes[offset + 2] ?? 0;
      byteValues.add(r);
      byteValues.add(g);
      byteValues.add(b);
      min = Math.min(min, r, g, b);
      max = Math.max(max, r, g, b);
      addChannelValue(red, r / 255);
      addChannelValue(green, g / 255);
      addChannelValue(blue, b / 255);
    }
  }

  return {
    uniqueByteValues: byteValues.size,
    maxChannelStddev: Math.max(
      finalizeChannelStats(red, width * height).stddev,
      finalizeChannelStats(green, width * height).stddev,
      finalizeChannelStats(blue, width * height).stddev,
    ),
    min,
    max,
  };
}

export function imageProofImageToBmpBytes(image: MxArray, label: string): Uint8Array {
  return imageToBmpProofBytes(image, label).bytes;
}

export function writeImageProofBmp(
  image: MxArray,
  outputPath: string,
  options: ImageProofWriteOptions,
): ImageProofArtifactReport {
  if (!outputPath.toLowerCase().endsWith(".bmp")) {
    throw new Error(`${options.label} image output path must end with .bmp.`);
  }
  const proof = imageToBmpProofBytes(image, options.label);
  mkdirSync(dirname(outputPath), { recursive: true });
  writeFileSync(outputPath, proof.bytes);

  const baseReport: Omit<ImageProofArtifactReport, "checks" | "status"> = {
    path: outputPath,
    format: "bmp",
    width: proof.width,
    height: proof.height,
    bitsPerPixel: 24,
    rowStride: proof.rowStride,
    pixelBytes: proof.pixelBytes,
    bytes: proof.bytes.byteLength,
    sha256: sha256Hex(proof.bytes),
    tensor: proof.tensor,
  };
  const checks = checksFor(baseReport);
  const status = statusFor(checks);
  if (status !== "passed") {
    throw new Error(
      `${options.label} proof artifact failed checks: ${failedCheckNames(checks).join(", ")}`,
    );
  }
  return { ...baseReport, checks, status };
}

export function verifyImageProofArtifact(report: ImageProofArtifactReport): ImageProofChecks {
  const bytes = readFileSync(report.path);
  if (bytes.byteLength < BMP_HEADER_BYTES) {
    return {
      bmpHeaderValid: false,
      dimensionsMatch: false,
      byteLengthMatches: false,
      sha256Present: sha256Hex(bytes) === report.sha256 && report.sha256.length === 64,
      finiteTensor: report.checks.finiteTensor && report.tensor.nonFiniteValues === 0,
      tensorHasDynamicRange: false,
      tensorHasChannelVariance: false,
      bmpHasMultipleByteValues: false,
    };
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const magic = String.fromCharCode(view.getUint8(0), view.getUint8(1));
  const expectedBytes = expectedBmpByteLength(report.width, report.height);
  const payloadStats =
    bytes.byteLength >= expectedBytes
      ? bmpPayloadStats(bytes, report.width, report.height, report.rowStride)
      : { uniqueByteValues: 0, maxChannelStddev: 0, min: 0, max: 0 };
  return {
    bmpHeaderValid: magic === "BM" && view.getUint16(28, true) === report.bitsPerPixel,
    dimensionsMatch:
      view.getInt32(18, true) === report.width && view.getInt32(22, true) === report.height,
    byteLengthMatches: bytes.byteLength === report.bytes && bytes.byteLength === expectedBytes,
    sha256Present: sha256Hex(bytes) === report.sha256 && report.sha256.length === 64,
    finiteTensor: report.checks.finiteTensor && report.tensor.nonFiniteValues === 0,
    tensorHasDynamicRange:
      report.checks.tensorHasDynamicRange &&
      report.tensor.max > report.tensor.min &&
      payloadStats.max > payloadStats.min,
    tensorHasChannelVariance:
      report.checks.tensorHasChannelVariance &&
      report.tensor.maxChannelStddev > 0 &&
      payloadStats.maxChannelStddev > 0,
    bmpHasMultipleByteValues:
      report.checks.bmpHasMultipleByteValues &&
      report.tensor.uniqueByteValues > 1 &&
      payloadStats.uniqueByteValues > 1 &&
      payloadStats.maxChannelStddev > 0,
  };
}
