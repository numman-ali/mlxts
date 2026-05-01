import type { ImageProofArtifactReport } from "./artifact";

export function exampleImageProofArtifactReport(
  overrides: Partial<ImageProofArtifactReport> = {},
): ImageProofArtifactReport {
  const report: ImageProofArtifactReport = {
    path: ".tmp/image-proof/sample.bmp",
    format: "bmp",
    width: 64,
    height: 64,
    bitsPerPixel: 24,
    rowStride: 192,
    pixelBytes: 12_288,
    bytes: 12_342,
    sha256: "a".repeat(64),
    tensor: {
      min: 0,
      max: 1,
      mean: 0.5,
      nonFiniteValues: 0,
      clippedLowValues: 1,
      clippedHighValues: 1,
      uniqueByteValues: 4,
      red: { min: 0, max: 1, mean: 0.5, stddev: 0.5 },
      green: { min: 0, max: 1, mean: 0.5, stddev: 0.5 },
      blue: { min: 0, max: 1, mean: 0.5, stddev: 0.5 },
      maxChannelStddev: 0.5,
    },
    checks: {
      bmpHeaderValid: true,
      dimensionsMatch: true,
      byteLengthMatches: true,
      sha256Present: true,
      finiteTensor: true,
      tensorHasDynamicRange: true,
      tensorHasChannelVariance: true,
      bmpHasMultipleByteValues: true,
    },
    status: "passed",
  };
  return { ...report, ...overrides };
}
