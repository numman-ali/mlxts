import { concatenate, type MxArray, reshape, slice } from "@mlxts/core";

import {
  type ImageProofArtifactReport,
  imageProofImageToBmpBytes,
  writeImageProofBmp,
} from "../image-proof/artifact";

export type LtxVideoPreviewBmpWriteResult = ImageProofArtifactReport;

function sampledPreviewFrameIndices(frames: number): number[] {
  const count = Math.min(frames, 4);
  const indices = new Set<number>();
  for (let index = 0; index < count; index += 1) {
    indices.add(Math.round((index * (frames - 1)) / Math.max(1, count - 1)));
  }
  return [...indices].toSorted((left, right) => left - right);
}

function videoPreviewSheet(video: MxArray): MxArray {
  const [batch, frames, height, width, channels] = video.shape;
  if (
    video.shape.length !== 5 ||
    batch !== 1 ||
    frames === undefined ||
    height === undefined ||
    width === undefined ||
    channels === undefined ||
    frames < 1 ||
    channels < 3
  ) {
    throw new Error(
      `LTX-Video proof output expects one BFHWC RGB video, got [${video.shape.join(",")}].`,
    );
  }
  const previewFrames: MxArray[] = [];
  try {
    for (const frameIndex of sampledPreviewFrameIndices(frames)) {
      using frame = slice(
        video,
        [0, frameIndex, 0, 0, 0],
        [1, frameIndex + 1, height, width, channels],
      );
      previewFrames.push(reshape(frame, [1, height, width, channels]));
    }
    if (previewFrames.length === 1) {
      const onlyFrame = previewFrames[0];
      if (onlyFrame === undefined) {
        throw new Error("LTX-Video preview sheet requires at least one frame.");
      }
      previewFrames.length = 0;
      return onlyFrame;
    }
    return concatenate(previewFrames, 2);
  } finally {
    for (const frame of previewFrames) {
      frame.free();
    }
  }
}

/** Convert sampled LTX-Video proof frames into uncompressed BMP preview bytes. */
export function ltxVideoPreviewFrameToBmpBytes(video: MxArray): Uint8Array {
  using sheet = videoPreviewSheet(video);
  return imageProofImageToBmpBytes(sheet, "LTX-Video preview sheet");
}

/** Write sampled LTX-Video proof frames and return verifier evidence. */
export function writeLtxVideoPreviewBmp(
  video: MxArray,
  outputPath: string,
): LtxVideoPreviewBmpWriteResult {
  using sheet = videoPreviewSheet(video);
  return writeImageProofBmp(sheet, outputPath, {
    label: "LTX-Video preview sheet",
  });
}
