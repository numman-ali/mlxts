import type { MxArray } from "@mlxts/core";

import {
  type ImageProofArtifactReport,
  imageProofImageToBmpBytes,
  writeImageProofBmp,
} from "../image-proof/artifact";

export type QwenImageBmpWriteResult = ImageProofArtifactReport;

/** Convert one Qwen-Image proof image tensor into uncompressed BMP bytes. */
export function qwenImageToBmpBytes(image: MxArray): Uint8Array {
  return imageProofImageToBmpBytes(image, "Qwen-Image");
}

/** Write one generated Qwen-Image proof image and return verifier evidence. */
export function writeQwenImageBmp(image: MxArray, outputPath: string): QwenImageBmpWriteResult {
  return writeImageProofBmp(image, outputPath, {
    label: "Qwen-Image",
  });
}
