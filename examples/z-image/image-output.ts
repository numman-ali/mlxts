import type { MxArray } from "@mlxts/core";

import {
  type ImageProofArtifactReport,
  imageProofImageToBmpBytes,
  writeImageProofBmp,
} from "../image-proof/artifact";

export type ZImageBmpWriteResult = ImageProofArtifactReport;

/** Convert one Z-Image proof image tensor into uncompressed BMP bytes. */
export function zImageToBmpBytes(image: MxArray): Uint8Array {
  return imageProofImageToBmpBytes(image, "Z-Image");
}

/** Write one generated Z-Image proof image and return verifier evidence. */
export function writeZImageBmp(image: MxArray, outputPath: string): ZImageBmpWriteResult {
  return writeImageProofBmp(image, outputPath, {
    label: "Z-Image",
  });
}
