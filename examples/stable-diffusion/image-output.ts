import type { MxArray } from "@mlxts/core";

import {
  type ImageProofArtifactReport,
  imageProofImageToBmpBytes,
  writeImageProofBmp,
} from "../image-proof/artifact";

export type StableDiffusionBmpWriteResult = ImageProofArtifactReport;

/** Convert one Stable Diffusion proof image tensor into uncompressed BMP bytes. */
export function stableDiffusionImageToBmpBytes(image: MxArray): Uint8Array {
  return imageProofImageToBmpBytes(image, "Stable Diffusion");
}

/** Write one generated Stable Diffusion proof image and return verifier evidence. */
export function writeStableDiffusionBmp(
  image: MxArray,
  outputPath: string,
): StableDiffusionBmpWriteResult {
  return writeImageProofBmp(image, outputPath, {
    label: "Stable Diffusion",
  });
}
