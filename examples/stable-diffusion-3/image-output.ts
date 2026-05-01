import type { MxArray } from "@mlxts/core";

import {
  type ImageProofArtifactReport,
  imageProofImageToBmpBytes,
  writeImageProofBmp,
} from "../image-proof/artifact";

export type StableDiffusion3BmpWriteResult = ImageProofArtifactReport;

/** Convert one Stable Diffusion 3 proof image tensor into uncompressed BMP bytes. */
export function stableDiffusion3ImageToBmpBytes(image: MxArray): Uint8Array {
  return imageProofImageToBmpBytes(image, "Stable Diffusion 3");
}

/** Write one generated Stable Diffusion 3 proof image and return verifier evidence. */
export function writeStableDiffusion3Bmp(
  image: MxArray,
  outputPath: string,
): StableDiffusion3BmpWriteResult {
  return writeImageProofBmp(image, outputPath, {
    label: "Stable Diffusion 3",
  });
}
