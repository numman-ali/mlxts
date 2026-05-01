import type { MxArray } from "@mlxts/core";

import {
  type ImageProofArtifactReport,
  imageProofImageToBmpBytes,
  writeImageProofBmp,
} from "../image-proof/artifact";

export type FluxBmpWriteResult = ImageProofArtifactReport;

/** Convert one FLUX proof image tensor into uncompressed BMP bytes. */
export function fluxImageToBmpBytes(image: MxArray): Uint8Array {
  return imageProofImageToBmpBytes(image, "FLUX");
}

/** Write one generated FLUX proof image and return verifier evidence. */
export function writeFluxBmp(image: MxArray, outputPath: string): FluxBmpWriteResult {
  return writeImageProofBmp(image, outputPath, {
    label: "FLUX",
  });
}
