import type { MxArray } from "@mlxts/core";

import {
  type ImageProofArtifactReport,
  imageProofImageToBmpBytes,
  writeImageProofBmp,
} from "../image-proof/artifact";

export type Flux2KleinBmpWriteResult = ImageProofArtifactReport;

/** Convert one FLUX.2 Klein proof image tensor into uncompressed BMP bytes. */
export function flux2KleinToBmpBytes(image: MxArray): Uint8Array {
  return imageProofImageToBmpBytes(image, "FLUX.2 Klein");
}

/** Write one generated FLUX.2 Klein proof image and return verifier evidence. */
export function writeFlux2KleinBmp(image: MxArray, outputPath: string): Flux2KleinBmpWriteResult {
  return writeImageProofBmp(image, outputPath, {
    label: "FLUX.2 Klein",
  });
}
