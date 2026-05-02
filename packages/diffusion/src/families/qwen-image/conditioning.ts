import { concatenate, divide, formatShape, type MxArray, retainArray, subtract } from "@mlxts/core";

import {
  type QwenImageLatentStats,
  qwenImageLatentStatsTensor,
  validateQwenImageLatentStats,
} from "./latent-stats";
import {
  packQwenImageLatents,
  type QwenImageRopeImageShape,
  qwenImageRopeImageShapeFromLatents,
} from "./latents";

/** VAE encoder surface required by Qwen-Image image conditioning paths. */
export type QwenImageLatentEncoder = QwenImageLatentStats & {
  encodeRaw(sample: MxArray): MxArray;
};

/** Packed reference image latents plus the RoPE shape segments they occupy. */
export type QwenImageReferenceLatents = {
  packedLatents: MxArray;
  imageShapes: readonly QwenImageRopeImageShape[];
};

type QwenImageReferenceLatentSegment = {
  packedLatents: MxArray;
  imageShape: QwenImageRopeImageShape;
  batchSize: number;
  packedChannels: number;
};

/** Encode an NCFHW sample volume into normalized packed Qwen-Image latents. */
export function encodeQwenImageLatents(
  vae: QwenImageLatentEncoder,
  sample: MxArray,
  patchSize = 2,
): MxArray {
  validateQwenImageLatentStats("encodeQwenImageLatents", vae);

  using rawLatents = vae.encodeRaw(sample);
  using mean = qwenImageLatentStatsTensor(vae.latentsMean, vae.latentChannels, rawLatents.dtype);
  using std = qwenImageLatentStatsTensor(vae.latentsStd, vae.latentChannels, rawLatents.dtype);
  using shifted = subtract(rawLatents, mean);
  using normalized = divide(shifted, std);
  return packQwenImageLatents(normalized, patchSize);
}

function encodeReferenceLatentSegment(
  vae: QwenImageLatentEncoder,
  sample: MxArray,
  patchSize: number,
): QwenImageReferenceLatentSegment {
  using rawLatents = vae.encodeRaw(sample);
  const imageShape = qwenImageRopeImageShapeFromLatents(rawLatents, patchSize);
  using mean = qwenImageLatentStatsTensor(vae.latentsMean, vae.latentChannels, rawLatents.dtype);
  using std = qwenImageLatentStatsTensor(vae.latentsStd, vae.latentChannels, rawLatents.dtype);
  using shifted = subtract(rawLatents, mean);
  using normalized = divide(shifted, std);
  const packedLatents = packQwenImageLatents(normalized, patchSize);
  try {
    const [batchSize, packedChannels] = expectPackedReferenceLatents(packedLatents);
    return { packedLatents, imageShape, batchSize, packedChannels };
  } catch (error) {
    packedLatents.free();
    throw error;
  }
}

function expectPackedReferenceLatents(packedLatents: MxArray): readonly [number, number] {
  const [batchSize, , packedChannels] = packedLatents.shape;
  if (packedLatents.shape.length !== 3 || batchSize === undefined || packedChannels === undefined) {
    throw new Error(
      `prepareQwenImageReferenceLatents: expected packed latents, got ${formatShape(
        packedLatents.shape,
      )}.`,
    );
  }
  return [batchSize, packedChannels];
}

function assertCompatibleReferenceSegment(
  segment: QwenImageReferenceLatentSegment,
  expectedShape: Pick<QwenImageReferenceLatentSegment, "batchSize" | "packedChannels">,
): void {
  if (segment.batchSize !== expectedShape.batchSize) {
    throw new Error("prepareQwenImageReferenceLatents: reference batches must match.");
  }
  if (segment.packedChannels !== expectedShape.packedChannels) {
    throw new Error("prepareQwenImageReferenceLatents: reference packed channels must match.");
  }
}

/** Prepare packed reference latents for future Qwen-Image edit denoising paths. */
export function prepareQwenImageReferenceLatents(
  vae: QwenImageLatentEncoder,
  samples: readonly MxArray[],
  patchSize = 2,
): QwenImageReferenceLatents {
  validateQwenImageLatentStats("prepareQwenImageReferenceLatents", vae);
  if (samples.length === 0) {
    throw new Error("prepareQwenImageReferenceLatents: at least one reference sample is required.");
  }

  const packedSegments: MxArray[] = [];
  const imageShapes: QwenImageRopeImageShape[] = [];
  let expectedShape:
    | Pick<QwenImageReferenceLatentSegment, "batchSize" | "packedChannels">
    | undefined;
  try {
    for (const sample of samples) {
      const segment = encodeReferenceLatentSegment(vae, sample, patchSize);
      packedSegments.push(segment.packedLatents);
      if (expectedShape === undefined) {
        expectedShape = {
          batchSize: segment.batchSize,
          packedChannels: segment.packedChannels,
        };
      } else {
        assertCompatibleReferenceSegment(segment, expectedShape);
      }
      imageShapes.push(segment.imageShape);
    }

    const first = packedSegments[0];
    if (first === undefined) {
      throw new Error("prepareQwenImageReferenceLatents: missing packed reference latents.");
    }
    let packedLatents: MxArray;
    if (packedSegments.length === 1) {
      packedLatents = retainArray(first);
    } else {
      packedLatents = concatenate(packedSegments, 1);
    }
    return { packedLatents, imageShapes };
  } finally {
    for (const packed of packedSegments) {
      packed.free();
    }
  }
}
