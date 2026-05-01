import { asType, expandDims, formatShape, type MxArray, retainArray } from "@mlxts/core";

import {
  disposeLtx2AdaLayerNormSingleOutput,
  type Ltx2AdaLayerNormSingle,
} from "./conditioning-ltx2";
import type { Ltx2VideoTransformerConfig } from "./config";
import type { LtxRotaryEmbeddings } from "./embeddings";
import type { Ltx2DenoiserInput } from "./pipeline-ltx2-types";
import { sliceAxis } from "./tensor-utils";

export type Ltx2TransformerEmbeddings = {
  videoRotary: LtxRotaryEmbeddings;
  audioRotary: LtxRotaryEmbeddings;
  caVideoRotary: LtxRotaryEmbeddings;
  caAudioRotary: LtxRotaryEmbeddings;
};

export type Ltx2TransformerTiming = {
  video: ReturnType<Ltx2AdaLayerNormSingle["embed"]>;
  audio: ReturnType<Ltx2AdaLayerNormSingle["embed"]>;
  videoCrossScaleShift: ReturnType<Ltx2AdaLayerNormSingle["embed"]>;
  audioCrossScaleShift: ReturnType<Ltx2AdaLayerNormSingle["embed"]>;
  videoCrossGate: ReturnType<Ltx2AdaLayerNormSingle["embed"]>;
  audioCrossGate: ReturnType<Ltx2AdaLayerNormSingle["embed"]>;
};

export function partAt(parts: readonly MxArray[], index: number, owner: string): MxArray {
  const part = parts[index];
  if (part === undefined) {
    throw new Error(`${owner}: split failed.`);
  }
  return part;
}

export function disposeRotaryEmbeddings(embeddings: LtxRotaryEmbeddings): void {
  embeddings.cos.free();
  embeddings.sin.free();
}

export function disposeTransformerEmbeddings(embeddings: Ltx2TransformerEmbeddings): void {
  disposeRotaryEmbeddings(embeddings.videoRotary);
  disposeRotaryEmbeddings(embeddings.audioRotary);
  disposeRotaryEmbeddings(embeddings.caVideoRotary);
  disposeRotaryEmbeddings(embeddings.caAudioRotary);
}

export function disposeTiming(timing: Ltx2TransformerTiming): void {
  disposeLtx2AdaLayerNormSingleOutput(timing.video);
  disposeLtx2AdaLayerNormSingleOutput(timing.audio);
  disposeLtx2AdaLayerNormSingleOutput(timing.videoCrossScaleShift);
  disposeLtx2AdaLayerNormSingleOutput(timing.audioCrossScaleShift);
  disposeLtx2AdaLayerNormSingleOutput(timing.videoCrossGate);
  disposeLtx2AdaLayerNormSingleOutput(timing.audioCrossGate);
}

export function validateLtx2VideoTransformerConfig(config: Ltx2VideoTransformerConfig): void {
  if (config.patchSize !== 1 || config.patchSizeT !== 1) {
    throw new Error("Ltx2VideoTransformer3DModel: video patch sizes other than 1 are unsupported.");
  }
  if (config.audioPatchSize !== 1 || config.audioPatchSizeT !== 1) {
    throw new Error("Ltx2VideoTransformer3DModel: audio patch sizes other than 1 are unsupported.");
  }
  if (config.activationFn !== "gelu-approximate") {
    throw new Error("Ltx2VideoTransformer3DModel: only gelu-approximate activation is supported.");
  }
  if (config.qkNorm !== "rms_norm_across_heads") {
    throw new Error(
      "Ltx2VideoTransformer3DModel: only rms_norm_across_heads qk_norm is supported.",
    );
  }
  if (config.normElementwiseAffine) {
    throw new Error("Ltx2VideoTransformer3DModel: affine LTX-2 norms are unsupported.");
  }
  if (!config.usePromptEmbeddings) {
    throw new Error("Ltx2VideoTransformer3DModel: prepared prompt embeddings are required.");
  }
  if (config.gatedAttn || config.audioGatedAttn) {
    throw new Error("Ltx2VideoTransformer3DModel: gated attention is unsupported.");
  }
  if (config.crossAttnMod || config.audioCrossAttnMod) {
    throw new Error(
      "Ltx2VideoTransformer3DModel: prompt cross-attention modulation is unsupported.",
    );
  }
  if (config.perturbedAttn) {
    throw new Error("Ltx2VideoTransformer3DModel: perturbed attention requires the STG path.");
  }
}

export function encoderAttentionMask(mask: MxArray, batch: number, textLength: number): MxArray {
  if (mask.shape.length !== 2 || mask.shape[0] !== batch || mask.shape[1] !== textLength) {
    throw new Error(
      `Ltx2VideoTransformer3DModel.forward: encoderAttentionMask must have shape [${batch}, ${textLength}], got ${formatShape(
        mask.shape,
      )}.`,
    );
  }
  using boolMask = mask.dtype === "bool" ? retainArray(mask) : asType(mask, "bool");
  using keyMask = expandDims(boolMask, 1);
  return expandDims(keyMask, 1);
}

export function timestepLike(input: Ltx2DenoiserInput, useSigma: boolean): MxArray {
  return useSigma ? retainArray(input.sigma) : retainArray(input.timestep);
}

export function temporalVideoCoords(videoCoords: MxArray): MxArray {
  return sliceAxis(videoCoords, 1, 0, 1);
}
