import { formatShape, type MxArray } from "@mlxts/core";

import { ltx2AudioPackedLatentShape, ltxVideoPackedLatentShape } from "./latents";
import type { Ltx2Conditioning, Ltx2DenoiseOptions } from "./pipeline-ltx2-types";

export type ResolvedLtx2DenoiseShape = {
  batchSize: number;
  videoLength: number;
  audioLength: number;
  needsClassifierFreeGuidance: boolean;
};

function assertPositiveInteger(name: string, value: number): void {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${name} must be a positive integer.`);
  }
}

function assertNonNegativeFinite(name: string, value: number): void {
  if (!Number.isFinite(value) || value < 0) {
    throw new Error(`${name} must be a finite non-negative number.`);
  }
}

export function resolvePositiveFinite(
  value: number | undefined,
  fallback: number,
  name: string,
): number {
  const resolved = value ?? fallback;
  if (!Number.isFinite(resolved) || resolved <= 0) {
    throw new Error(`${name} must be a positive finite number.`);
  }
  return resolved;
}

function resolvePositiveInteger(value: number | undefined, fallback: number, name: string): number {
  const resolved = value ?? fallback;
  assertPositiveInteger(name, resolved);
  return resolved;
}

export function resolveGuidanceScale(value: number | undefined, name: string): number {
  const resolved = value ?? 1;
  assertNonNegativeFinite(name, resolved);
  return resolved;
}

function assertDefaultFinite(value: number | undefined, fallback: number, name: string): number {
  const resolved = value ?? fallback;
  if (!Number.isFinite(resolved)) {
    throw new Error(`${name} must be finite.`);
  }
  return resolved;
}

function assertUnsupportedGuidanceOptions(options: Ltx2DenoiseOptions): void {
  const stgScale = assertDefaultFinite(options.stgScale, 0, "stgScale");
  const audioStgScale = assertDefaultFinite(options.audioStgScale, stgScale, "audioStgScale");
  const modalityScale = assertDefaultFinite(options.modalityScale, 1, "modalityScale");
  const audioModalityScale = assertDefaultFinite(
    options.audioModalityScale,
    modalityScale,
    "audioModalityScale",
  );
  const guidanceRescale = assertDefaultFinite(options.guidanceRescale, 0, "guidanceRescale");
  const audioGuidanceRescale = assertDefaultFinite(
    options.audioGuidanceRescale,
    guidanceRescale,
    "audioGuidanceRescale",
  );
  if (stgScale !== 0 || audioStgScale !== 0) {
    throw new Error("LTX-2 STG requires the dedicated extra-forward guidance path.");
  }
  if (modalityScale !== 1 || audioModalityScale !== 1) {
    throw new Error("LTX-2 modality isolation guidance requires the dedicated extra-forward path.");
  }
  if (guidanceRescale !== 0 || audioGuidanceRescale !== 0) {
    throw new Error("LTX-2 guidance rescale is not part of prepared denoising yet.");
  }
  if ((options.spatioTemporalGuidanceBlocks?.length ?? 0) > 0) {
    throw new Error("LTX-2 STG block selection requires STG guidance to be implemented.");
  }
}

function usesClassifierFreeGuidance(options: Ltx2DenoiseOptions): boolean {
  const videoScale = resolveGuidanceScale(options.guidanceScale, "guidanceScale");
  const audioScale = resolveGuidanceScale(options.audioGuidanceScale, "audioGuidanceScale");
  return videoScale > 1 || audioScale > 1;
}

function assertPackedLatents(
  latents: MxArray,
  name: string,
  expectedLength: number,
): readonly [number, number, number] {
  const [batchSize, sequenceLength, channels] = latents.shape;
  if (
    latents.shape.length !== 3 ||
    batchSize === undefined ||
    sequenceLength === undefined ||
    channels === undefined
  ) {
    throw new Error(`${name} must be packed LTX-2 latents, got ${formatShape(latents.shape)}.`);
  }
  assertPositiveInteger("batchSize", batchSize);
  assertPositiveInteger("sequenceLength", sequenceLength);
  assertPositiveInteger("channels", channels);
  if (sequenceLength !== expectedLength) {
    throw new Error(`${name} sequence length must be ${expectedLength}, got ${sequenceLength}.`);
  }
  return [batchSize, sequenceLength, channels];
}

function assertPromptEmbeds(
  value: MxArray,
  name: string,
  batchSize: number,
): readonly [number, number] {
  const [promptBatch, textLength, hiddenSize] = value.shape;
  if (
    value.shape.length !== 3 ||
    promptBatch !== batchSize ||
    textLength === undefined ||
    hiddenSize === undefined
  ) {
    throw new Error(`${name} must have batch ${batchSize}, got ${formatShape(value.shape)}.`);
  }
  assertPositiveInteger("textLength", textLength);
  assertPositiveInteger("hiddenSize", hiddenSize);
  return [textLength, hiddenSize];
}

function assertPromptMask(
  mask: MxArray,
  name: string,
  batchSize: number,
  textLength: number,
): void {
  const [maskBatch, maskLength] = mask.shape;
  if (mask.shape.length !== 2 || maskBatch !== batchSize || maskLength !== textLength) {
    throw new Error(
      `${name} must have shape [${batchSize}, ${textLength}], got ${formatShape(mask.shape)}.`,
    );
  }
}

function assertMatchingPromptShape(positive: MxArray, negative: MxArray, owner: string): void {
  if (positive.shape.length !== negative.shape.length) {
    throw new Error(`${owner} rank must match the positive prompt embeddings.`);
  }
  for (let index = 1; index < positive.shape.length; index += 1) {
    if (positive.shape[index] !== negative.shape[index]) {
      throw new Error(`${owner} shape must match the positive prompt embeddings.`);
    }
  }
}

function assertConditioning(
  conditioning: Ltx2Conditioning,
  batchSize: number,
  needsClassifierFreeGuidance: boolean,
): void {
  const [textLength] = assertPromptEmbeds(
    conditioning.promptEmbeds,
    "conditioning.promptEmbeds",
    batchSize,
  );
  const [audioTextLength] = assertPromptEmbeds(
    conditioning.audioPromptEmbeds,
    "conditioning.audioPromptEmbeds",
    batchSize,
  );
  if (audioTextLength !== textLength) {
    throw new Error(
      "conditioning.audioPromptEmbeds text length must match conditioning.promptEmbeds.",
    );
  }
  assertPromptMask(
    conditioning.promptAttentionMask,
    "conditioning.promptAttentionMask",
    batchSize,
    textLength,
  );
  if (!needsClassifierFreeGuidance) {
    return;
  }
  assertGuidedConditioning(conditioning, batchSize, textLength);
}

function assertGuidedConditioning(
  conditioning: Ltx2Conditioning,
  batchSize: number,
  textLength: number,
): void {
  if (conditioning.negativePromptEmbeds === undefined) {
    throw new Error(
      "conditioning.negativePromptEmbeds is required when LTX-2 guidance is enabled.",
    );
  }
  if (conditioning.negativeAudioPromptEmbeds === undefined) {
    throw new Error(
      "conditioning.negativeAudioPromptEmbeds is required when LTX-2 guidance is enabled.",
    );
  }
  if (conditioning.negativePromptAttentionMask === undefined) {
    throw new Error(
      "conditioning.negativePromptAttentionMask is required when LTX-2 guidance is enabled.",
    );
  }
  assertPromptEmbeds(
    conditioning.negativePromptEmbeds,
    "conditioning.negativePromptEmbeds",
    batchSize,
  );
  assertPromptEmbeds(
    conditioning.negativeAudioPromptEmbeds,
    "conditioning.negativeAudioPromptEmbeds",
    batchSize,
  );
  assertMatchingPromptShape(
    conditioning.promptEmbeds,
    conditioning.negativePromptEmbeds,
    "conditioning.negativePromptEmbeds",
  );
  assertMatchingPromptShape(
    conditioning.audioPromptEmbeds,
    conditioning.negativeAudioPromptEmbeds,
    "conditioning.negativeAudioPromptEmbeds",
  );
  assertPromptMask(
    conditioning.negativePromptAttentionMask,
    "conditioning.negativePromptAttentionMask",
    batchSize,
    textLength,
  );
}

export function resolveDenoiseShape(options: Ltx2DenoiseOptions): ResolvedLtx2DenoiseShape {
  assertUnsupportedGuidanceOptions(options);
  assertPositiveInteger("numInferenceSteps", options.numInferenceSteps);
  assertPositiveInteger("latentFrames", options.latentFrames);
  assertPositiveInteger("latentHeight", options.latentHeight);
  assertPositiveInteger("latentWidth", options.latentWidth);
  assertPositiveInteger("audioLatentFrames", options.audioLatentFrames);
  assertPositiveInteger("audioLatentMelBins", options.audioLatentMelBins);
  const patchSize = resolvePositiveInteger(options.patchSize, 1, "patchSize");
  const patchSizeT = resolvePositiveInteger(options.patchSizeT, 1, "patchSizeT");
  const audioPatchSize = resolvePositiveInteger(options.audioPatchSize, 1, "audioPatchSize");
  const audioPatchSizeT = resolvePositiveInteger(options.audioPatchSizeT, 1, "audioPatchSizeT");
  const videoLength = ltxVideoPackedLatentShape(
    1,
    options.latentFrames,
    options.latentHeight,
    options.latentWidth,
    1,
    patchSize,
    patchSizeT,
  )[1];
  const audioLength = ltx2AudioPackedLatentShape(
    1,
    options.audioLatentFrames,
    options.audioLatentMelBins,
    1,
    audioPatchSize,
    audioPatchSizeT,
  )[1];
  const [batchSize] = assertPackedLatents(
    options.initialVideoLatents,
    "initialVideoLatents",
    videoLength,
  );
  const [audioBatch] = assertPackedLatents(
    options.initialAudioLatents,
    "initialAudioLatents",
    audioLength,
  );
  if (audioBatch !== batchSize) {
    throw new Error("initialAudioLatents batch must match initialVideoLatents.");
  }
  const needsCfg = usesClassifierFreeGuidance(options);
  assertConditioning(options.conditioning, batchSize, needsCfg);
  return {
    batchSize,
    videoLength,
    audioLength,
    needsClassifierFreeGuidance: needsCfg,
  };
}
