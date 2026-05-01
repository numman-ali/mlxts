/**
 * Whisper config and audio preprocessing contracts.
 * @module
 */

import type { MxArray } from "@mlxts/core";

export type WhisperActivation = "gelu";

export type WhisperConfig = {
  modelType: "whisper";
  rawConfig: Record<string, unknown>;
  vocabSize: number;
  numMelBins: number;
  encoderLayers: number;
  encoderAttentionHeads: number;
  decoderLayers: number;
  decoderAttentionHeads: number;
  encoderFfnDim: number;
  decoderFfnDim: number;
  dModel: number;
  encoderHeadDim: number;
  decoderHeadDim: number;
  activationFunction: WhisperActivation;
  maxSourcePositions: number;
  maxTargetPositions: number;
  padTokenId: number | null;
  bosTokenId: number | null;
  eosTokenId: number | readonly number[] | null;
  decoderStartTokenId: number;
  scaleEmbedding: boolean;
  useCache: boolean;
};

export type WhisperFeatureExtractorConfig = {
  featureSize: number;
  samplingRate: number;
  hopLength: number;
  chunkLength: number;
  nFft: number;
  paddingValue: number;
  nSamples: number;
  nFrames: number;
};

export type WhisperAudioFeatures = {
  inputFeatures: MxArray;
};

export type WhisperRunOptions = {
  outputHiddenStates?: boolean;
};

export type WhisperEncoderOutput = {
  lastHiddenState: MxArray;
  hiddenStates?: MxArray[];
};

export type WhisperDecoderOutput = {
  lastHiddenState: MxArray;
  hiddenStates?: MxArray[];
};

export type WhisperModelOutput = {
  lastHiddenState: MxArray;
  encoderLastHiddenState: MxArray;
  encoderHiddenStates?: MxArray[];
  decoderHiddenStates?: MxArray[];
};

export type WhisperConditionalGenerationOutput = WhisperModelOutput & {
  logits: MxArray;
};
