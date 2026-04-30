/**
 * T5 encoder config and output contracts.
 * @module
 */

import type { MxArray } from "@mlxts/core";

export type T5DenseActivation = "relu" | "gelu" | "gelu_new" | "silu";

export type T5FeedForwardProjection = "relu" | "gated-gelu" | "gated-silu";

export type T5EncoderConfig = {
  modelType: "t5_encoder_model";
  rawConfig: Record<string, unknown>;
  vocabSize: number;
  dModel: number;
  dKv: number;
  dFf: number;
  numLayers: number;
  numHeads: number;
  innerDim: number;
  relativeAttentionNumBuckets: number;
  relativeAttentionMaxDistance: number;
  layerNormEps: number;
  dropoutRate: number;
  feedForwardProjection: T5FeedForwardProjection;
  denseActivation: T5DenseActivation;
  isGatedActivation: boolean;
  padTokenId: number | null;
  eosTokenId: number | null;
};

export type T5EncoderModelOptions = {
  outputHiddenStates?: boolean;
};

export type T5EncoderModelOutput = {
  lastHiddenState: MxArray;
  hiddenStates?: MxArray[];
};
