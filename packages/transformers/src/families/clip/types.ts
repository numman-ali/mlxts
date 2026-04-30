/**
 * CLIP text encoder config and output contracts.
 * @module
 */

import type { MxArray } from "@mlxts/core";

export type CLIPHiddenActivation = "quick_gelu" | "gelu";

export type CLIPTextConfig = {
  modelType: "clip_text_model";
  rawConfig: Record<string, unknown>;
  vocabSize: number;
  hiddenSize: number;
  intermediateSize: number;
  projectionDim: number | null;
  numHiddenLayers: number;
  numAttentionHeads: number;
  headDim: number;
  maxPositionEmbeddings: number;
  hiddenAct: CLIPHiddenActivation;
  layerNormEps: number;
  attentionDropout: number;
  padTokenId: number | null;
  bosTokenId: number | null;
  eosTokenId: number | null;
};

export type CLIPTextModelOptions = {
  outputHiddenStates?: boolean;
};

export type CLIPTextModelOutput = {
  lastHiddenState: MxArray;
  pooledOutput: MxArray;
  hiddenStates?: MxArray[];
};

export type CLIPTextProjectionOutput = CLIPTextModelOutput & {
  textEmbeds: MxArray;
};
