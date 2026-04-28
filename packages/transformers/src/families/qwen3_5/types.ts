/**
 * Local config contracts for the Qwen 3.5 multimodal wrapper family.
 * @module
 */

export type Qwen3_5Family = "qwen";

export type Qwen3_5ModelType = "qwen3_5" | "qwen3_5_moe";

export type Qwen3_5TextModelType = "qwen3_5_text" | "qwen3_5_moe_text";

export type Qwen3_5VisionModelType = "qwen3_5" | "qwen3_5_vision";

export type Qwen3_5LayerType = "linear_attention" | "full_attention";

export type Qwen3_5PatchShape = number | readonly number[];

export type Qwen3_5TokenId = number | null;

export type Qwen3_5EosTokenId = number | readonly number[] | null;

export type Qwen3_5TextRopeParameters = {
  ropeType: string;
  ropeTheta: number;
  partialRotaryFactor: number;
  mropeSection: readonly number[];
  mropeInterleaved: boolean;
};

export type Qwen3_5TextConfig = {
  family: Qwen3_5Family;
  modelType: Qwen3_5TextModelType;
  rawConfig: Record<string, unknown>;
  vocabSize: number;
  hiddenSize: number;
  intermediateSize: number;
  feedForwardKind: "dense" | "moe";
  moeIntermediateSize: number | null;
  sharedExpertIntermediateSize: number | null;
  numExperts: number | null;
  numExpertsPerToken: number | null;
  routerAuxLossCoef: number | null;
  numHiddenLayers: number;
  numAttentionHeads: number;
  numKeyValueHeads: number;
  headDim: number;
  hiddenAct: string;
  maxPositionEmbeddings: number;
  initializerRange: number;
  rmsNormEps: number;
  useCache: boolean;
  tieWordEmbeddings: boolean;
  attentionBias: boolean;
  attentionDropout: number;
  attnOutputGate: boolean;
  outputGateType: string | null;
  linearConvKernelDim: number;
  linearKeyHeadDim: number;
  linearValueHeadDim: number;
  linearNumKeyHeads: number;
  linearNumValueHeads: number;
  layerTypes: Qwen3_5LayerType[];
  fullAttentionInterval: number | null;
  ropeParameters: Qwen3_5TextRopeParameters;
  partialRotaryFactor: number;
  mtpNumHiddenLayers: number;
  mtpUseDedicatedEmbeddings: boolean;
  mambaSsmDtype: string | null;
  bosTokenId: Qwen3_5TokenId;
  eosTokenId: Qwen3_5EosTokenId;
  padTokenId: Qwen3_5TokenId;
};

export type Qwen3_5VisionConfig = {
  family: Qwen3_5Family;
  modelType: Qwen3_5VisionModelType;
  rawConfig: Record<string, unknown>;
  depth: number;
  hiddenSize: number;
  hiddenAct: string;
  intermediateSize: number;
  numHeads: number;
  inChannels: number;
  patchSize: Qwen3_5PatchShape;
  spatialMergeSize: number;
  temporalPatchSize: Qwen3_5PatchShape;
  outHiddenSize: number;
  numPositionEmbeddings: number;
  deepstackVisualIndexes: readonly number[];
  initializerRange: number;
};

export type Qwen3_5Config = {
  family: Qwen3_5Family;
  modelType: Qwen3_5ModelType;
  rawConfig: Record<string, unknown>;
  vocabSize: number;
  hiddenSize: number;
  numHiddenLayers: number;
  textConfig: Qwen3_5TextConfig;
  visionConfig: Qwen3_5VisionConfig;
  imageTokenId: number;
  videoTokenId: number;
  visionStartTokenId: number;
  visionEndTokenId: number;
  tieWordEmbeddings: boolean;
  languageModelOnly: boolean;
};
