/**
 * Public contracts for pretrained decoder loading and generation.
 * @module
 */

import type { MxArray, ParameterTree } from "@mlxts/core";
import type { TokenizerFormat } from "@mlxts/tokenizers";

import type {
  LoadSourceOptions,
  PretrainedLoadProgressEvent,
  ResolvedSnapshot,
  SnapshotInspection,
} from "./pretrained/types";

export type SupportedModelFamily = "llama" | "mistral" | "gemma" | "phi" | "qwen";

export type ForwardOptions = {
  cache?: DecoderCache;
  inputEmbeddings?: MxArray;
  positionIds?: MxArray;
};

export type SamplerOptions = {
  temperature?: number;
  topK?: number;
  topP?: number;
  minP?: number;
  repetitionPenalty?: number;
  seed?: number;
};

export type GenerationOptions = SamplerOptions & {
  maxTokens: number;
  eosTokenIds?: number[];
  useCache?: boolean;
  cache?: TransformerCache;
  addSpecialTokens?: boolean;
  prefillStepSize?: number;
};

export type BatchGenerationOptions = Omit<GenerationOptions, "maxTokens"> & {
  maxTokens: number | readonly number[];
  padTokenId?: number;
};

export type GenerationDefaults = Omit<
  GenerationOptions,
  "addSpecialTokens" | "maxTokens" | "cache"
>;

export type GenerationResult = {
  tokenIds: number[];
  finishReason: "length" | "eos";
};

export type TokenGenerationEvent =
  | {
      type: "token";
      tokenId: number;
      completionTokens: number;
    }
  | {
      type: "done";
      tokenIds: readonly number[];
      finishReason: GenerationResult["finishReason"];
    };

export type BatchTokenGenerationEvent =
  | {
      type: "token";
      batchIndex: number;
      tokenId: number;
      completionTokens: number;
    }
  | {
      type: "done";
      batchIndex: number;
      tokenIds: readonly number[];
      finishReason: GenerationResult["finishReason"];
    };

export type TextGenerationResult = GenerationResult & {
  text: string;
};

export type PreparedPrompt = {
  tokenIds: readonly number[];
  inputEmbeddings?: MxArray;
  positionIds?: MxArray;
};

export type LoadCausalLMOptions = LoadSourceOptions & {
  strictUnexpectedWeights?: boolean;
};

export type LoadPretrainedTokenizerOptions = LoadSourceOptions & {
  format?: TokenizerFormat;
};

export type { LoadSourceOptions, PretrainedLoadProgressEvent };

export type BaseModelConfig = {
  family: SupportedModelFamily;
  modelType: string;
  rawConfig: Record<string, unknown>;
  vocabSize: number;
  hiddenSize: number;
  numHiddenLayers: number;
  generationDefaults?: GenerationDefaults;
};

export interface TransformerCache extends Disposable {
  readonly layerCount: number;
  readonly offset: number;

  updateAndFetch(
    layerIndex: number,
    keys: MxArray,
    values: MxArray,
  ): { keys: MxArray; values: MxArray };
  advance(sequenceLength: number): void;
  isEmpty(): boolean;
  isTrimmable(): boolean;
  /** Return retained cache-state arrays for explicit eval. The caller owns the returned views. */
  arrays(): MxArray[];
}

export interface TransformerBatchCache extends Disposable {
  readonly layerCount: number;
  readonly batchSize: number;
  readonly length: number;
  readonly leftPadding: readonly number[];
  readonly offsets: readonly number[];

  updateAndFetch(
    layerIndex: number,
    keys: MxArray,
    values: MxArray,
  ): { keys: MxArray; values: MxArray };
  advance(sequenceLength: number): void;
  filter(batchIndices: readonly number[]): void;
  extend(other: TransformerBatchCache): void;
  extract(batchIndex: number): TransformerCache;
  /** Return per-request RoPE offsets as an int32 tensor. The caller owns the returned array. */
  offsetTensor(): MxArray;
  /** Return per-request left-padding lengths as an int32 tensor. The caller owns the returned array. */
  leftPaddingTensor(): MxArray;
  isEmpty(): boolean;
  isTrimmable(): boolean;
  /** Return retained cache-state arrays for explicit eval. The caller owns the returned views. */
  arrays(): MxArray[];
}

export type DecoderCache = TransformerCache | TransformerBatchCache;

export interface CausalLM extends Disposable {
  readonly family: SupportedModelFamily;
  readonly layerCount: number;
  readonly config: BaseModelConfig;

  forward(inputIds: MxArray, options?: ForwardOptions): MxArray;
  createCache(): TransformerCache;

  parameters(): ParameterTree;
  trainableParameters(): ParameterTree;
  update(params: ParameterTree): void;
  freeze(keys?: string[]): this;
  unfreeze(keys?: string[]): this;
  eval(): this;
  train(mode?: boolean): this;
}

export type ExceptionalWeightLoaderContext<Config extends BaseModelConfig = BaseModelConfig> = {
  snapshot: ResolvedSnapshot;
  config: Config;
  model: CausalLM;
  assignWeight(path: string, tensor: MxArray): void;
};

export type CheckpointTensorTransform<_Config extends BaseModelConfig = BaseModelConfig> = (
  checkpointName: string,
  weightPath: string,
  tensor: MxArray,
) => MxArray;

export type CheckpointTensorTransformContext<Config extends BaseModelConfig = BaseModelConfig> = {
  snapshot: ResolvedSnapshot;
  inspection: SnapshotInspection;
  config: Config;
};

export type FamilyRegistration<Config extends BaseModelConfig = BaseModelConfig> = {
  readonly family: SupportedModelFamily;
  readonly modelTypes: readonly string[];

  parseConfig(rawConfig: Record<string, unknown>): Config;
  createModel(config: Config): CausalLM;
  sanitizeWeight(config: Config, checkpointName: string): string | null;
  isIgnoredWeight?(config: Config, checkpointName: string): boolean;
  createCheckpointTensorTransform?(
    context: CheckpointTensorTransformContext<Config>,
  ):
    | Promise<CheckpointTensorTransform<Config> | undefined>
    | CheckpointTensorTransform<Config>
    | undefined;
  exceptionalWeightNames?(config: Config): readonly string[];
  loadExceptionalWeights?(context: ExceptionalWeightLoaderContext<Config>): Promise<void>;
};

/** `model_type` is unsupported by the current registry. */
export class UnsupportedModelError extends Error {
  constructor(modelType: string, supportedModelTypes: readonly string[]) {
    super(
      `loadCausalLM: unsupported model_type "${modelType}". Supported model types: ${supportedModelTypes.join(", ")}.`,
    );
    this.name = "UnsupportedModelError";
  }
}

/** A config field is missing or has the wrong runtime shape. */
export class ConfigParseError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ConfigParseError";
  }
}

/** A checkpoint tensor shape does not match the target model parameter. */
export class WeightMismatchError extends Error {
  constructor(path: string, expectedShape: readonly number[], actualShape: readonly number[]) {
    super(
      `loadCausalLM: checkpoint tensor for "${path}" has shape [${actualShape.join(", ")}], expected [${expectedShape.join(", ")}].`,
    );
    this.name = "WeightMismatchError";
  }
}

/** Required model parameters were still unassigned after all checkpoint shards loaded. */
export class MissingWeightsError extends Error {
  constructor(paths: readonly string[]) {
    const sortedPaths = [...paths].toSorted((left, right) => left.localeCompare(right));
    super(`loadCausalLM: checkpoint is missing required parameters: ${sortedPaths.join(", ")}.`);
    this.name = "MissingWeightsError";
  }
}
