/**
 * Public contracts for pretrained decoder loading and generation.
 * @module
 */

import type { MxArray, ParameterTree } from "@mlxts/core";
import type { TokenizerFormat } from "@mlxts/tokenizers";
import type { CacheLayerKind } from "./infrastructure/cache/layer-kind";
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
  /** Full sampler history when cache input is only a prompt suffix. */
  samplerHistoryTokenIds?: readonly number[];
  /** Receives an owned prompt-boundary cache snapshot after prefill. The callback owns it. */
  onPromptCacheSnapshot?: (event: PromptCacheSnapshotEvent) => void;
  addSpecialTokens?: boolean;
  prefillStepSize?: number;
  /** Cooperative cancellation signal checked between prefill chunks and decode steps. */
  abortSignal?: AbortSignal;
  /** Called after each cached prompt-prefill chunk completes. */
  onPrefillProgress?: (event: PrefillProgressEvent) => void;
};

export type BatchGenerationOptions = Omit<GenerationOptions, "maxTokens" | "onPrefillProgress"> & {
  maxTokens: number | readonly number[];
  padTokenId?: number;
};

export type GenerationDefaults = Omit<
  GenerationOptions,
  | "addSpecialTokens"
  | "maxTokens"
  | "cache"
  | "samplerHistoryTokenIds"
  | "onPromptCacheSnapshot"
  | "onPrefillProgress"
  | "abortSignal"
>;

export type GenerationResult = {
  tokenIds: number[];
  finishReason: "length" | "eos";
};

/** Progress emitted while chunking a prompt into the model cache before decode. */
export type PrefillProgressEvent = {
  processedTokens: number;
  totalTokens: number;
  chunkTokens: number;
};

/** Owned cache snapshot emitted after prompt prefill for prompt-prefix reuse. */
export type PromptCacheSnapshotEvent = {
  /** Logical cache offset captured by the snapshot. */
  offset: number;
  /** Owned snapshot; the callback must store or dispose it. */
  snapshot: TransformerCacheSnapshot;
};

/** Incremental token-generation event for a single request. */
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

/** Options for restoring a cache snapshot at a supported logical prefix. */
export type TransformerCacheForkOptions = {
  /** Logical prefix offset to restore. Defaults to the snapshot's full offset. */
  offset?: number;
};

/** Owned prompt/cache snapshot that can produce disposable request-local cache forks. */
export interface TransformerCacheSnapshot extends Disposable {
  /** Logical token offset represented by this snapshot. */
  readonly offset: number;
  /** Per-decoder-layer semantic cache state kind. */
  readonly layerKinds: readonly CacheLayerKind[];
  /** Whether this snapshot can fork at earlier offsets as well as its full offset. */
  readonly trimmable: boolean;

  /** Return whether `fork()` can produce a cache at the requested offset. */
  canFork(options?: TransformerCacheForkOptions): boolean;
  /** Return a disposable cache fork owned by the caller. */
  fork(options?: TransformerCacheForkOptions): TransformerCache;
}

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
  readonly layerKinds: readonly CacheLayerKind[];
  readonly offset: number;

  updateAndFetch(
    layerIndex: number,
    keys: MxArray,
    values: MxArray,
  ): { keys: MxArray; values: MxArray };
  advance(sequenceLength: number): void;
  isEmpty(): boolean;
  isTrimmable(): boolean;
  snapshot(): TransformerCacheSnapshot;
  /** Return retained cache-state arrays for explicit eval. The caller owns the returned views. */
  arrays(): MxArray[];
}

export interface TransformerBatchCache extends Disposable {
  readonly layerCount: number;
  readonly layerKinds: readonly CacheLayerKind[];
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
