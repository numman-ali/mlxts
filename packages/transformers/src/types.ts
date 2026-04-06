/**
 * Public contracts for pretrained decoder loading and generation.
 * @module
 */

import type { MxArray, ParameterTree } from "@mlxts/core";
import type { TokenizerFormat } from "@mlxts/tokenizers";

import type { LoadSourceOptions, PretrainedLoadProgressEvent } from "./pretrained/types";

export type SupportedModelFamily = "llama" | "mistral" | "gemma" | "phi";

export type ForwardOptions = {
  cache?: TransformerCache;
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

export type GenerationDefaults = Omit<
  GenerationOptions,
  "addSpecialTokens" | "maxTokens" | "cache"
>;

export type GenerationResult = {
  tokenIds: number[];
  finishReason: "length" | "eos";
};

export type TextGenerationResult = GenerationResult & {
  text: string;
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
  /** Return retained cache-state arrays for explicit eval. The caller owns the returned views. */
  arrays(): MxArray[];
}

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
  snapshot: import("./pretrained/types").ResolvedSnapshot;
  config: Config;
  model: CausalLM;
  assignWeight(path: string, tensor: MxArray): void;
};

export type FamilyRegistration<Config extends BaseModelConfig = BaseModelConfig> = {
  readonly family: SupportedModelFamily;
  readonly modelTypes: readonly string[];

  parseConfig(rawConfig: Record<string, unknown>): Config;
  createModel(config: Config): CausalLM;
  sanitizeWeight(config: Config, checkpointName: string): string | null;
  isIgnoredWeight?(config: Config, checkpointName: string): boolean;
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
