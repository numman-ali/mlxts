import type { MxArray } from "@mlxts/core";
import type {
  Ltx2Conditioning,
  Ltx2TextConnectorOutput,
  Ltx2TextConnectorsConfig,
  Ltx2TextConnectorWeightLoadOptions,
} from "@mlxts/diffusion";
import type { Tokenizer } from "@mlxts/tokenizers";
import type {
  Gemma3TextModelOptions,
  Gemma3TextModelOutput,
  LoadCausalLMOptions,
  LoadPretrainedTokenizerOptions,
} from "@mlxts/transformers";

export type Ltx2Prompt = string | readonly string[];

export type Ltx2PromptConditioningOptions = {
  prompt: Ltx2Prompt;
  negativePrompt?: Ltx2Prompt;
  numVideosPerPrompt?: number;
  maxSequenceLength?: number;
  includeNegativePrompt?: boolean;
};

export type Ltx2PromptConditioning = Disposable & {
  readonly batchSize: number;
  readonly conditioning: Ltx2Conditioning;
  readonly promptTruncated: boolean;
  readonly negativePromptTruncated: boolean;
};

export type Ltx2PromptConditioner = Disposable & {
  readonly pipelineKind: "ltx2";
  encodePrompt(options: Ltx2PromptConditioningOptions): Ltx2PromptConditioning;
};

export type Ltx2Gemma3TextModel = {
  runWithHiddenStates(inputIds: MxArray, options?: Gemma3TextModelOptions): Gemma3TextModelOutput;
};

export type Ltx2Gemma3TextEncoder = Disposable & {
  readonly model: Ltx2Gemma3TextModel;
};

export type Ltx2PromptConnector = Disposable & {
  readonly config: Ltx2TextConnectorsConfig;
  run(textEncoderHiddenStates: MxArray, attentionMask: MxArray): Ltx2TextConnectorOutput;
};

export type Ltx2PromptConditionerComponents = {
  tokenizer: Tokenizer;
  textEncoder: Ltx2Gemma3TextEncoder;
  connectors: Ltx2PromptConnector;
};

export type LoadLtx2PromptConditionerOptions = {
  tokenizer?: LoadPretrainedTokenizerOptions;
  textEncoder?: LoadCausalLMOptions;
  connectors?: Ltx2TextConnectorWeightLoadOptions;
};
