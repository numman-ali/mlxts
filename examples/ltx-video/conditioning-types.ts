import type { DType, MxArray } from "@mlxts/core";
import type { LtxVideoConditioning } from "@mlxts/diffusion";
import type { SentencePieceTokenizer } from "@mlxts/tokenizers";
import type {
  LoadT5EncoderModelOptions,
  T5EncoderModelOptions,
  T5EncoderModelOutput,
} from "@mlxts/transformers";

export type LtxVideoPrompt = string | readonly string[];

export type LtxVideoPromptConditioningOptions = {
  prompt: LtxVideoPrompt;
  negativePrompt?: LtxVideoPrompt;
  numVideosPerPrompt?: number;
  maxSequenceLength?: number;
  includeNegativePrompt?: boolean;
  maskDType?: DType;
};

export type LtxVideoPromptConditioning = Disposable & {
  readonly batchSize: number;
  readonly conditioning: LtxVideoConditioning;
  readonly promptTruncated: boolean;
  readonly negativePromptTruncated: boolean;
};

export type LtxVideoPromptConditioner = Disposable & {
  readonly pipelineKind: "ltx-video";
  encodePrompt(options: LtxVideoPromptConditioningOptions): LtxVideoPromptConditioning;
};

export type LtxVideoT5TextEncoder = Disposable & {
  run(inputIds: MxArray, options?: T5EncoderModelOptions): T5EncoderModelOutput;
};

export type LtxVideoPromptConditionerComponents = {
  tokenizer: SentencePieceTokenizer;
  textEncoder: LtxVideoT5TextEncoder;
};

export type LoadLtxVideoPromptConditionerOptions = {
  textEncoder?: LoadT5EncoderModelOptions;
};
