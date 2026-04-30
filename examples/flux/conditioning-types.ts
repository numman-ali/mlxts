import type { DType, MxArray } from "@mlxts/core";
import type { FluxConditioning } from "@mlxts/diffusion";
import type { CLIPTokenizer, SentencePieceTokenizer } from "@mlxts/tokenizers";
import type {
  CLIPTextModelOptions,
  CLIPTextModelOutput,
  LoadCLIPTextModelOptions,
  LoadT5EncoderModelOptions,
  T5EncoderModelOptions,
  T5EncoderModelOutput,
} from "@mlxts/transformers";

export type FluxPrompt = string | readonly string[];

export type FluxPromptConditioningOptions = {
  prompt: FluxPrompt;
  prompt2?: FluxPrompt;
  numImagesPerPrompt?: number;
  clipMaxLength?: number;
  maxSequenceLength?: number;
  guidanceScale?: number;
  guidanceDType?: DType;
};

export type FluxPromptConditioning = Disposable & {
  readonly batchSize: number;
  readonly conditioning: FluxConditioning;
  readonly promptTruncated: boolean;
  readonly prompt2Truncated: boolean;
};

export type FluxPromptConditioner = Disposable & {
  readonly pipelineKind: "flux";
  encodePrompt(options: FluxPromptConditioningOptions): FluxPromptConditioning;
};

export type FluxCLIPTextEncoder = Disposable & {
  run(inputIds: MxArray, options?: CLIPTextModelOptions): CLIPTextModelOutput;
};

export type FluxT5TextEncoder = Disposable & {
  run(inputIds: MxArray, options?: T5EncoderModelOptions): T5EncoderModelOutput;
};

export type FluxPromptConditionerComponents = {
  tokenizer: CLIPTokenizer;
  textEncoder: FluxCLIPTextEncoder;
  tokenizer2: SentencePieceTokenizer;
  textEncoder2: FluxT5TextEncoder;
};

export type LoadFluxPromptConditionerOptions = {
  textEncoder?: LoadCLIPTextModelOptions;
  textEncoder2?: LoadT5EncoderModelOptions;
};
