import type { MxArray } from "@mlxts/core";
import type { StableDiffusion3Conditioning } from "@mlxts/diffusion";
import type { CLIPTokenizer, SentencePieceTokenizer } from "@mlxts/tokenizers";
import type {
  CLIPTextModelOptions,
  CLIPTextProjectionOutput,
  LoadCLIPTextModelOptions,
  LoadT5EncoderModelOptions,
  T5EncoderModelOptions,
  T5EncoderModelOutput,
} from "@mlxts/transformers";

export type StableDiffusion3Prompt = string | readonly string[];

export type StableDiffusion3PromptConditioningOptions = {
  prompt: StableDiffusion3Prompt;
  prompt2?: StableDiffusion3Prompt;
  prompt3?: StableDiffusion3Prompt;
  negativePrompt?: StableDiffusion3Prompt;
  negativePrompt2?: StableDiffusion3Prompt;
  negativePrompt3?: StableDiffusion3Prompt;
  guidanceScale?: number;
  numImagesPerPrompt?: number;
  clipMaxLength?: number;
  maxSequenceLength?: number;
  clipSkip?: number;
};

export type StableDiffusion3PromptConditioning = Disposable & {
  readonly batchSize: number;
  readonly conditioning: StableDiffusion3Conditioning;
  readonly negativeConditioning: StableDiffusion3Conditioning | undefined;
  readonly promptTruncated: boolean;
  readonly prompt2Truncated: boolean;
  readonly prompt3Truncated: boolean;
  readonly negativePromptTruncated: boolean;
  readonly negativePrompt2Truncated: boolean;
  readonly negativePrompt3Truncated: boolean;
};

export type StableDiffusion3PromptConditioner = Disposable & {
  readonly pipelineKind: "stable-diffusion-3";
  encodePrompt(
    options: StableDiffusion3PromptConditioningOptions,
  ): StableDiffusion3PromptConditioning;
};

export type StableDiffusion3CLIPTextEncoder = Disposable & {
  run(inputIds: MxArray, options?: CLIPTextModelOptions): CLIPTextProjectionOutput;
};

export type StableDiffusion3T5TextEncoder = Disposable & {
  run(inputIds: MxArray, options?: T5EncoderModelOptions): T5EncoderModelOutput;
};

export type StableDiffusion3PromptConditionerComponents = {
  tokenizer: CLIPTokenizer;
  tokenizer2: CLIPTokenizer;
  tokenizer3: SentencePieceTokenizer;
  textEncoder: StableDiffusion3CLIPTextEncoder;
  textEncoder2: StableDiffusion3CLIPTextEncoder;
  textEncoder3: StableDiffusion3T5TextEncoder;
  jointAttentionDim?: number;
  pooledProjectionDim?: number;
};

export type LoadStableDiffusion3PromptConditionerOptions = {
  textEncoder?: LoadCLIPTextModelOptions;
  textEncoder2?: LoadCLIPTextModelOptions;
  textEncoder3?: LoadT5EncoderModelOptions;
};
