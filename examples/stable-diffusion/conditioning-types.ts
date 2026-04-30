import type { DType, MxArray } from "@mlxts/core";
import type { DiffusionPipelineKind, StableDiffusionConditioning } from "@mlxts/diffusion";
import type { CLIPTokenizer } from "@mlxts/tokenizers";
import type {
  CLIPTextModelOptions,
  CLIPTextModelOutput,
  CLIPTextProjectionOutput,
  LoadCLIPTextModelOptions,
} from "@mlxts/transformers";

export type StableDiffusionPrompt = string | readonly string[];

export type StableDiffusionPromptConditioningOptions = {
  prompt: StableDiffusionPrompt;
  prompt2?: StableDiffusionPrompt;
  negativePrompt?: StableDiffusionPrompt;
  negativePrompt2?: StableDiffusionPrompt;
  guidanceScale?: number;
  numImagesPerPrompt?: number;
  maxLength?: number;
  maxLength2?: number;
  targetSize?: readonly [height: number, width: number];
  originalSize?: readonly [height: number, width: number];
  cropTopLeft?: readonly [top: number, left: number];
  negativeOriginalSize?: readonly [height: number, width: number];
  negativeTargetSize?: readonly [height: number, width: number];
  negativeCropTopLeft?: readonly [top: number, left: number];
  timeIdsDType?: DType;
};

export type StableDiffusionPromptConditioning = Disposable & {
  readonly batchSize: number;
  readonly conditioning: StableDiffusionConditioning;
  readonly negativeConditioning: StableDiffusionConditioning | undefined;
  readonly promptTruncated: boolean;
  readonly negativePromptTruncated: boolean;
};

export type StableDiffusionPromptConditioner = Disposable & {
  readonly pipelineKind: "stable-diffusion" | "stable-diffusion-xl";
  encodePrompt(
    options: StableDiffusionPromptConditioningOptions,
  ): StableDiffusionPromptConditioning;
};

export type StableDiffusionTextEncoder = Disposable & {
  run(inputIds: MxArray, options?: CLIPTextModelOptions): CLIPTextModelOutput;
};

export type StableDiffusionProjectedTextEncoder = Disposable & {
  run(inputIds: MxArray, options?: CLIPTextModelOptions): CLIPTextProjectionOutput;
};

export type StableDiffusionPromptConditionerComponents = {
  pipelineKind: Extract<DiffusionPipelineKind, "stable-diffusion" | "stable-diffusion-xl">;
  tokenizer: CLIPTokenizer;
  textEncoder: StableDiffusionTextEncoder;
  tokenizer2?: CLIPTokenizer;
  textEncoder2?: StableDiffusionProjectedTextEncoder;
  forceZerosForEmptyPrompt?: boolean;
};

export type LoadStableDiffusionPromptConditionerOptions = {
  textEncoder?: LoadCLIPTextModelOptions;
  textEncoder2?: LoadCLIPTextModelOptions;
};
