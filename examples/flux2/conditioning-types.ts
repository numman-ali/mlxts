import type { MxArray } from "@mlxts/core";
import type { Flux2KleinConditioning } from "@mlxts/diffusion";
import type { Tokenizer } from "@mlxts/tokenizers";
import type {
  InteractionProfile,
  LoadCausalLMOptions,
  LoadPretrainedTokenizerOptions,
  LoadSourceOptions,
} from "@mlxts/transformers";

export type Flux2KleinPromptConditioningOptions = {
  prompt: string;
  negativePrompt?: string;
  guidanceScale?: number;
  maxSequenceLength?: number;
  textEncoderOutLayers?: readonly number[];
};

export type Flux2KleinPromptConditioning = Disposable & {
  readonly batchSize: number;
  readonly conditioning: Flux2KleinConditioning;
  readonly promptTruncated: boolean;
  readonly negativePromptTruncated: boolean;
};

export type Flux2KleinPromptConditioner = Disposable & {
  readonly pipelineKind: "flux2-klein";
  encodePrompt(options: Flux2KleinPromptConditioningOptions): Flux2KleinPromptConditioning;
};

export type Flux2KleinTextModelOutput = {
  lastHiddenState: MxArray;
  hiddenStates?: MxArray[];
};

export type Flux2KleinQwen3TextModel = {
  runWithHiddenStates(
    inputIds: MxArray,
    options?: { outputHiddenStates?: boolean },
  ): Flux2KleinTextModelOutput;
};

export type Flux2KleinQwen3TextEncoder = Disposable & {
  readonly model: Flux2KleinQwen3TextModel;
};

export type Flux2KleinPromptConditionerComponents = {
  tokenizer: Tokenizer;
  interactionProfile: InteractionProfile;
  textEncoder: Flux2KleinQwen3TextEncoder;
};

export type LoadFlux2KleinPromptConditionerOptions = {
  tokenizer?: LoadPretrainedTokenizerOptions;
  chatTemplate?: LoadSourceOptions;
  textEncoder?: LoadCausalLMOptions;
};
