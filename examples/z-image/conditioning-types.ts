import type { MxArray } from "@mlxts/core";
import type { ZImageConditioning } from "@mlxts/diffusion";
import type { Tokenizer } from "@mlxts/tokenizers";
import type {
  InteractionProfile,
  LoadCausalLMOptions,
  LoadPretrainedTokenizerOptions,
  LoadSourceOptions,
} from "@mlxts/transformers";

export type ZImagePromptConditioningOptions = {
  prompt: string;
  maxSequenceLength?: number;
};

export type ZImagePromptConditioning = Disposable & {
  readonly batchSize: number;
  readonly conditioning: ZImageConditioning;
  readonly promptTruncated: boolean;
};

export type ZImagePromptConditioner = Disposable & {
  readonly pipelineKind: "z-image";
  encodePrompt(options: ZImagePromptConditioningOptions): ZImagePromptConditioning;
};

export type ZImageTextModelOutput = {
  lastHiddenState: MxArray;
  hiddenStates?: MxArray[];
};

export type ZImageQwen3TextModel = {
  runWithHiddenStates(
    inputIds: MxArray,
    options?: { outputHiddenStates?: boolean },
  ): ZImageTextModelOutput;
};

export type ZImageQwen3TextEncoder = Disposable & {
  readonly model: ZImageQwen3TextModel;
};

export type ZImagePromptConditionerComponents = {
  tokenizer: Tokenizer;
  interactionProfile: InteractionProfile;
  textEncoder: ZImageQwen3TextEncoder;
};

export type LoadZImagePromptConditionerOptions = {
  tokenizer?: LoadPretrainedTokenizerOptions;
  chatTemplate?: LoadSourceOptions;
  textEncoder?: LoadCausalLMOptions;
};
