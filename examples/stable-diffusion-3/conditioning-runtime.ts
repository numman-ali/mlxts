import {
  array,
  asType,
  concatenate,
  formatShape,
  type MxArray,
  repeat,
  retainArray,
  zeros,
} from "@mlxts/core";
import type { StableDiffusion3Conditioning } from "@mlxts/diffusion";
import { encodeCLIPTextInput, encodeT5TextInput } from "@mlxts/tokenizers";
import type { CLIPTextProjectionOutput, T5EncoderModelOutput } from "@mlxts/transformers";

import { StableDiffusion3PromptConditioningResult } from "./conditioning-result";
import type {
  StableDiffusion3CLIPTextEncoder,
  StableDiffusion3Prompt,
  StableDiffusion3PromptConditionerComponents,
  StableDiffusion3PromptConditioning,
  StableDiffusion3PromptConditioningOptions,
  StableDiffusion3T5TextEncoder,
} from "./conditioning-types";

type EncodedPrompts = {
  inputIds: MxArray;
  truncated: boolean;
};

type CLIPBranch = {
  hiddenState: MxArray;
  pooledProjection: MxArray;
  truncated: boolean;
};

type T5Branch = {
  hiddenState: MxArray;
  truncated: boolean;
};

type ConditioningBranch = {
  conditioning: StableDiffusion3Conditioning;
  promptTruncated: boolean;
  prompt2Truncated: boolean;
  prompt3Truncated: boolean;
};

type SequenceShape = {
  batch: number;
  sequenceLength: number;
  channels: number;
};

export const SD3_DEFAULT_MAX_SEQUENCE_LENGTH = 256;
export const SD3_MAX_SEQUENCE_LENGTH = 512;

function disposeHiddenStates(hiddenStates: MxArray[] | undefined): void {
  if (hiddenStates === undefined) {
    return;
  }
  for (const hiddenState of hiddenStates) {
    hiddenState.free();
  }
}

function disposeCLIPOutputExceptTextEmbeds(output: CLIPTextProjectionOutput): void {
  output.lastHiddenState.free();
  output.pooledOutput.free();
  disposeHiddenStates(output.hiddenStates);
}

function disposeT5OutputExceptLastHidden(output: T5EncoderModelOutput): void {
  disposeHiddenStates(output.hiddenStates);
}

function promptBatch(value: StableDiffusion3Prompt, name: string): readonly string[] {
  if (typeof value === "string") {
    return [value];
  }
  if (value.length === 0) {
    throw new Error(`${name} must contain at least one prompt.`);
  }
  return value;
}

function matchingPromptBatch(
  value: StableDiffusion3Prompt | undefined,
  fallback: readonly string[],
  name: string,
  options: { emptyStringFallsBack?: boolean } = {},
): readonly string[] {
  if (value === undefined) {
    return fallback;
  }
  if (typeof value === "string") {
    if (options.emptyStringFallsBack === true && value === "") {
      return fallback;
    }
    return Array.from({ length: fallback.length }, () => value);
  }
  if (value.length !== fallback.length) {
    throw new Error(`${name} batch size must match prompt batch size.`);
  }
  return value;
}

function resolveNumImagesPerPrompt(value: number | undefined): number {
  const resolved = value ?? 1;
  if (!Number.isInteger(resolved) || resolved <= 0) {
    throw new Error("numImagesPerPrompt must be a positive integer.");
  }
  return resolved;
}

function resolveMaxSequenceLength(value: number | undefined): number {
  const resolved = value ?? SD3_DEFAULT_MAX_SEQUENCE_LENGTH;
  if (!Number.isInteger(resolved) || resolved <= 0 || resolved > SD3_MAX_SEQUENCE_LENGTH) {
    throw new Error(
      `maxSequenceLength must be a positive integer no greater than ${SD3_MAX_SEQUENCE_LENGTH}.`,
    );
  }
  return resolved;
}

function resolveClipSkip(value: number | undefined): number {
  const resolved = value ?? 0;
  if (!Number.isInteger(resolved) || resolved < 0) {
    throw new Error("clipSkip must be a non-negative integer.");
  }
  return resolved;
}

function usesClassifierFreeGuidance(guidanceScale: number | undefined): boolean {
  const resolved = guidanceScale ?? 1;
  if (!Number.isFinite(resolved) || resolved < 0) {
    throw new Error("guidanceScale must be a finite non-negative number.");
  }
  return resolved > 1;
}

function repeatPromptBatch(value: MxArray, numImagesPerPrompt: number): MxArray {
  if (numImagesPerPrompt === 1) {
    return value;
  }
  try {
    return repeat(value, numImagesPerPrompt, 0);
  } finally {
    value.free();
  }
}

function encodeCLIPPromptIds(
  tokenizer: StableDiffusion3PromptConditionerComponents["tokenizer"],
  prompts: readonly string[],
  maxLength: number | undefined,
): EncodedPrompts {
  const rows: number[][] = [];
  let truncated = false;
  const encodeOptions = maxLength === undefined ? {} : { maxLength };
  for (const prompt of prompts) {
    const encoded = encodeCLIPTextInput(tokenizer, prompt, encodeOptions);
    rows.push(encoded.inputIds);
    truncated = truncated || encoded.truncated;
  }
  return {
    inputIds: array(rows, "int32"),
    truncated,
  };
}

function encodeT5PromptIds(
  tokenizer: StableDiffusion3PromptConditionerComponents["tokenizer3"],
  prompts: readonly string[],
  maxSequenceLength: number,
): EncodedPrompts {
  const rows: number[][] = [];
  let truncated = false;
  for (const prompt of prompts) {
    const encoded = encodeT5TextInput(tokenizer, prompt, { maxLength: maxSequenceLength });
    rows.push(encoded.inputIds);
    truncated = truncated || encoded.truncated;
  }
  return {
    inputIds: array(rows, "int32"),
    truncated,
  };
}

function selectCLIPHiddenState(output: CLIPTextProjectionOutput, clipSkip: number): MxArray {
  const hiddenStates = output.hiddenStates;
  const index = hiddenStates === undefined ? -1 : hiddenStates.length - (clipSkip + 2);
  if (hiddenStates === undefined || index < 0) {
    throw new Error("SD3 prompt conditioning requires enough CLIP hidden states.");
  }
  const hiddenState = hiddenStates[index];
  if (hiddenState === undefined) {
    throw new Error("SD3 prompt conditioning could not read the requested CLIP hidden state.");
  }
  return retainArray(hiddenState);
}

function encodeCLIPBranch(
  tokenizer: StableDiffusion3PromptConditionerComponents["tokenizer"],
  textEncoder: StableDiffusion3CLIPTextEncoder,
  prompts: readonly string[],
  maxLength: number | undefined,
  numImagesPerPrompt: number,
  clipSkip: number,
): CLIPBranch {
  const encoded = encodeCLIPPromptIds(tokenizer, prompts, maxLength);
  try {
    const output = textEncoder.run(encoded.inputIds, { outputHiddenStates: true });
    let hiddenState: MxArray | undefined;
    let pooledProjection: MxArray | undefined = output.textEmbeds;
    try {
      hiddenState = selectCLIPHiddenState(output, clipSkip);
      const repeatedHidden = repeatPromptBatch(hiddenState, numImagesPerPrompt);
      hiddenState = undefined;
      const repeatedPooled = repeatPromptBatch(pooledProjection, numImagesPerPrompt);
      pooledProjection = undefined;
      return {
        hiddenState: repeatedHidden,
        pooledProjection: repeatedPooled,
        truncated: encoded.truncated,
      };
    } finally {
      hiddenState?.free();
      pooledProjection?.free();
      disposeCLIPOutputExceptTextEmbeds(output);
    }
  } finally {
    encoded.inputIds.free();
  }
}

function encodeT5Branch(
  tokenizer: StableDiffusion3PromptConditionerComponents["tokenizer3"],
  textEncoder: StableDiffusion3T5TextEncoder,
  prompts: readonly string[],
  maxSequenceLength: number,
  numImagesPerPrompt: number,
): T5Branch {
  const encoded = encodeT5PromptIds(tokenizer, prompts, maxSequenceLength);
  try {
    const output = textEncoder.run(encoded.inputIds);
    let hiddenState: MxArray | undefined = output.lastHiddenState;
    try {
      const repeatedHidden = repeatPromptBatch(hiddenState, numImagesPerPrompt);
      hiddenState = undefined;
      return {
        hiddenState: repeatedHidden,
        truncated: encoded.truncated,
      };
    } finally {
      hiddenState?.free();
      disposeT5OutputExceptLastHidden(output);
    }
  } finally {
    encoded.inputIds.free();
  }
}

function sequenceShape(tensor: MxArray, name: string): SequenceShape {
  const [batch, sequenceLength, channels] = tensor.shape;
  if (
    tensor.shape.length !== 3 ||
    batch === undefined ||
    sequenceLength === undefined ||
    channels === undefined
  ) {
    throw new Error(`${name} must be [batch, sequence, hidden], got ${formatShape(tensor.shape)}.`);
  }
  return { batch, sequenceLength, channels };
}

function paddedCLIPHidden(clipHidden: MxArray, targetChannels: number): MxArray {
  const shape = sequenceShape(clipHidden, "SD3 CLIP prompt embeds");
  const paddingChannels = targetChannels - shape.channels;
  if (paddingChannels < 0) {
    throw new Error("SD3 CLIP hidden size cannot exceed T5 hidden size.");
  }
  if (paddingChannels === 0) {
    return retainArray(clipHidden);
  }
  using padding = zeros([shape.batch, shape.sequenceLength, paddingChannels], clipHidden.dtype);
  return concatenate([clipHidden, padding], -1);
}

function assertPooledShape(tensor: MxArray, expectedChannels: number | undefined): void {
  const [batch, channels] = tensor.shape;
  if (tensor.shape.length !== 2 || batch === undefined || channels === undefined) {
    throw new Error(
      `SD3 pooled projections must be [batch, hidden], got ${formatShape(tensor.shape)}.`,
    );
  }
  if (expectedChannels !== undefined && channels !== expectedChannels) {
    throw new Error(
      `SD3 pooled projections must have hidden size ${expectedChannels}, got ${channels}.`,
    );
  }
}

function composeBranch(
  components: StableDiffusion3PromptConditionerComponents,
  first: CLIPBranch,
  second: CLIPBranch,
  t5: T5Branch,
): ConditioningBranch {
  let encoderHiddenStates: MxArray | undefined;
  let pooledProjections: MxArray | undefined;
  try {
    using clipHidden = concatenate([first.hiddenState, second.hiddenState], -1);
    const clipShape = sequenceShape(clipHidden, "SD3 combined CLIP prompt embeds");
    const t5Shape = sequenceShape(t5.hiddenState, "SD3 T5 prompt embeds");
    if (clipShape.batch !== t5Shape.batch) {
      throw new Error("SD3 CLIP and T5 prompt batches must match.");
    }
    if (
      components.jointAttentionDim !== undefined &&
      t5Shape.channels !== components.jointAttentionDim
    ) {
      throw new Error(
        `SD3 T5 hidden size must match jointAttentionDim ${components.jointAttentionDim}, got ${t5Shape.channels}.`,
      );
    }
    using clipHiddenForT5 =
      clipHidden.dtype === t5.hiddenState.dtype
        ? retainArray(clipHidden)
        : asType(clipHidden, t5.hiddenState.dtype);
    using paddedClip = paddedCLIPHidden(clipHiddenForT5, t5Shape.channels);
    encoderHiddenStates = concatenate([paddedClip, t5.hiddenState], 1);
    pooledProjections = concatenate([first.pooledProjection, second.pooledProjection], -1);
    assertPooledShape(pooledProjections, components.pooledProjectionDim);
    const conditioning = {
      encoderHiddenStates,
      pooledProjections,
    };
    encoderHiddenStates = undefined;
    pooledProjections = undefined;
    return {
      conditioning,
      promptTruncated: first.truncated,
      prompt2Truncated: second.truncated,
      prompt3Truncated: t5.truncated,
    };
  } finally {
    encoderHiddenStates?.free();
    pooledProjections?.free();
  }
}

function disposeCLIPBranch(branch: CLIPBranch): void {
  branch.hiddenState.free();
  branch.pooledProjection.free();
}

function encodeBranch(
  components: StableDiffusion3PromptConditionerComponents,
  prompts: readonly string[],
  prompts2: readonly string[],
  prompts3: readonly string[],
  options: StableDiffusion3PromptConditioningOptions,
  numImagesPerPrompt: number,
  clipSkip: number,
): ConditioningBranch {
  const first = encodeCLIPBranch(
    components.tokenizer,
    components.textEncoder,
    prompts,
    options.clipMaxLength,
    numImagesPerPrompt,
    clipSkip,
  );
  try {
    const second = encodeCLIPBranch(
      components.tokenizer2,
      components.textEncoder2,
      prompts2,
      options.clipMaxLength,
      numImagesPerPrompt,
      clipSkip,
    );
    try {
      const t5 = encodeT5Branch(
        components.tokenizer3,
        components.textEncoder3,
        prompts3,
        resolveMaxSequenceLength(options.maxSequenceLength),
        numImagesPerPrompt,
      );
      try {
        return composeBranch(components, first, second, t5);
      } finally {
        t5.hiddenState.free();
      }
    } finally {
      disposeCLIPBranch(second);
    }
  } finally {
    disposeCLIPBranch(first);
  }
}

export function encodeStableDiffusion3Prompt(
  components: StableDiffusion3PromptConditionerComponents,
  options: StableDiffusion3PromptConditioningOptions,
): StableDiffusion3PromptConditioning {
  const prompts = promptBatch(options.prompt, "prompt");
  const prompts2 = matchingPromptBatch(options.prompt2, prompts, "prompt2", {
    emptyStringFallsBack: true,
  });
  const prompts3 = matchingPromptBatch(options.prompt3, prompts, "prompt3", {
    emptyStringFallsBack: true,
  });
  const numImagesPerPrompt = resolveNumImagesPerPrompt(options.numImagesPerPrompt);
  const clipSkip = resolveClipSkip(options.clipSkip);
  const positive = encodeBranch(
    components,
    prompts,
    prompts2,
    prompts3,
    options,
    numImagesPerPrompt,
    clipSkip,
  );
  try {
    if (!usesClassifierFreeGuidance(options.guidanceScale)) {
      return new StableDiffusion3PromptConditioningResult(
        prompts.length * numImagesPerPrompt,
        positive.conditioning,
        undefined,
        positive.promptTruncated,
        positive.prompt2Truncated,
        positive.prompt3Truncated,
        false,
        false,
        false,
      );
    }
    const negativePrompts = matchingPromptBatch(
      options.negativePrompt ?? "",
      prompts,
      "negativePrompt",
    );
    const negativePrompts2 = matchingPromptBatch(
      options.negativePrompt2,
      negativePrompts,
      "negativePrompt2",
      { emptyStringFallsBack: true },
    );
    const negativePrompts3 = matchingPromptBatch(
      options.negativePrompt3,
      negativePrompts,
      "negativePrompt3",
      { emptyStringFallsBack: true },
    );
    const negative = encodeBranch(
      components,
      negativePrompts,
      negativePrompts2,
      negativePrompts3,
      options,
      numImagesPerPrompt,
      0,
    );
    return new StableDiffusion3PromptConditioningResult(
      prompts.length * numImagesPerPrompt,
      positive.conditioning,
      negative.conditioning,
      positive.promptTruncated,
      positive.prompt2Truncated,
      positive.prompt3Truncated,
      negative.promptTruncated,
      negative.prompt2Truncated,
      negative.prompt3Truncated,
    );
  } catch (error) {
    positive.conditioning.encoderHiddenStates.free();
    positive.conditioning.pooledProjections.free();
    throw error;
  }
}
