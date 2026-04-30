import {
  array,
  concatenate,
  type DType,
  type MxArray,
  repeat,
  retainArray,
  zeros,
} from "@mlxts/core";
import type { StableDiffusionConditioning } from "@mlxts/diffusion";
import { encodeCLIPTextInput } from "@mlxts/tokenizers";
import type { CLIPTextModelOutput, CLIPTextProjectionOutput } from "@mlxts/transformers";

import { disposeConditioning, PromptConditioningResult } from "./conditioning-result";
import type {
  StableDiffusionProjectedTextEncoder,
  StableDiffusionPrompt,
  StableDiffusionPromptConditionerComponents,
  StableDiffusionPromptConditioning,
  StableDiffusionPromptConditioningOptions,
  StableDiffusionTextEncoder,
} from "./conditioning-types";

type EncodedPrompts = {
  inputIds: MxArray;
  truncated: boolean;
};

type XLGeometry = {
  targetSize: readonly [number, number];
  originalSize: readonly [number, number];
  cropTopLeft: readonly [number, number];
  dtype: DType;
};

type XLBranch = {
  conditioning: StableDiffusionConditioning;
  truncated: boolean;
};

function disposeHiddenStates(hiddenStates: MxArray[] | undefined): void {
  if (hiddenStates === undefined) {
    return;
  }
  for (const hiddenState of hiddenStates) {
    hiddenState.free();
  }
}

function disposeCLIPOutput(output: CLIPTextModelOutput): void {
  output.lastHiddenState.free();
  output.pooledOutput.free();
  disposeHiddenStates(output.hiddenStates);
}

function disposeCLIPOutputExceptLastHidden(output: CLIPTextModelOutput): void {
  output.pooledOutput.free();
  disposeHiddenStates(output.hiddenStates);
}

function disposeCLIPProjectionOutputExceptTextEmbeds(output: CLIPTextProjectionOutput): void {
  disposeCLIPOutput(output);
}

function promptBatch(value: StableDiffusionPrompt, name: string): readonly string[] {
  if (typeof value === "string") {
    return [value];
  }
  if (value.length === 0) {
    throw new Error(`${name} must contain at least one prompt.`);
  }
  return value;
}

function matchingPromptBatch(
  value: StableDiffusionPrompt | undefined,
  fallback: readonly string[],
  name: string,
): readonly string[] {
  if (value === undefined) {
    return fallback;
  }
  if (typeof value === "string") {
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

function usesClassifierFreeGuidance(guidanceScale: number | undefined): boolean {
  return (guidanceScale ?? 1) > 1;
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

function encodePromptIds(
  tokenizer: StableDiffusionPromptConditionerComponents["tokenizer"],
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

function encodeSDBranch(
  tokenizer: StableDiffusionPromptConditionerComponents["tokenizer"],
  textEncoder: StableDiffusionTextEncoder,
  prompts: readonly string[],
  maxLength: number | undefined,
  numImagesPerPrompt: number,
): { conditioning: StableDiffusionConditioning; truncated: boolean } {
  const encoded = encodePromptIds(tokenizer, prompts, maxLength);
  try {
    const output = textEncoder.run(encoded.inputIds);
    try {
      return {
        conditioning: {
          encoderHiddenStates: repeatPromptBatch(output.lastHiddenState, numImagesPerPrompt),
        },
        truncated: encoded.truncated,
      };
    } finally {
      disposeCLIPOutputExceptLastHidden(output);
    }
  } finally {
    encoded.inputIds.free();
  }
}

function requirePenultimateHiddenState(output: CLIPTextModelOutput): MxArray {
  const hiddenStates = output.hiddenStates;
  if (hiddenStates === undefined || hiddenStates.length < 2) {
    throw new Error("SDXL prompt conditioning requires CLIP hidden states.");
  }
  const hiddenState = hiddenStates[hiddenStates.length - 2];
  if (hiddenState === undefined) {
    throw new Error("SDXL prompt conditioning could not read penultimate CLIP hidden state.");
  }
  return retainArray(hiddenState);
}

function encodeXLHiddenBranch(
  tokenizer: StableDiffusionPromptConditionerComponents["tokenizer"],
  textEncoder: StableDiffusionTextEncoder,
  prompts: readonly string[],
  maxLength: number | undefined,
): { hiddenState: MxArray; truncated: boolean } {
  const encoded = encodePromptIds(tokenizer, prompts, maxLength);
  try {
    const output = textEncoder.run(encoded.inputIds, { outputHiddenStates: true });
    try {
      return {
        hiddenState: requirePenultimateHiddenState(output),
        truncated: encoded.truncated,
      };
    } finally {
      disposeCLIPOutput(output);
    }
  } finally {
    encoded.inputIds.free();
  }
}

function encodeXLProjectedBranch(
  tokenizer: StableDiffusionPromptConditionerComponents["tokenizer"],
  textEncoder: StableDiffusionProjectedTextEncoder,
  prompts: readonly string[],
  maxLength: number | undefined,
): { hiddenState: MxArray; textEmbeds: MxArray; truncated: boolean } {
  const encoded = encodePromptIds(tokenizer, prompts, maxLength);
  try {
    const output = textEncoder.run(encoded.inputIds, { outputHiddenStates: true });
    let hiddenState: MxArray | undefined;
    try {
      hiddenState = requirePenultimateHiddenState(output);
      return {
        hiddenState,
        textEmbeds: output.textEmbeds,
        truncated: encoded.truncated,
      };
    } catch (error) {
      hiddenState?.free();
      output.textEmbeds.free();
      throw error;
    } finally {
      disposeCLIPProjectionOutputExceptTextEmbeds(output);
    }
  } finally {
    encoded.inputIds.free();
  }
}

function assertPositivePair(name: string, value: readonly [number, number]): void {
  for (const dimension of value) {
    if (!Number.isInteger(dimension) || dimension <= 0) {
      throw new Error(`${name} dimensions must be positive integers.`);
    }
  }
}

function assertNonNegativePair(name: string, value: readonly [number, number]): void {
  for (const dimension of value) {
    if (!Number.isInteger(dimension) || dimension < 0) {
      throw new Error(`${name} dimensions must be non-negative integers.`);
    }
  }
}

function resolveXLGeometry(
  options: StableDiffusionPromptConditioningOptions,
  negative: boolean,
): XLGeometry {
  const targetSize = negative
    ? (options.negativeTargetSize ?? options.targetSize)
    : options.targetSize;
  if (targetSize === undefined) {
    throw new Error("targetSize is required for SDXL prompt conditioning.");
  }
  const originalSize = negative
    ? (options.negativeOriginalSize ?? options.originalSize ?? targetSize)
    : (options.originalSize ?? targetSize);
  const cropTopLeft = negative
    ? (options.negativeCropTopLeft ?? options.cropTopLeft ?? [0, 0])
    : (options.cropTopLeft ?? [0, 0]);
  assertPositivePair("targetSize", targetSize);
  assertPositivePair("originalSize", originalSize);
  assertNonNegativePair("cropTopLeft", cropTopLeft);
  return {
    targetSize,
    originalSize,
    cropTopLeft,
    dtype: options.timeIdsDType ?? "float32",
  };
}

function createXLTimeIds(
  batchSize: number,
  geometry: XLGeometry,
  numImagesPerPrompt: number,
): MxArray {
  const row = [...geometry.originalSize, ...geometry.cropTopLeft, ...geometry.targetSize];
  const rows = Array.from({ length: batchSize }, () => row);
  return repeatPromptBatch(array(rows, geometry.dtype), numImagesPerPrompt);
}

function zeroConditioningLike(
  conditioning: StableDiffusionConditioning,
  timeIds: MxArray | undefined,
): StableDiffusionConditioning {
  const encoderHiddenStates = zeros(
    [...conditioning.encoderHiddenStates.shape],
    conditioning.encoderHiddenStates.dtype,
  );
  if (conditioning.textTime === undefined) {
    return { encoderHiddenStates };
  }
  const textEmbeds = zeros(
    [...conditioning.textTime.textEmbeds.shape],
    conditioning.textTime.textEmbeds.dtype,
  );
  return {
    encoderHiddenStates,
    textTime: {
      textEmbeds,
      timeIds: timeIds ?? retainArray(conditioning.textTime.timeIds),
    },
  };
}

function requireXLComponents(components: StableDiffusionPromptConditionerComponents): {
  tokenizer2: StableDiffusionPromptConditionerComponents["tokenizer"];
  textEncoder2: StableDiffusionProjectedTextEncoder;
} {
  if (components.tokenizer2 === undefined || components.textEncoder2 === undefined) {
    throw new Error("SDXL prompt conditioning requires tokenizer_2 and text_encoder_2.");
  }
  return {
    tokenizer2: components.tokenizer2,
    textEncoder2: components.textEncoder2,
  };
}

function encodeXLBranch(
  components: StableDiffusionPromptConditionerComponents,
  options: StableDiffusionPromptConditioningOptions,
  prompts: readonly string[],
  prompts2: readonly string[],
  negative: boolean,
): XLBranch {
  const xl = requireXLComponents(components);
  const numImagesPerPrompt = resolveNumImagesPerPrompt(options.numImagesPerPrompt);
  const first = encodeXLHiddenBranch(
    components.tokenizer,
    components.textEncoder,
    prompts,
    options.maxLength,
  );
  try {
    const second = encodeXLProjectedBranch(
      xl.tokenizer2,
      xl.textEncoder2,
      prompts2,
      options.maxLength2,
    );
    let encoderHiddenStates: MxArray | undefined;
    let textEmbeds: MxArray | undefined;
    let timeIds: MxArray | undefined;
    try {
      const hiddenStates = concatenate([first.hiddenState, second.hiddenState], -1);
      encoderHiddenStates = repeatPromptBatch(hiddenStates, numImagesPerPrompt);
      textEmbeds = repeatPromptBatch(second.textEmbeds, numImagesPerPrompt);
      timeIds = createXLTimeIds(
        prompts.length,
        resolveXLGeometry(options, negative),
        numImagesPerPrompt,
      );
      return {
        conditioning: {
          encoderHiddenStates,
          textTime: { textEmbeds, timeIds },
        },
        truncated: first.truncated || second.truncated,
      };
    } catch (error) {
      encoderHiddenStates?.free();
      textEmbeds?.free();
      timeIds?.free();
      throw error;
    } finally {
      second.hiddenState.free();
    }
  } finally {
    first.hiddenState.free();
  }
}

export function encodeSDPrompt(
  components: StableDiffusionPromptConditionerComponents,
  options: StableDiffusionPromptConditioningOptions,
): StableDiffusionPromptConditioning {
  const prompts = promptBatch(options.prompt, "prompt");
  const numImagesPerPrompt = resolveNumImagesPerPrompt(options.numImagesPerPrompt);
  const positive = encodeSDBranch(
    components.tokenizer,
    components.textEncoder,
    prompts,
    options.maxLength,
    numImagesPerPrompt,
  );
  try {
    if (
      !usesClassifierFreeGuidance(options.guidanceScale) &&
      options.negativePrompt === undefined
    ) {
      return new PromptConditioningResult(
        prompts.length * numImagesPerPrompt,
        positive.conditioning,
        undefined,
        positive.truncated,
        false,
      );
    }
    const negativePrompts = matchingPromptBatch(
      options.negativePrompt ?? "",
      prompts,
      "negativePrompt",
    );
    const negative = encodeSDBranch(
      components.tokenizer,
      components.textEncoder,
      negativePrompts,
      options.maxLength,
      numImagesPerPrompt,
    );
    return new PromptConditioningResult(
      prompts.length * numImagesPerPrompt,
      positive.conditioning,
      negative.conditioning,
      positive.truncated,
      negative.truncated,
    );
  } catch (error) {
    disposeConditioning(positive.conditioning);
    throw error;
  }
}

export function encodeXLPrompt(
  components: StableDiffusionPromptConditionerComponents,
  options: StableDiffusionPromptConditioningOptions,
): StableDiffusionPromptConditioning {
  const prompts = promptBatch(options.prompt, "prompt");
  const prompts2 = matchingPromptBatch(options.prompt2, prompts, "prompt2");
  const numImagesPerPrompt = resolveNumImagesPerPrompt(options.numImagesPerPrompt);
  const positive = encodeXLBranch(components, options, prompts, prompts2, false);
  try {
    if (
      !usesClassifierFreeGuidance(options.guidanceScale) &&
      options.negativePrompt === undefined
    ) {
      return new PromptConditioningResult(
        prompts.length * numImagesPerPrompt,
        positive.conditioning,
        undefined,
        positive.truncated,
        false,
      );
    }
    if (
      components.forceZerosForEmptyPrompt === true &&
      options.negativePrompt === undefined &&
      options.negativePrompt2 === undefined
    ) {
      const negativeTimeIds = createXLTimeIds(
        prompts.length,
        resolveXLGeometry(options, true),
        numImagesPerPrompt,
      );
      const negativeConditioning = zeroConditioningLike(positive.conditioning, negativeTimeIds);
      return new PromptConditioningResult(
        prompts.length * numImagesPerPrompt,
        positive.conditioning,
        negativeConditioning,
        positive.truncated,
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
    );
    const negative = encodeXLBranch(components, options, negativePrompts, negativePrompts2, true);
    return new PromptConditioningResult(
      prompts.length * numImagesPerPrompt,
      positive.conditioning,
      negative.conditioning,
      positive.truncated,
      negative.truncated,
    );
  } catch (error) {
    disposeConditioning(positive.conditioning);
    throw error;
  }
}
