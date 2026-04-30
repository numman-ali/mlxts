import { array, full, type MxArray, repeat } from "@mlxts/core";
import { createFluxTextIds, type FluxConditioning } from "@mlxts/diffusion";
import { encodeCLIPTextInput, encodeT5TextInput } from "@mlxts/tokenizers";
import type { CLIPTextModelOutput, T5EncoderModelOutput } from "@mlxts/transformers";

import { FluxPromptConditioningResult } from "./conditioning-result";
import type {
  FluxCLIPTextEncoder,
  FluxPrompt,
  FluxPromptConditionerComponents,
  FluxPromptConditioning,
  FluxPromptConditioningOptions,
  FluxT5TextEncoder,
} from "./conditioning-types";

type EncodedPrompts = {
  inputIds: MxArray;
  truncated: boolean;
};

type CLIPBranch = {
  pooledProjections: MxArray;
  truncated: boolean;
};

type T5Branch = {
  encoderHiddenStates: MxArray;
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

function disposeCLIPOutputExceptPooled(output: CLIPTextModelOutput): void {
  output.lastHiddenState.free();
  disposeHiddenStates(output.hiddenStates);
}

function disposeT5OutputExceptLastHidden(output: T5EncoderModelOutput): void {
  disposeHiddenStates(output.hiddenStates);
}

function promptBatch(value: FluxPrompt, name: string): readonly string[] {
  if (typeof value === "string") {
    return [value];
  }
  if (value.length === 0) {
    throw new Error(`${name} must contain at least one prompt.`);
  }
  return value;
}

function matchingPromptBatch(
  value: FluxPrompt | undefined,
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

function resolveMaxSequenceLength(value: number | undefined): number {
  const resolved = value ?? 512;
  if (!Number.isInteger(resolved) || resolved <= 0 || resolved > 512) {
    throw new Error("maxSequenceLength must be a positive integer no greater than 512.");
  }
  return resolved;
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
  tokenizer: FluxPromptConditionerComponents["tokenizer"],
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
  tokenizer: FluxPromptConditionerComponents["tokenizer2"],
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

function encodeCLIPBranch(
  tokenizer: FluxPromptConditionerComponents["tokenizer"],
  textEncoder: FluxCLIPTextEncoder,
  prompts: readonly string[],
  maxLength: number | undefined,
  numImagesPerPrompt: number,
): CLIPBranch {
  const encoded = encodeCLIPPromptIds(tokenizer, prompts, maxLength);
  try {
    const output = textEncoder.run(encoded.inputIds);
    try {
      return {
        pooledProjections: repeatPromptBatch(output.pooledOutput, numImagesPerPrompt),
        truncated: encoded.truncated,
      };
    } finally {
      disposeCLIPOutputExceptPooled(output);
    }
  } finally {
    encoded.inputIds.free();
  }
}

function encodeT5Branch(
  tokenizer: FluxPromptConditionerComponents["tokenizer2"],
  textEncoder: FluxT5TextEncoder,
  prompts: readonly string[],
  maxSequenceLength: number,
  numImagesPerPrompt: number,
): T5Branch {
  const encoded = encodeT5PromptIds(tokenizer, prompts, maxSequenceLength);
  try {
    const output = textEncoder.run(encoded.inputIds);
    try {
      return {
        encoderHiddenStates: repeatPromptBatch(output.lastHiddenState, numImagesPerPrompt),
        truncated: encoded.truncated,
      };
    } finally {
      disposeT5OutputExceptLastHidden(output);
    }
  } finally {
    encoded.inputIds.free();
  }
}

function createGuidance(
  options: FluxPromptConditioningOptions,
  batchSize: number,
): MxArray | undefined {
  if (options.guidanceScale === undefined) {
    return undefined;
  }
  return full([batchSize], options.guidanceScale, options.guidanceDType ?? "float32");
}

export function encodeFluxPrompt(
  components: FluxPromptConditionerComponents,
  options: FluxPromptConditioningOptions,
): FluxPromptConditioning {
  const prompts = promptBatch(options.prompt, "prompt");
  const prompts2 = matchingPromptBatch(options.prompt2, prompts, "prompt2");
  const numImagesPerPrompt = resolveNumImagesPerPrompt(options.numImagesPerPrompt);
  const batchSize = prompts.length * numImagesPerPrompt;
  const clip = encodeCLIPBranch(
    components.tokenizer,
    components.textEncoder,
    prompts,
    options.clipMaxLength,
    numImagesPerPrompt,
  );
  try {
    const t5 = encodeT5Branch(
      components.tokenizer2,
      components.textEncoder2,
      prompts2,
      resolveMaxSequenceLength(options.maxSequenceLength),
      numImagesPerPrompt,
    );
    let textIds: MxArray | undefined;
    let guidance: MxArray | undefined;
    try {
      const textLength = t5.encoderHiddenStates.shape[1];
      if (textLength === undefined) {
        throw new Error("FLUX T5 conditioning must have a visible sequence length.");
      }
      textIds = createFluxTextIds(textLength);
      guidance = createGuidance(options, batchSize);
      const conditioning: FluxConditioning = {
        encoderHiddenStates: t5.encoderHiddenStates,
        pooledProjections: clip.pooledProjections,
        textIds,
      };
      if (guidance !== undefined) {
        conditioning.guidance = guidance;
      }
      return new FluxPromptConditioningResult(
        batchSize,
        conditioning,
        clip.truncated,
        t5.truncated,
      );
    } catch (error) {
      t5.encoderHiddenStates.free();
      textIds?.free();
      guidance?.free();
      throw error;
    }
  } catch (error) {
    clip.pooledProjections.free();
    throw error;
  }
}
