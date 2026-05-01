import { array, type MxArray, repeat } from "@mlxts/core";
import { encodeT5TextInput } from "@mlxts/tokenizers";
import type { T5EncoderModelOutput } from "@mlxts/transformers";

import { LtxVideoPromptConditioningResult } from "./conditioning-result";
import type {
  LtxVideoPrompt,
  LtxVideoPromptConditionerComponents,
  LtxVideoPromptConditioning,
  LtxVideoPromptConditioningOptions,
  LtxVideoT5TextEncoder,
} from "./conditioning-types";

type EncodedPrompts = {
  inputIds: MxArray;
  attentionMask: MxArray;
  truncated: boolean;
};

type T5Branch = {
  encoderHiddenStates: MxArray;
  attentionMask: MxArray;
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

function disposeT5OutputExceptLastHidden(output: T5EncoderModelOutput): void {
  disposeHiddenStates(output.hiddenStates);
}

function promptBatch(value: LtxVideoPrompt, name: string): readonly string[] {
  if (typeof value === "string") {
    return [value];
  }
  if (value.length === 0) {
    throw new Error(`${name} must contain at least one prompt.`);
  }
  return value;
}

function matchingPromptBatch(
  value: LtxVideoPrompt | undefined,
  fallback: readonly string[],
  name: string,
): readonly string[] {
  if (value === undefined) {
    return Array.from({ length: fallback.length }, () => "");
  }
  if (typeof value === "string") {
    return Array.from({ length: fallback.length }, () => value);
  }
  if (value.length !== fallback.length) {
    throw new Error(`${name} batch size must match prompt batch size.`);
  }
  return value;
}

function resolveNumVideosPerPrompt(value: number | undefined): number {
  const resolved = value ?? 1;
  if (!Number.isInteger(resolved) || resolved <= 0) {
    throw new Error("numVideosPerPrompt must be a positive integer.");
  }
  return resolved;
}

function resolveMaxSequenceLength(value: number | undefined): number {
  const resolved = value ?? 128;
  if (!Number.isInteger(resolved) || resolved <= 0 || resolved > 128) {
    throw new Error("maxSequenceLength must be a positive integer no greater than 128.");
  }
  return resolved;
}

function repeatPromptBatch(value: MxArray, numVideosPerPrompt: number): MxArray {
  if (numVideosPerPrompt === 1) {
    return value;
  }
  try {
    return repeat(value, numVideosPerPrompt, 0);
  } finally {
    value.free();
  }
}

function encodeT5PromptIds(
  tokenizer: LtxVideoPromptConditionerComponents["tokenizer"],
  prompts: readonly string[],
  maxSequenceLength: number,
): EncodedPrompts {
  const inputRows: number[][] = [];
  const maskRows: number[][] = [];
  let truncated = false;
  for (const prompt of prompts) {
    const encoded = encodeT5TextInput(tokenizer, prompt, { maxLength: maxSequenceLength });
    inputRows.push(encoded.inputIds);
    maskRows.push(encoded.attentionMask);
    truncated = truncated || encoded.truncated;
  }
  return {
    inputIds: array(inputRows, "int32"),
    attentionMask: array(maskRows, "int32"),
    truncated,
  };
}

function encodeT5Branch(
  tokenizer: LtxVideoPromptConditionerComponents["tokenizer"],
  textEncoder: LtxVideoT5TextEncoder,
  prompts: readonly string[],
  maxSequenceLength: number,
  numVideosPerPrompt: number,
): T5Branch {
  const encoded = encodeT5PromptIds(tokenizer, prompts, maxSequenceLength);
  try {
    const output = textEncoder.run(encoded.inputIds);
    try {
      return {
        encoderHiddenStates: repeatPromptBatch(output.lastHiddenState, numVideosPerPrompt),
        attentionMask: repeatPromptBatch(encoded.attentionMask, numVideosPerPrompt),
        truncated: encoded.truncated,
      };
    } finally {
      disposeT5OutputExceptLastHidden(output);
    }
  } catch (error) {
    encoded.attentionMask.free();
    throw error;
  } finally {
    encoded.inputIds.free();
  }
}

export function encodeLtxVideoPrompt(
  components: LtxVideoPromptConditionerComponents,
  options: LtxVideoPromptConditioningOptions,
): LtxVideoPromptConditioning {
  const prompts = promptBatch(options.prompt, "prompt");
  const numVideosPerPrompt = resolveNumVideosPerPrompt(options.numVideosPerPrompt);
  const maxSequenceLength = resolveMaxSequenceLength(options.maxSequenceLength);
  const batchSize = prompts.length * numVideosPerPrompt;
  const prompt = encodeT5Branch(
    components.tokenizer,
    components.textEncoder,
    prompts,
    maxSequenceLength,
    numVideosPerPrompt,
  );
  let negative: T5Branch | undefined;
  try {
    if (options.includeNegativePrompt === true) {
      negative = encodeT5Branch(
        components.tokenizer,
        components.textEncoder,
        matchingPromptBatch(options.negativePrompt, prompts, "negativePrompt"),
        maxSequenceLength,
        numVideosPerPrompt,
      );
    }
    return new LtxVideoPromptConditioningResult(
      batchSize,
      {
        promptEmbeds: prompt.encoderHiddenStates,
        promptAttentionMask: prompt.attentionMask,
        ...(negative === undefined
          ? {}
          : {
              negativePromptEmbeds: negative.encoderHiddenStates,
              negativePromptAttentionMask: negative.attentionMask,
            }),
      },
      prompt.truncated,
      negative?.truncated ?? false,
    );
  } catch (error) {
    prompt.encoderHiddenStates.free();
    prompt.attentionMask.free();
    negative?.encoderHiddenStates.free();
    negative?.attentionMask.free();
    throw error;
  }
}
