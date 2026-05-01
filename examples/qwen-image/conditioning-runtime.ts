import { array, formatShape, type MxArray, retainArray, slice } from "@mlxts/core";

import { QwenImagePromptConditioningResult } from "./conditioning-result";
import type {
  QwenImagePromptConditionerComponents,
  QwenImagePromptConditioning,
  QwenImagePromptConditioningOptions,
  QwenImageTextModelOutput,
} from "./conditioning-types";

export const QWEN_IMAGE_PROMPT_TEMPLATE_PREFIX =
  "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n";
export const QWEN_IMAGE_PROMPT_TEMPLATE_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n";
export const QWEN_IMAGE_PROMPT_DROP_TOKENS = 34;
export const QWEN_IMAGE_DEFAULT_NEGATIVE_PROMPT = " ";
export const QWEN_IMAGE_DEFAULT_TRUE_CFG_SCALE = 4;
export const QWEN_IMAGE_MAX_SEQUENCE_LENGTH = 1024;

type EncodedPrompt = {
  inputIds: MxArray;
  truncated: boolean;
};

type PromptEmbedding = {
  embeds: MxArray;
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

function disposeTextModelOutput(output: QwenImageTextModelOutput): void {
  output.lastHiddenState.free();
  disposeHiddenStates(output.hiddenStates);
}

function resolvePrompt(prompt: string, name: string): string {
  if (prompt === "" || (name === "prompt" && prompt.trim() === "")) {
    throw new Error(`${name} must not be empty.`);
  }
  return prompt;
}

function resolveMaxSequenceLength(value: number | undefined): number {
  const resolved = value ?? QWEN_IMAGE_MAX_SEQUENCE_LENGTH;
  if (!Number.isInteger(resolved) || resolved <= 0 || resolved > QWEN_IMAGE_MAX_SEQUENCE_LENGTH) {
    throw new Error(
      `maxSequenceLength must be a positive integer no greater than ${QWEN_IMAGE_MAX_SEQUENCE_LENGTH}.`,
    );
  }
  return resolved;
}

function resolveTrueCfgScale(value: number | undefined): number {
  const resolved = value ?? QWEN_IMAGE_DEFAULT_TRUE_CFG_SCALE;
  if (!Number.isFinite(resolved) || resolved <= 0) {
    throw new Error("trueCfgScale must be a positive finite number.");
  }
  return resolved;
}

export function renderQwenImagePrompt(prompt: string): string {
  return `${QWEN_IMAGE_PROMPT_TEMPLATE_PREFIX}${prompt}${QWEN_IMAGE_PROMPT_TEMPLATE_SUFFIX}`;
}

function encodePromptIds(
  components: QwenImagePromptConditionerComponents,
  prompt: string,
  maxSequenceLength: number,
): EncodedPrompt {
  const tokenIds = components.tokenizer.encode(renderQwenImagePrompt(prompt));
  const maxInputLength = maxSequenceLength + QWEN_IMAGE_PROMPT_DROP_TOKENS;
  const inputTokenIds =
    tokenIds.length > maxInputLength ? tokenIds.slice(0, maxInputLength) : tokenIds;
  if (inputTokenIds.length <= QWEN_IMAGE_PROMPT_DROP_TOKENS) {
    throw new Error("Qwen-Image prompt encoding produced no post-template token ids.");
  }
  return {
    inputIds: array([inputTokenIds], "int32"),
    truncated: inputTokenIds.length !== tokenIds.length,
  };
}

function retainPromptEmbeds(output: QwenImageTextModelOutput): MxArray {
  const [batchSize, sequenceLength, hiddenSize] = output.lastHiddenState.shape;
  if (
    output.lastHiddenState.shape.length !== 3 ||
    batchSize !== 1 ||
    sequenceLength === undefined ||
    hiddenSize === undefined
  ) {
    throw new Error(
      `Qwen-Image text encoder hidden state must be [1, sequence, hidden], got ${formatShape(
        output.lastHiddenState.shape,
      )}.`,
    );
  }
  if (sequenceLength <= QWEN_IMAGE_PROMPT_DROP_TOKENS) {
    throw new Error("Qwen-Image text encoder output contains no post-template tokens.");
  }
  using promptEmbeds = slice(
    output.lastHiddenState,
    [0, QWEN_IMAGE_PROMPT_DROP_TOKENS, 0],
    [1, sequenceLength, hiddenSize],
  );
  return retainArray(promptEmbeds);
}

function encodePromptEmbedding(
  components: QwenImagePromptConditionerComponents,
  prompt: string,
  maxSequenceLength: number,
): PromptEmbedding {
  const encoded = encodePromptIds(components, prompt, maxSequenceLength);
  try {
    const output = components.textEncoder.model.runWithHiddenStates(encoded.inputIds);
    try {
      return {
        embeds: retainPromptEmbeds(output),
        truncated: encoded.truncated,
      };
    } finally {
      disposeTextModelOutput(output);
    }
  } finally {
    encoded.inputIds.free();
  }
}

export function encodeQwenImagePrompt(
  components: QwenImagePromptConditionerComponents,
  options: QwenImagePromptConditioningOptions,
): QwenImagePromptConditioning {
  const prompt = resolvePrompt(options.prompt, "prompt");
  const maxSequenceLength = resolveMaxSequenceLength(options.maxSequenceLength);
  const trueCfgScale = resolveTrueCfgScale(options.trueCfgScale);
  const positive = encodePromptEmbedding(components, prompt, maxSequenceLength);
  let promptEmbeds: MxArray | undefined = positive.embeds;
  let negativePromptEmbeds: MxArray | undefined;
  let negativePromptTruncated = false;

  try {
    if (trueCfgScale > 1) {
      const negativePrompt = resolvePrompt(
        options.negativePrompt ?? QWEN_IMAGE_DEFAULT_NEGATIVE_PROMPT,
        "negativePrompt",
      );
      const negative = encodePromptEmbedding(components, negativePrompt, maxSequenceLength);
      negativePromptEmbeds = negative.embeds;
      negativePromptTruncated = negative.truncated;
    }

    const conditioning = {
      promptEmbeds,
      trueCfgScale,
      ...(negativePromptEmbeds === undefined ? {} : { negativePromptEmbeds }),
    };
    const result = new QwenImagePromptConditioningResult(
      1,
      conditioning,
      positive.truncated,
      negativePromptTruncated,
    );
    promptEmbeds = undefined;
    negativePromptEmbeds = undefined;
    return result;
  } finally {
    promptEmbeds?.free();
    negativePromptEmbeds?.free();
  }
}
