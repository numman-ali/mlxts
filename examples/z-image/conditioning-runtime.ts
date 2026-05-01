import { array, formatShape, type MxArray, retainArray, squeeze } from "@mlxts/core";

import { ZImagePromptConditioningResult } from "./conditioning-result";
import type {
  ZImagePromptConditionerComponents,
  ZImagePromptConditioning,
  ZImagePromptConditioningOptions,
  ZImageTextModelOutput,
} from "./conditioning-types";

type EncodedPrompt = {
  inputIds: MxArray;
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

function disposeTextModelOutput(output: ZImageTextModelOutput): void {
  output.lastHiddenState.free();
  disposeHiddenStates(output.hiddenStates);
}

function resolvePrompt(prompt: string): string {
  if (prompt.trim() === "") {
    throw new Error("prompt must not be empty.");
  }
  return prompt;
}

function resolveMaxSequenceLength(value: number | undefined): number {
  const resolved = value ?? 512;
  if (!Number.isInteger(resolved) || resolved <= 0 || resolved > 512) {
    throw new Error("maxSequenceLength must be a positive integer no greater than 512.");
  }
  return resolved;
}

function encodePromptIds(
  components: ZImagePromptConditionerComponents,
  prompt: string,
  maxSequenceLength: number,
): EncodedPrompt {
  const compiled = components.interactionProfile.compileMessages(
    components.tokenizer,
    [{ role: "user", content: prompt }],
    {
      addGenerationPrompt: true,
      enableThinking: true,
    },
  );
  const tokenIds =
    compiled.tokenIds.length > maxSequenceLength
      ? compiled.tokenIds.slice(0, maxSequenceLength)
      : compiled.tokenIds;
  if (tokenIds.length === 0) {
    throw new Error("Z-Image prompt compilation produced no token ids.");
  }
  return {
    inputIds: array([tokenIds], "int32"),
    truncated: tokenIds.length !== compiled.tokenIds.length,
  };
}

function retainPenultimateHiddenState(output: ZImageTextModelOutput): MxArray {
  const hiddenStates = output.hiddenStates;
  if (hiddenStates === undefined || hiddenStates.length < 2) {
    throw new Error("Z-Image prompt conditioning requires text encoder hidden states.");
  }
  const penultimate = hiddenStates[hiddenStates.length - 2];
  if (penultimate === undefined) {
    throw new Error("Z-Image prompt conditioning could not select the penultimate hidden state.");
  }
  const [batchSize, sequenceLength, hiddenSize] = penultimate.shape;
  if (
    penultimate.shape.length !== 3 ||
    batchSize !== 1 ||
    sequenceLength === undefined ||
    hiddenSize === undefined
  ) {
    throw new Error(
      `Z-Image text encoder hidden state must be [1, sequence, hidden], got ${formatShape(
        penultimate.shape,
      )}.`,
    );
  }
  using captionFeatures = squeeze(penultimate, 0);
  return retainArray(captionFeatures);
}

function encodeCaptionFeature(
  components: ZImagePromptConditionerComponents,
  inputIds: MxArray,
): MxArray {
  const output = components.textEncoder.model.runWithHiddenStates(inputIds, {
    outputHiddenStates: true,
  });
  try {
    return retainPenultimateHiddenState(output);
  } finally {
    disposeTextModelOutput(output);
  }
}

export function encodeZImagePrompt(
  components: ZImagePromptConditionerComponents,
  options: ZImagePromptConditioningOptions,
): ZImagePromptConditioning {
  const prompt = resolvePrompt(options.prompt);
  const maxSequenceLength = resolveMaxSequenceLength(options.maxSequenceLength);
  const encoded = encodePromptIds(components, prompt, maxSequenceLength);
  try {
    const captionFeature = encodeCaptionFeature(components, encoded.inputIds);
    try {
      return new ZImagePromptConditioningResult(
        1,
        { captionFeatures: [captionFeature] },
        encoded.truncated,
      );
    } catch (error) {
      captionFeature.free();
      throw error;
    }
  } finally {
    encoded.inputIds.free();
  }
}
