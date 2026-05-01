import {
  array,
  formatShape,
  type MxArray,
  reshape,
  retainArray,
  stack,
  transpose,
} from "@mlxts/core";

import { Flux2KleinPromptConditioningResult } from "./conditioning-result";
import type {
  Flux2KleinPromptConditionerComponents,
  Flux2KleinPromptConditioning,
  Flux2KleinPromptConditioningOptions,
  Flux2KleinTextModelOutput,
} from "./conditioning-types";

export const FLUX2_KLEIN_MAX_SEQUENCE_LENGTH = 512;
export const FLUX2_KLEIN_DEFAULT_GUIDANCE_SCALE = 4;
export const FLUX2_KLEIN_DEFAULT_NEGATIVE_PROMPT = "";
export const FLUX2_KLEIN_DEFAULT_TEXT_ENCODER_OUT_LAYERS = [9, 18, 27] as const;

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

function disposeTextModelOutput(output: Flux2KleinTextModelOutput): void {
  output.lastHiddenState.free();
  disposeHiddenStates(output.hiddenStates);
}

function resolvePrompt(prompt: string): string {
  if (prompt.trim() === "") {
    throw new Error("prompt must not be empty.");
  }
  return prompt;
}

function resolveNegativePrompt(prompt: string | undefined): string {
  return prompt ?? FLUX2_KLEIN_DEFAULT_NEGATIVE_PROMPT;
}

function resolveMaxSequenceLength(value: number | undefined): number {
  const resolved = value ?? FLUX2_KLEIN_MAX_SEQUENCE_LENGTH;
  if (!Number.isInteger(resolved) || resolved <= 0 || resolved > FLUX2_KLEIN_MAX_SEQUENCE_LENGTH) {
    throw new Error(
      `maxSequenceLength must be a positive integer no greater than ${FLUX2_KLEIN_MAX_SEQUENCE_LENGTH}.`,
    );
  }
  return resolved;
}

function resolveGuidanceScale(value: number | undefined): number {
  const resolved = value ?? FLUX2_KLEIN_DEFAULT_GUIDANCE_SCALE;
  if (!Number.isFinite(resolved) || resolved <= 0) {
    throw new Error("guidanceScale must be a positive finite number.");
  }
  return resolved;
}

function resolveTextEncoderOutLayers(value: readonly number[] | undefined): readonly number[] {
  const resolved = value ?? FLUX2_KLEIN_DEFAULT_TEXT_ENCODER_OUT_LAYERS;
  if (resolved.length === 0) {
    throw new Error("textEncoderOutLayers must contain at least one layer index.");
  }
  for (const layer of resolved) {
    if (!Number.isInteger(layer) || layer < 0) {
      throw new Error("textEncoderOutLayers must contain non-negative integer layer indices.");
    }
  }
  return resolved;
}

function encodePromptIds(
  components: Flux2KleinPromptConditionerComponents,
  prompt: string,
  maxSequenceLength: number,
): EncodedPrompt {
  const compiled = components.interactionProfile.compileMessages(
    components.tokenizer,
    [{ role: "user", content: prompt }],
    {
      addGenerationPrompt: true,
      enableThinking: false,
    },
  );
  const tokenIds =
    compiled.tokenIds.length > maxSequenceLength
      ? compiled.tokenIds.slice(0, maxSequenceLength)
      : compiled.tokenIds;
  const paddedTokenIds =
    components.tokenizer.padTokenId === undefined || tokenIds.length === maxSequenceLength
      ? tokenIds
      : [
          ...tokenIds,
          ...Array.from(
            { length: maxSequenceLength - tokenIds.length },
            () => components.tokenizer.padTokenId ?? 0,
          ),
        ];
  if (tokenIds.length === 0) {
    throw new Error("FLUX.2 Klein prompt compilation produced no token ids.");
  }
  return {
    inputIds: array([paddedTokenIds], "int32"),
    truncated: tokenIds.length !== compiled.tokenIds.length,
  };
}

function retainPromptEmbeds(
  output: Flux2KleinTextModelOutput,
  textEncoderOutLayers: readonly number[],
): MxArray {
  const hiddenStates = output.hiddenStates;
  if (hiddenStates === undefined) {
    throw new Error("FLUX.2 Klein prompt conditioning requires text encoder hidden states.");
  }
  const selected: MxArray[] = [];
  for (const layerIndex of textEncoderOutLayers) {
    const hiddenState = hiddenStates[layerIndex];
    if (hiddenState === undefined) {
      throw new Error(`FLUX.2 Klein prompt conditioning is missing hidden state ${layerIndex}.`);
    }
    selected.push(hiddenState);
  }

  const [batchSize, sequenceLength, hiddenSize] = selected[0]?.shape ?? [];
  if (
    batchSize !== 1 ||
    sequenceLength === undefined ||
    hiddenSize === undefined ||
    selected[0]?.shape.length !== 3
  ) {
    throw new Error(
      `FLUX.2 Klein text encoder hidden state must be [1, sequence, hidden], got ${formatShape(
        selected[0]?.shape ?? [],
      )}.`,
    );
  }
  for (const hiddenState of selected) {
    if (
      hiddenState.shape.length !== 3 ||
      hiddenState.shape[0] !== 1 ||
      hiddenState.shape[1] !== sequenceLength ||
      hiddenState.shape[2] !== hiddenSize
    ) {
      throw new Error(
        `FLUX.2 Klein selected hidden states must share shape [1, ${sequenceLength}, ${hiddenSize}], got ${formatShape(
          hiddenState.shape,
        )}.`,
      );
    }
  }

  using stacked = stack(selected, 1);
  using mergedOrder = transpose(stacked, [0, 2, 1, 3]);
  using promptEmbeds = reshape(mergedOrder, [
    1,
    sequenceLength,
    textEncoderOutLayers.length * hiddenSize,
  ]);
  return retainArray(promptEmbeds);
}

function encodePromptEmbedding(
  components: Flux2KleinPromptConditionerComponents,
  prompt: string,
  maxSequenceLength: number,
  textEncoderOutLayers: readonly number[],
): PromptEmbedding {
  const encoded = encodePromptIds(components, prompt, maxSequenceLength);
  try {
    const output = components.textEncoder.model.runWithHiddenStates(encoded.inputIds, {
      outputHiddenStates: true,
    });
    try {
      return {
        embeds: retainPromptEmbeds(output, textEncoderOutLayers),
        truncated: encoded.truncated,
      };
    } finally {
      disposeTextModelOutput(output);
    }
  } finally {
    encoded.inputIds.free();
  }
}

export function encodeFlux2KleinPrompt(
  components: Flux2KleinPromptConditionerComponents,
  options: Flux2KleinPromptConditioningOptions,
): Flux2KleinPromptConditioning {
  const prompt = resolvePrompt(options.prompt);
  const maxSequenceLength = resolveMaxSequenceLength(options.maxSequenceLength);
  const guidanceScale = resolveGuidanceScale(options.guidanceScale);
  const textEncoderOutLayers = resolveTextEncoderOutLayers(options.textEncoderOutLayers);
  const positive = encodePromptEmbedding(
    components,
    prompt,
    maxSequenceLength,
    textEncoderOutLayers,
  );
  let promptEmbeds: MxArray | undefined = positive.embeds;
  let negativePromptEmbeds: MxArray | undefined;
  let negativePromptTruncated = false;

  try {
    if (guidanceScale > 1) {
      const negative = encodePromptEmbedding(
        components,
        resolveNegativePrompt(options.negativePrompt),
        maxSequenceLength,
        textEncoderOutLayers,
      );
      negativePromptEmbeds = negative.embeds;
      negativePromptTruncated = negative.truncated;
    }

    const conditioning = {
      promptEmbeds,
      guidanceScale,
      ...(negativePromptEmbeds === undefined ? {} : { negativePromptEmbeds }),
    };
    const result = new Flux2KleinPromptConditioningResult(
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
