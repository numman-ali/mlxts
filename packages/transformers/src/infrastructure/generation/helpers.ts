/**
 * Shared generation helpers for decoder inference.
 * @module
 */

import { array, clearMemoryCache, MxArray, mxEval, slice, squeeze } from "@mlxts/core";
import type { CausalLM, ForwardOptions, TransformerCache } from "../../types";
import { cacheStateArrays } from "../cache";
import {
  slicePromptInputEmbeddings,
  slicePromptPositionIds,
  validatePromptInputEmbeddings,
  validatePromptPositionIds,
} from "../input-embeddings";

/** Convert prompt tokens into the `[batch, sequence]` input tensor shape. */
export function inputTensor(tokenIds: readonly number[] | MxArray): MxArray {
  return tokenIds instanceof MxArray ? tokenIds : array([tokenIds], "int32");
}

/** Slice the final time step logits without materializing the full sequence on CPU. */
export function takeLastLogits(logits: MxArray, context: string): MxArray {
  const sequenceLength = logits.shape[1];
  if (sequenceLength === undefined || sequenceLength <= 0 || logits.shape.length !== 3) {
    throw new Error(
      `${context}: model logits must have shape [batch, seq, vocab], got [${logits.shape.join(", ")}].`,
    );
  }

  const [batch, , vocabSize] = logits.shape;
  if (batch === undefined || vocabSize === undefined) {
    throw new Error(`${context}: model logits are missing batch or vocabulary axes.`);
  }

  using lastStep = slice(logits, [0, sequenceLength - 1, 0], [batch, sequenceLength, vocabSize]);
  return squeeze(lastStep, 1);
}

/** Validate the configured chunk size for prompt prefill. */
export function validatePrefillStepSize(prefillStepSize: number, context: string): void {
  if (!Number.isInteger(prefillStepSize) || prefillStepSize <= 0) {
    throw new Error(
      `${context}: prefillStepSize must be a positive integer, got ${prefillStepSize}.`,
    );
  }
}

export type PrefilledPrompt = {
  tokenIds: number[];
  inputEmbeddings: MxArray | null;
  positionIds: MxArray | null;
};

function chunkForwardOptions(
  cache: TransformerCache,
  inputEmbeddings: MxArray | null,
  positionIds: MxArray | null,
): ForwardOptions {
  const options: ForwardOptions = { cache };
  if (inputEmbeddings !== null) {
    options.inputEmbeddings = inputEmbeddings;
  }
  if (positionIds !== null) {
    options.positionIds = positionIds;
  }
  return options;
}

/** Materialize retained cache state arrays so recurrent cache graphs do not grow across tokens. */
export function materializeCacheState(cache: TransformerCache): boolean {
  const stateArrays = cacheStateArrays(cache);
  try {
    if (stateArrays.length === 0) {
      return false;
    }
    mxEval(...stateArrays);
    return true;
  } finally {
    for (const stateArray of stateArrays) {
      stateArray.free();
    }
  }
}

/** Chunk long prompts so cache state is materialized without giant intermediate graphs. */
export function prefillPromptCache(
  model: CausalLM,
  promptTokenIds: readonly number[],
  cache: TransformerCache,
  prefillStepSize: number,
  inputEmbeddings?: MxArray,
  positionIds?: MxArray,
): PrefilledPrompt {
  if (inputEmbeddings !== undefined) {
    validatePromptInputEmbeddings(promptTokenIds, inputEmbeddings, "prefillPromptCache");
  }
  if (positionIds !== undefined) {
    validatePromptPositionIds(promptTokenIds, positionIds, "prefillPromptCache");
  }

  let cursor = 0;

  while (promptTokenIds.length - cursor > 1) {
    const remaining = promptTokenIds.length - cursor - 1;
    const chunkSize = Math.min(prefillStepSize, remaining);
    const chunk = promptTokenIds.slice(cursor, cursor + chunkSize);

    using chunkIds = array([chunk], "int32");
    const chunkInputEmbeddings =
      inputEmbeddings === undefined
        ? null
        : slicePromptInputEmbeddings(inputEmbeddings, cursor, chunkSize, "prefillPromptCache");
    const chunkPositionIds =
      positionIds === undefined
        ? null
        : slicePromptPositionIds(positionIds, cursor, chunkSize, "prefillPromptCache");
    try {
      using chunkLogits = model.forward(
        chunkIds,
        chunkForwardOptions(cache, chunkInputEmbeddings, chunkPositionIds),
      );
      if (!materializeCacheState(cache)) {
        chunkLogits.eval();
      }
    } finally {
      chunkInputEmbeddings?.free();
      chunkPositionIds?.free();
    }

    cursor += chunkSize;
    clearMemoryCache();
  }

  return {
    tokenIds: promptTokenIds.slice(cursor),
    inputEmbeddings:
      inputEmbeddings === undefined
        ? null
        : slicePromptInputEmbeddings(
            inputEmbeddings,
            cursor,
            promptTokenIds.length - cursor,
            "prefillPromptCache",
          ),
    positionIds:
      positionIds === undefined
        ? null
        : slicePromptPositionIds(
            positionIds,
            cursor,
            promptTokenIds.length - cursor,
            "prefillPromptCache",
          ),
  };
}
