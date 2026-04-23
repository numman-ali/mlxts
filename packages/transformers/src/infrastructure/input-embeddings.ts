/**
 * Shared helpers for decoder input-embedding composition.
 * @module
 */

import { formatShape, type MxArray, retainArray, slice } from "@mlxts/core";

/** Validate and retain caller-provided decoder input embeddings. */
export function retainInputEmbeddings(
  inputIds: MxArray,
  inputEmbeddings: MxArray | undefined,
  hiddenSize: number,
  context: string,
): MxArray | null {
  if (inputEmbeddings === undefined) {
    return null;
  }

  const [batch, sequenceLength] = inputIds.shape;
  const [embeddingBatch, embeddingSequenceLength, embeddingHiddenSize] = inputEmbeddings.shape;
  if (
    batch === undefined ||
    sequenceLength === undefined ||
    inputIds.shape.length !== 2 ||
    embeddingBatch === undefined ||
    embeddingSequenceLength === undefined ||
    embeddingHiddenSize === undefined ||
    inputEmbeddings.shape.length !== 3
  ) {
    throw new Error(
      `${context}: inputEmbeddings requires token ids with shape [batch, seq] and embeddings with shape [batch, seq, hidden], got ${formatShape(inputIds.shape)} and ${formatShape(inputEmbeddings.shape)}.`,
    );
  }
  if (batch !== embeddingBatch || sequenceLength !== embeddingSequenceLength) {
    throw new Error(
      `${context}: inputEmbeddings batch/sequence shape ${formatShape(inputEmbeddings.shape)} must match token ids ${formatShape(inputIds.shape)}.`,
    );
  }
  if (embeddingHiddenSize !== hiddenSize) {
    throw new Error(
      `${context}: inputEmbeddings hidden size ${embeddingHiddenSize} must match model hidden size ${hiddenSize}.`,
    );
  }
  return retainArray(inputEmbeddings);
}

/** Validate and retain caller-provided decoder position ids. */
export function retainInputPositionIds(
  inputIds: MxArray,
  positionIds: MxArray | undefined,
  context: string,
): MxArray | null {
  if (positionIds === undefined) {
    return null;
  }

  const [batch, sequenceLength] = inputIds.shape;
  if (batch === undefined || sequenceLength === undefined || inputIds.shape.length !== 2) {
    throw new Error(
      `${context}: positionIds requires token ids with shape [batch, seq], got ${formatShape(inputIds.shape)}.`,
    );
  }

  if (positionIds.shape.length === 2) {
    const [positionBatch, positionSequenceLength] = positionIds.shape;
    if (positionBatch !== batch || positionSequenceLength !== sequenceLength) {
      throw new Error(
        `${context}: positionIds shape ${formatShape(positionIds.shape)} must match token ids ${formatShape(inputIds.shape)}.`,
      );
    }
    return retainArray(positionIds);
  }

  if (positionIds.shape.length === 3) {
    const [, positionBatch, positionSequenceLength] = positionIds.shape;
    if (positionBatch !== batch || positionSequenceLength !== sequenceLength) {
      throw new Error(
        `${context}: positionIds shape ${formatShape(positionIds.shape)} must match token ids ${formatShape(inputIds.shape)}.`,
      );
    }
    return retainArray(positionIds);
  }

  throw new Error(
    `${context}: positionIds must have shape [batch, seq] or [axes, batch, seq], got ${formatShape(positionIds.shape)}.`,
  );
}

/** Validate that prompt-aligned embeddings can be used for cached generation prefill. */
export function validatePromptInputEmbeddings(
  promptTokenIds: readonly number[],
  inputEmbeddings: MxArray,
  context: string,
): void {
  const [batch, sequenceLength] = inputEmbeddings.shape;
  if (batch === undefined || sequenceLength === undefined || inputEmbeddings.shape.length !== 3) {
    throw new Error(
      `${context}: inputEmbeddings must have shape [batch, seq, hidden], got ${formatShape(inputEmbeddings.shape)}.`,
    );
  }
  if (batch !== 1) {
    throw new Error(`${context}: generation inputEmbeddings currently requires batch size 1.`);
  }
  if (sequenceLength !== promptTokenIds.length) {
    throw new Error(
      `${context}: inputEmbeddings sequence length ${sequenceLength} must match prompt length ${promptTokenIds.length}.`,
    );
  }
}

/** Validate that prompt-aligned position ids can be used for cached generation prefill. */
export function validatePromptPositionIds(
  promptTokenIds: readonly number[],
  positionIds: MxArray,
  context: string,
): void {
  if (positionIds.shape.length === 2) {
    const [batch, sequenceLength] = positionIds.shape;
    if (batch !== 1 || sequenceLength !== promptTokenIds.length) {
      throw new Error(
        `${context}: positionIds shape ${formatShape(positionIds.shape)} must be [1, ${promptTokenIds.length}] for prompt-aligned generation.`,
      );
    }
    return;
  }

  if (positionIds.shape.length === 3) {
    const [, batch, sequenceLength] = positionIds.shape;
    if (batch !== 1 || sequenceLength !== promptTokenIds.length) {
      throw new Error(
        `${context}: positionIds shape ${formatShape(positionIds.shape)} must have batch size 1 and sequence length ${promptTokenIds.length}.`,
      );
    }
    return;
  }

  throw new Error(
    `${context}: positionIds must have shape [1, seq] or [axes, 1, seq], got ${formatShape(positionIds.shape)}.`,
  );
}

/** Validate and retain prompt-aligned embeddings for cached or uncached generation. */
export function retainPromptInputEmbeddings(
  promptTokenIds: readonly number[],
  inputEmbeddings: MxArray | undefined,
  context: string,
): MxArray | null {
  if (inputEmbeddings === undefined) {
    return null;
  }
  validatePromptInputEmbeddings(promptTokenIds, inputEmbeddings, context);
  return retainArray(inputEmbeddings);
}

/** Validate and retain prompt-aligned position ids for cached or uncached generation. */
export function retainPromptPositionIds(
  promptTokenIds: readonly number[],
  positionIds: MxArray | undefined,
  context: string,
): MxArray | null {
  if (positionIds === undefined) {
    return null;
  }
  validatePromptPositionIds(promptTokenIds, positionIds, context);
  return retainArray(positionIds);
}

/** Slice the embedding-aligned prompt window for one cached prefill step. */
export function slicePromptInputEmbeddings(
  inputEmbeddings: MxArray,
  start: number,
  length: number,
  context: string,
): MxArray {
  const hiddenSize = inputEmbeddings.shape[2];
  if (hiddenSize === undefined || inputEmbeddings.shape.length !== 3) {
    throw new Error(
      `${context}: inputEmbeddings must have shape [batch, seq, hidden], got ${formatShape(inputEmbeddings.shape)}.`,
    );
  }
  return slice(inputEmbeddings, [0, start, 0], [1, start + length, hiddenSize]);
}

/** Slice the position-id-aligned prompt window for one cached prefill step. */
export function slicePromptPositionIds(
  positionIds: MxArray,
  start: number,
  length: number,
  context: string,
): MxArray {
  if (positionIds.shape.length === 2) {
    const [batch] = positionIds.shape;
    if (batch !== 1) {
      throw new Error(`${context}: prompt positionIds requires batch size 1.`);
    }
    return slice(positionIds, [0, start], [1, start + length]);
  }

  if (positionIds.shape.length === 3) {
    const [axes, batch] = positionIds.shape;
    if (axes === undefined || batch !== 1) {
      throw new Error(`${context}: prompt positionIds requires shape [axes, 1, seq].`);
    }
    return slice(positionIds, [0, 0, start], [axes, 1, start + length]);
  }

  throw new Error(
    `${context}: positionIds must have shape [1, seq] or [axes, 1, seq], got ${formatShape(positionIds.shape)}.`,
  );
}
