/**
 * Ownership helpers for prepared decoder prompts.
 * @module
 */

import {
  slicePromptInputEmbeddings,
  slicePromptPositionIds,
} from "./infrastructure/input-embeddings";
import type { PreparedPrompt } from "./types";

/** Free owned tensors attached to a prepared prompt. */
export function disposePreparedPrompt(prompt: PreparedPrompt): void {
  prompt.inputEmbeddings?.free();
  prompt.positionIds?.free();
}

/** Return an owned prompt window aligned across token ids, embeddings, and position ids. */
export function slicePreparedPrompt(
  prompt: PreparedPrompt,
  start: number,
  length: number,
  context = "slicePreparedPrompt",
): PreparedPrompt {
  if (!Number.isInteger(start) || start < 0) {
    throw new Error(`${context}: start must be a non-negative integer, got ${start}.`);
  }
  if (!Number.isInteger(length) || length <= 0) {
    throw new Error(`${context}: length must be a positive integer, got ${length}.`);
  }
  if (start + length > prompt.tokenIds.length) {
    throw new Error(
      `${context}: cannot slice [${start}, ${start + length}) from prompt length ${prompt.tokenIds.length}.`,
    );
  }

  const window: PreparedPrompt = {
    tokenIds: prompt.tokenIds.slice(start, start + length),
  };
  if (prompt.inputEmbeddings !== undefined) {
    window.inputEmbeddings = slicePromptInputEmbeddings(
      prompt.inputEmbeddings,
      start,
      length,
      context,
    );
  }
  if (prompt.positionIds !== undefined) {
    window.positionIds = slicePromptPositionIds(prompt.positionIds, start, length, context);
  }
  return window;
}
