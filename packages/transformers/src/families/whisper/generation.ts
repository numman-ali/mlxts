/**
 * Whisper greedy transcription helpers.
 * @module
 */

import { argmax, MxArray, mxEval, slice } from "@mlxts/core";
import type { Tokenizer } from "@mlxts/tokenizers";

import type { WhisperForConditionalGeneration } from "./model";
import {
  createWhisperDecoderPromptTokenIds,
  decodeWhisperGeneratedTokenIds,
  type WhisperPromptOptions,
  type WhisperSpecialTokens,
} from "./tokenizer";
import type { WhisperEncoderOutput } from "./types";

export type WhisperGreedyTokenEvent = {
  step: number;
  tokenId: number;
  tokenIds: readonly number[];
};

export type WhisperGreedyTranscriptionOptions = WhisperPromptOptions & {
  maxNewTokens?: number;
  onToken?: (event: WhisperGreedyTokenEvent) => void;
};

export type WhisperGreedyTranscriptionResult = {
  text: string;
  promptTokenIds: readonly number[];
  generatedTokenIds: readonly number[];
  tokenIds: readonly number[];
  generatedTokens: number;
  stoppedReason: "eos" | "max_tokens";
};

function disposeEncoderOutput(output: WhisperEncoderOutput): void {
  output.lastHiddenState.free();
  if (output.hiddenStates !== undefined) {
    for (const hiddenState of output.hiddenStates) {
      hiddenState.free();
    }
  }
}

function inputIdsTensor(tokenIds: readonly number[]): MxArray {
  return MxArray.fromData(new Int32Array(tokenIds), [1, tokenIds.length], "int32");
}

function normalizeMaxNewTokens(
  requested: number | undefined,
  promptLength: number,
  maxTargetPositions: number,
): number {
  const available = maxTargetPositions - promptLength;
  if (available <= 0) {
    throw new Error(
      `generateWhisperGreedyTranscription: prompt length ${promptLength} leaves no decoder positions.`,
    );
  }
  if (requested === undefined) {
    return Math.min(64, available);
  }
  if (!Number.isInteger(requested) || requested <= 0) {
    throw new Error("generateWhisperGreedyTranscription: maxNewTokens must be a positive integer.");
  }
  return Math.min(requested, available);
}

function nextGreedyToken(model: WhisperForConditionalGeneration, ids: MxArray, audio: MxArray) {
  const decoderOutput = model.model.decoder.run(ids, audio);
  let logits: MxArray | null = null;
  try {
    logits = model.model.decoder.projectLogits(decoderOutput.lastHiddenState);
    const [, sequenceLength, vocabSize] = logits.shape;
    if (sequenceLength === undefined || vocabSize === undefined) {
      throw new Error("generateWhisperGreedyTranscription: decoder logits have unknown shape.");
    }
    using lastLogits = slice(logits, [0, sequenceLength - 1, 0], [1, sequenceLength, vocabSize]);
    using token = argmax(lastLogits, 2);
    mxEval(token);
    return token.item();
  } finally {
    logits?.free();
    decoderOutput.lastHiddenState.free();
    if (decoderOutput.hiddenStates !== undefined) {
      for (const hiddenState of decoderOutput.hiddenStates) {
        hiddenState.free();
      }
    }
  }
}

/** Run finite greedy Whisper transcription over prepared log-mel input features. */
export function generateWhisperGreedyTranscription(
  model: WhisperForConditionalGeneration,
  inputFeatures: MxArray,
  tokenizer: Tokenizer,
  specialTokens: WhisperSpecialTokens,
  options: WhisperGreedyTranscriptionOptions = {},
): WhisperGreedyTranscriptionResult {
  const promptTokenIds = createWhisperDecoderPromptTokenIds(specialTokens, options);
  const maxNewTokens = normalizeMaxNewTokens(
    options.maxNewTokens,
    promptTokenIds.length,
    model.config.maxTargetPositions,
  );
  const endOfText = new Set(specialTokens.endOfTextTokenIds);
  const tokenIds = [...promptTokenIds];
  const generatedTokenIds: number[] = [];
  const encoderOutput = model.model.encoder.run(inputFeatures);
  let stoppedReason: "eos" | "max_tokens" = "max_tokens";

  try {
    for (let step = 0; step < maxNewTokens; step += 1) {
      using decoderInputIds = inputIdsTensor(tokenIds);
      const tokenId = nextGreedyToken(model, decoderInputIds, encoderOutput.lastHiddenState);
      tokenIds.push(tokenId);
      generatedTokenIds.push(tokenId);
      options.onToken?.({ step: step + 1, tokenId, tokenIds });
      if (endOfText.has(tokenId)) {
        stoppedReason = "eos";
        break;
      }
    }
  } finally {
    disposeEncoderOutput(encoderOutput);
  }

  return {
    text: decodeWhisperGeneratedTokenIds(tokenizer, generatedTokenIds, specialTokens),
    promptTokenIds,
    generatedTokenIds,
    tokenIds,
    generatedTokens: generatedTokenIds.length,
    stoppedReason,
  };
}
