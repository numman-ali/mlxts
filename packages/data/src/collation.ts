import { array, type MxArray, reshape } from "@mlxts/core";

import type { PreferenceExample, TokenSupervisionExample } from "./preference";

export type TokenBatch = {
  inputIds: MxArray;
  targetIds: MxArray;
  lossMask: MxArray;
};

export type PreferenceBatch = {
  chosen: TokenBatch;
  rejected: TokenBatch;
};

function sequenceLength(example: TokenSupervisionExample): number {
  if (example.inputIds.length !== example.targetIds.length) {
    throw new Error("data.collateTokenSupervisionBatch: inputIds and targetIds must match.");
  }
  return example.inputIds.length;
}

function allocateBatch(
  examples: readonly TokenSupervisionExample[],
  padTokenId: number,
): {
  inputIds: Int32Array;
  targetIds: Int32Array;
  lossMask: Float32Array;
  maxLength: number;
} {
  const maxLength = examples.reduce(
    (largest, example) => Math.max(largest, sequenceLength(example)),
    0,
  );
  const batchSize = examples.length;
  const inputIds = new Int32Array(batchSize * maxLength).fill(padTokenId);
  const targetIds = new Int32Array(batchSize * maxLength).fill(padTokenId);
  const lossMask = new Float32Array(batchSize * maxLength);
  return {
    inputIds,
    targetIds,
    lossMask,
    maxLength,
  };
}

function writeExample(
  target: {
    inputIds: Int32Array;
    targetIds: Int32Array;
    lossMask: Float32Array;
    maxLength: number;
  },
  example: TokenSupervisionExample,
  batchIndex: number,
): void {
  const offset = batchIndex * target.maxLength;
  for (let index = 0; index < example.inputIds.length; index += 1) {
    const inputId = example.inputIds[index];
    const targetId = example.targetIds[index];
    if (inputId === undefined || targetId === undefined) {
      throw new Error("data.collateTokenSupervisionBatch: encountered an undefined token.");
    }
    target.inputIds[offset + index] = inputId;
    target.targetIds[offset + index] = targetId;
    target.lossMask[offset + index] = example.lossMask?.[index] ?? 1;
  }
}

function makeTokenBatch(
  inputIds: Int32Array,
  targetIds: Int32Array,
  lossMask: Float32Array,
  batchSize: number,
  maxLength: number,
): TokenBatch {
  using flatInput = array(inputIds, "int32");
  using flatTarget = array(targetIds, "int32");
  using flatMask = array(lossMask, "float32");
  return {
    inputIds: reshape(flatInput, [batchSize, maxLength]),
    targetIds: reshape(flatTarget, [batchSize, maxLength]),
    lossMask: reshape(flatMask, [batchSize, maxLength]),
  };
}

/** Pad and collate token supervision examples into MLX arrays. */
export function collateTokenSupervisionBatch(
  examples: readonly TokenSupervisionExample[],
  padTokenId: number,
): TokenBatch {
  if (examples.length === 0) {
    throw new Error("data.collateTokenSupervisionBatch: expected at least one example.");
  }

  const batchSize = examples.length;
  const allocated = allocateBatch(examples, padTokenId);
  for (let index = 0; index < examples.length; index += 1) {
    const example = examples[index];
    if (example === undefined) {
      continue;
    }
    writeExample(allocated, example, index);
  }

  return makeTokenBatch(
    allocated.inputIds,
    allocated.targetIds,
    allocated.lossMask,
    batchSize,
    allocated.maxLength,
  );
}

function preferenceToSupervision(
  promptIds: readonly number[],
  completionIds: readonly number[],
): TokenSupervisionExample {
  const full = [...promptIds, ...completionIds];
  if (full.length < 2) {
    throw new Error(
      "data.collatePreferenceBatch: prompt + completion must contain at least 2 tokens.",
    );
  }

  const inputIds = full.slice(0, -1);
  const targetIds = full.slice(1);
  const lossMask = targetIds.map((_, index) => (index >= promptIds.length - 1 ? 1 : 0));
  return {
    inputIds,
    targetIds,
    lossMask,
  };
}

/** Pad and collate preference examples into chosen/rejected token batches. */
export function collatePreferenceBatch(
  examples: readonly PreferenceExample[],
  padTokenId: number,
): PreferenceBatch {
  if (examples.length === 0) {
    throw new Error("data.collatePreferenceBatch: expected at least one example.");
  }

  return {
    chosen: collateTokenSupervisionBatch(
      examples.map((example) => preferenceToSupervision(example.promptIds, example.chosenIds)),
      padTokenId,
    ),
    rejected: collateTokenSupervisionBatch(
      examples.map((example) => preferenceToSupervision(example.promptIds, example.rejectedIds)),
      padTokenId,
    ),
  };
}
