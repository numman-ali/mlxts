import { describe, expect, test } from "bun:test";
import { mxEval } from "@mlxts/core";

import { collatePreferenceBatch, collateTokenSupervisionBatch } from "./collation";

describe("collateTokenSupervisionBatch", () => {
  test("pads inputs, targets, and masks", () => {
    const batch = collateTokenSupervisionBatch(
      [
        {
          inputIds: [1, 2],
          targetIds: [2, 3],
        },
        {
          inputIds: [4],
          targetIds: [5],
          lossMask: [0],
        },
      ],
      0,
    );

    try {
      mxEval(batch.inputIds, batch.targetIds, batch.lossMask);
      expect(batch.inputIds.toList()).toEqual([
        [1, 2],
        [4, 0],
      ]);
      expect(batch.targetIds.toList()).toEqual([
        [2, 3],
        [5, 0],
      ]);
      expect(batch.lossMask.toList()).toEqual([
        [1, 1],
        [0, 0],
      ]);
    } finally {
      batch.inputIds.free();
      batch.targetIds.free();
      batch.lossMask.free();
    }
  });
});

describe("collatePreferenceBatch", () => {
  test("builds chosen and rejected supervision batches", () => {
    const batch = collatePreferenceBatch(
      [
        {
          promptIds: [1, 2],
          chosenIds: [3, 4],
          rejectedIds: [5],
        },
      ],
      0,
    );

    try {
      mxEval(
        batch.chosen.inputIds,
        batch.chosen.targetIds,
        batch.chosen.lossMask,
        batch.rejected.inputIds,
        batch.rejected.targetIds,
        batch.rejected.lossMask,
      );
      expect(batch.chosen.inputIds.toList()).toEqual([[1, 2, 3]]);
      expect(batch.chosen.targetIds.toList()).toEqual([[2, 3, 4]]);
      expect(batch.chosen.lossMask.toList()).toEqual([[0, 1, 1]]);
      expect(batch.rejected.inputIds.toList()).toEqual([[1, 2]]);
      expect(batch.rejected.targetIds.toList()).toEqual([[2, 5]]);
      expect(batch.rejected.lossMask.toList()).toEqual([[0, 1]]);
    } finally {
      batch.chosen.inputIds.free();
      batch.chosen.targetIds.free();
      batch.chosen.lossMask.free();
      batch.rejected.inputIds.free();
      batch.rejected.targetIds.free();
      batch.rejected.lossMask.free();
    }
  });
});
