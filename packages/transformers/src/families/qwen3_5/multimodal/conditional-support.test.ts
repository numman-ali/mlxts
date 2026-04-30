import { describe, expect, test } from "bun:test";
import { array } from "@mlxts/core";

import {
  buildPositionIds,
  countImageTokens,
  countQwen3_5ImageTokens,
  countQwen3_5ImageTokensFromGridThw,
  createImageMask,
  createQwen3_5MmTokenTypeIds,
  expandQwen3_5ImageTokens,
  expandQwen3_5ImageTokensFromGridThw,
  gridThwList,
  ropeDeltas,
} from "./conditional-support";

describe("Qwen 3.5 conditional support helpers", () => {
  test("parse image grids, count tokens, and build image masks", () => {
    using imageGridThw = array(
      [
        [1, 2, 2],
        [1, 4, 4],
      ],
      "int32",
    );

    expect(gridThwList(imageGridThw, "grid test")).toEqual([
      [1, 2, 2],
      [1, 4, 4],
    ]);
    expect(
      countImageTokens(
        [
          [1, 2, 2],
          [1, 4, 4],
        ],
        2,
      ),
    ).toBe(5);
    expect(countQwen3_5ImageTokens(imageGridThw, 2)).toBe(5);

    using mask = createImageMask([7, 28, 28, 9], 28, 3);
    expect(mask.shape).toEqual([1, 4, 3]);
    expect(mask.toList()).toEqual([
      [
        [0, 0, 0],
        [1, 1, 1],
        [1, 1, 1],
        [0, 0, 0],
      ],
    ]);
  });

  test("builds multimodal position ids and derives rope deltas", () => {
    using positionIds = buildPositionIds(
      [7, 28, 28, 28, 28, 9],
      [0, 1, 1, 1, 1, 0],
      [[1, 2, 2]],
      1,
    );

    expect(positionIds.toList()).toEqual([
      [[0, 1, 1, 1, 1, 3]],
      [[0, 1, 1, 2, 2, 3]],
      [[0, 1, 2, 1, 2, 3]],
    ]);
    expect(ropeDeltas(positionIds, 6)).toEqual([-2]);

    using rank2PositionIds = array(
      [
        [0, 1, 2],
        [5, 6, 7],
      ],
      "int32",
    );
    expect(ropeDeltas(rank2PositionIds, 3)).toEqual([0, 5]);
  });

  test("expands image placeholders and derives modality ids", () => {
    using imageGridThw = array(
      [
        [1, 2, 2],
        [1, 2, 2],
      ],
      "int32",
    );

    expect(expandQwen3_5ImageTokens([7, 28, 9, 28], imageGridThw, 28, 1)).toEqual([
      7, 28, 28, 28, 28, 9, 28, 28, 28, 28,
    ]);
    expect(
      expandQwen3_5ImageTokensFromGridThw(
        [7, 28, 9, 28],
        [
          [1, 2, 2],
          [1, 2, 2],
        ],
        28,
        1,
      ),
    ).toEqual([7, 28, 28, 28, 28, 9, 28, 28, 28, 28]);
    expect(
      countQwen3_5ImageTokensFromGridThw(
        [
          [1, 2, 2],
          [1, 2, 2],
        ],
        1,
      ),
    ).toBe(8);
    expect(createQwen3_5MmTokenTypeIds([7, 28, 29, 9], 28, 29)).toEqual([0, 1, 2, 0]);
  });

  test("rejects malformed image grids and multimodal spans", () => {
    using badGridShape = array([1, 2, 2], "int32");
    expect(() => gridThwList(badGridShape, "grid test")).toThrow("expected grid_thw with shape");

    expect(() => countImageTokens([[1, 3, 2]], 2)).toThrow("must be divisible by spatialMergeSize");
    expect(() => countImageTokens([[1, 2, 2]], 0)).toThrow(
      "spatialMergeSize must be a positive integer",
    );

    using oneImageGrid = array([[1, 2, 2]], "int32");
    expect(() => expandQwen3_5ImageTokens([28, 28], oneImageGrid, 28, 1)).toThrow(
      "prompt contains more image placeholders",
    );
    expect(() => expandQwen3_5ImageTokens([7, 9], oneImageGrid, 28, 1)).toThrow(
      "prompt referenced 0",
    );

    expect(() => buildPositionIds([7, 28], [0], [[1, 2, 2]], 1)).toThrow(
      "must match token count 2",
    );
    expect(() => buildPositionIds([7, 28], [0, 3 as 0 | 1 | 2], [[1, 2, 2]], 1)).toThrow(
      "invalid modality value 3",
    );
    expect(() => buildPositionIds([28], [2], [[1, 2, 2]], 1)).toThrow(
      "video token spans are not implemented yet",
    );
    expect(() => buildPositionIds([28], [1], [[1, 2, 2]], 1)).toThrow(
      "does not match grid-derived token count 4",
    );
  });
});
