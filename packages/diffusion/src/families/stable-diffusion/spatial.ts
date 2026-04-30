/**
 * Stable Diffusion spatial tensor helpers.
 * @module
 */

import type { MxArray } from "@mlxts/core";
import { formatShape, pad, repeat } from "@mlxts/core";

function assertRank4Image(x: MxArray, helperName: string): void {
  if (x.shape.length !== 4) {
    throw new Error(`${helperName}: expected rank-4 NHWC input, got ${formatShape(x.shape)}.`);
  }
}

/** Nearest-neighbor upsampling for Stable Diffusion NHWC feature maps. */
export function upsampleNearest2d(x: MxArray, scale = 2): MxArray {
  assertRank4Image(x, "upsampleNearest2d");
  if (!Number.isInteger(scale) || scale <= 0) {
    throw new Error(`upsampleNearest2d: scale must be a positive integer, got ${scale}.`);
  }
  using heightRepeated = repeat(x, scale, 1);
  return repeat(heightRepeated, scale, 2);
}

/** Bottom/right zero padding used before Stable Diffusion stride-2 downsample convolutions. */
export function padBottomRight2d(x: MxArray, bottom = 1, right = 1, value = 0): MxArray {
  assertRank4Image(x, "padBottomRight2d");
  if (!Number.isInteger(bottom) || bottom < 0 || !Number.isInteger(right) || right < 0) {
    throw new Error("padBottomRight2d: bottom and right pads must be non-negative integers.");
  }
  return pad(
    x,
    [
      [0, 0],
      [0, bottom],
      [0, right],
      [0, 0],
    ],
    value,
  );
}
