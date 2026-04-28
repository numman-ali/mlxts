import {
  array,
  broadcastTo,
  expandDims,
  formatShape,
  type MxArray,
  max,
  reshape,
} from "@mlxts/core";

type GridThw = readonly [number, number, number];

export function gridThwList(gridThw: MxArray, context: string): GridThw[] {
  const [rows, columns] = gridThw.shape;
  if (rows === undefined || columns !== 3 || gridThw.shape.length !== 2) {
    throw new Error(
      `${context}: expected grid_thw with shape [count, 3], got ${formatShape(gridThw.shape)}.`,
    );
  }

  const values = gridThw.toList();
  if (!Array.isArray(values)) {
    throw new Error(`${context}: expected grid_thw to decode to a row-major array.`);
  }

  return values.map((row, index) => {
    if (!Array.isArray(row) || row.length !== 3) {
      throw new Error(`${context}: row ${index} is not a [t, h, w] triple.`);
    }

    const [t, h, w] = row;
    if (
      typeof t !== "number" ||
      typeof h !== "number" ||
      typeof w !== "number" ||
      !Number.isInteger(t) ||
      !Number.isInteger(h) ||
      !Number.isInteger(w) ||
      t <= 0 ||
      h <= 0 ||
      w <= 0
    ) {
      throw new Error(`${context}: row ${index} must contain positive integer [t, h, w] values.`);
    }

    return [t, h, w] as const;
  });
}

export function countImageTokens(grids: readonly GridThw[], spatialMergeSize: number): number {
  if (!Number.isInteger(spatialMergeSize) || spatialMergeSize <= 0) {
    throw new Error(
      `countImageTokens: spatialMergeSize must be a positive integer, got ${spatialMergeSize}.`,
    );
  }

  return grids.reduce((total, [frames, height, width], index) => {
    if (height % spatialMergeSize !== 0 || width % spatialMergeSize !== 0) {
      throw new Error(
        `countImageTokens: grid ${index} with shape ${frames}x${height}x${width} must be divisible by spatialMergeSize=${spatialMergeSize}.`,
      );
    }
    return total + (frames * height * width) / spatialMergeSize ** 2;
  }, 0);
}

function mergedDimension(size: number, spatialMergeSize: number, context: string): number {
  if (size % spatialMergeSize !== 0) {
    throw new Error(
      `${context}: size ${size} must be divisible by spatialMergeSize=${spatialMergeSize}.`,
    );
  }
  return size / spatialMergeSize;
}

export function createImageMask(
  tokenIds: readonly number[],
  imageTokenId: number,
  hiddenSize: number,
): MxArray {
  const mask = tokenIds.map((tokenId) => (tokenId === imageTokenId ? 1 : 0));
  using maskArray = array([mask], "bool");
  using maskView = expandDims(maskArray, 2);
  return broadcastTo(maskView, [1, tokenIds.length, hiddenSize]);
}

function visionPositionIds(
  startPosition: number,
  grid: GridThw,
  spatialMergeSize: number,
): [number[], number[], number[]] {
  const [frames, height, width] = grid;
  const mergedHeight = mergedDimension(height, spatialMergeSize, "visionPositionIds");
  const mergedWidth = mergedDimension(width, spatialMergeSize, "visionPositionIds");
  const temporal: number[] = [];
  const rows: number[] = [];
  const columns: number[] = [];

  for (let frame = 0; frame < frames; frame += 1) {
    for (let row = 0; row < mergedHeight; row += 1) {
      for (let column = 0; column < mergedWidth; column += 1) {
        temporal.push(startPosition + frame);
        rows.push(startPosition + row);
        columns.push(startPosition + column);
      }
    }
  }

  return [temporal, rows, columns];
}

function expectModality(mmTokenTypeIds: readonly number[], cursor: number): 0 | 1 | 2 {
  const modality = mmTokenTypeIds[cursor];
  if (modality !== 0 && modality !== 1 && modality !== 2) {
    throw new Error(`buildPositionIds: token ${cursor} has invalid modality value ${modality}.`);
  }
  return modality;
}

function modalitySpanEnd(
  mmTokenTypeIds: readonly number[],
  cursor: number,
  modality: 0 | 1 | 2,
): number {
  let end = cursor + 1;
  while (end < mmTokenTypeIds.length && mmTokenTypeIds[end] === modality) {
    end += 1;
  }
  return end;
}

function appendTextPositionSpan(
  positions: [number[], number[], number[]],
  startPosition: number,
  tokenCount: number,
): number {
  for (let index = 0; index < tokenCount; index += 1) {
    const position = startPosition + index;
    positions[0].push(position);
    positions[1].push(position);
    positions[2].push(position);
  }
  return startPosition + tokenCount;
}

function appendImagePositionSpan(
  positions: [number[], number[], number[]],
  startPosition: number,
  imageSpanLength: number,
  grids: readonly GridThw[],
  imageIndex: number,
  spatialMergeSize: number,
): { nextPosition: number; nextImageIndex: number } {
  const grid = grids[imageIndex];
  if (grid === undefined) {
    throw new Error(
      "buildPositionIds: mmTokenTypeIds references more images than image_grid_thw provides.",
    );
  }

  const [temporal, rows, columns] = visionPositionIds(startPosition, grid, spatialMergeSize);
  if (temporal.length !== imageSpanLength) {
    throw new Error(
      `buildPositionIds: image token span length ${imageSpanLength} does not match grid-derived token count ${temporal.length}.`,
    );
  }

  positions[0].push(...temporal);
  positions[1].push(...rows);
  positions[2].push(...columns);

  return {
    nextPosition: startPosition + Math.max(grid[1], grid[2]) / spatialMergeSize,
    nextImageIndex: imageIndex + 1,
  };
}

export function buildPositionIds(
  tokenIds: readonly number[],
  mmTokenTypeIds: readonly number[],
  grids: readonly GridThw[],
  spatialMergeSize: number,
): MxArray {
  if (mmTokenTypeIds.length !== tokenIds.length) {
    throw new Error(
      `buildPositionIds: mmTokenTypeIds length ${mmTokenTypeIds.length} must match token count ${tokenIds.length}.`,
    );
  }

  const positions: [number[], number[], number[]] = [[], [], []];
  let currentPosition = 0;
  let cursor = 0;
  let imageIndex = 0;

  while (cursor < mmTokenTypeIds.length) {
    const modality = expectModality(mmTokenTypeIds, cursor);
    const end = modalitySpanEnd(mmTokenTypeIds, cursor, modality);

    if (modality === 0) {
      currentPosition = appendTextPositionSpan(positions, currentPosition, end - cursor);
    } else if (modality === 1) {
      const nextState = appendImagePositionSpan(
        positions,
        currentPosition,
        end - cursor,
        grids,
        imageIndex,
        spatialMergeSize,
      );
      currentPosition = nextState.nextPosition;
      imageIndex = nextState.nextImageIndex;
    } else {
      throw new Error("buildPositionIds: video token spans are not implemented yet.");
    }

    cursor = end;
  }

  if (imageIndex !== grids.length) {
    throw new Error(
      `buildPositionIds: image_grid_thw contains ${grids.length} image grids but the prompt consumed ${imageIndex}.`,
    );
  }

  using stacked = array(positions, "int32");
  return reshape(stacked, [3, 1, tokenIds.length]);
}

export function ropeDeltas(positionIds: MxArray, sequenceLength: number): number[] {
  const rank = positionIds.shape.length;
  const batchSize = rank === 2 ? positionIds.shape[0] : positionIds.shape[1];
  if (batchSize === undefined) {
    throw new Error(
      `ropeDeltas: could not determine batch size from position ids with shape ${formatShape(positionIds.shape)}.`,
    );
  }

  const reduced = rank === 2 ? max(positionIds, 1) : max(positionIds, [0, 2]);
  using batchMaxima = reduced;
  const values = batchMaxima.toList();
  if (!Array.isArray(values) || values.length !== batchSize) {
    throw new Error(
      `ropeDeltas: expected ${batchSize} per-batch maxima, got ${JSON.stringify(values)}.`,
    );
  }
  return values.map((value, batchIndex) => {
    if (typeof value !== "number") {
      throw new Error(`ropeDeltas: batch ${batchIndex} maximum must be numeric.`);
    }
    return value + 1 - sequenceLength;
  });
}

/** Create modality type ids directly from prompt token ids. */
export function createQwen3_5MmTokenTypeIds(
  tokenIds: readonly number[],
  imageTokenId: number,
  videoTokenId: number,
): number[] {
  return tokenIds.map((tokenId) => {
    if (tokenId === imageTokenId) {
      return 1;
    }
    if (tokenId === videoTokenId) {
      return 2;
    }
    return 0;
  });
}

/** Expand one image placeholder token per image into the repeated visual token span the model expects. */
export function expandQwen3_5ImageTokens(
  tokenIds: readonly number[],
  imageGridThw: MxArray,
  imageTokenId: number,
  spatialMergeSize: number,
): number[] {
  const grids = gridThwList(imageGridThw, "expandQwen3_5ImageTokens");
  const expanded: number[] = [];
  let imageIndex = 0;

  for (const tokenId of tokenIds) {
    if (tokenId !== imageTokenId) {
      expanded.push(tokenId);
      continue;
    }

    const grid = grids[imageIndex];
    if (grid === undefined) {
      throw new Error(
        "expandQwen3_5ImageTokens: prompt contains more image placeholders than image_grid_thw provides.",
      );
    }

    const repeatedCount = countImageTokens([grid], spatialMergeSize);
    for (let index = 0; index < repeatedCount; index += 1) {
      expanded.push(imageTokenId);
    }
    imageIndex += 1;
  }

  if (imageIndex !== grids.length) {
    throw new Error(
      `expandQwen3_5ImageTokens: image_grid_thw contains ${grids.length} image grids but the prompt referenced ${imageIndex}.`,
    );
  }

  return expanded;
}

/** Count how many repeated image placeholder tokens the current image grids require. */
export function countQwen3_5ImageTokens(imageGridThw: MxArray, spatialMergeSize: number): number {
  return countImageTokens(gridThwList(imageGridThw, "countQwen3_5ImageTokens"), spatialMergeSize);
}
