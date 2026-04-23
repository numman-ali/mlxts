import {
  add,
  concatenate,
  expandDims,
  formatShape,
  type MxArray,
  multiply,
  repeat,
  reshape,
  retainArray,
  slice,
} from "@mlxts/core";

export type GridThw = readonly [number, number, number];

type GridGeometry = Readonly<{
  frames: number;
  height: number;
  width: number;
  mergedHeight: number;
  mergedWidth: number;
  frameSize: number;
}>;

type BilinearAxis = Readonly<{ floor: number; ceil: number; delta: number }>;

type BilinearInterpolationTables = Readonly<{
  indices: [number[], number[], number[], number[]];
  weights: [number[], number[], number[], number[]];
}>;

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

export function takeSequenceSlice(
  x: MxArray,
  start: number,
  length: number,
  context: string,
): MxArray {
  const [sequenceLength, ...remainingShape] = x.shape;
  if (sequenceLength === undefined || start < 0 || length < 0 || start + length > sequenceLength) {
    throw new Error(
      `${context}: cannot slice range [${start}, ${start + length}) from shape ${formatShape(x.shape)}.`,
    );
  }
  const startIndices = new Array<number>(x.shape.length).fill(0);
  startIndices[0] = start;
  const endIndices = [start + length];
  for (const dimension of remainingShape) {
    if (dimension === undefined) {
      throw new Error(`${context}: cannot slice a tensor with undefined trailing dimensions.`);
    }
    endIndices.push(dimension);
  }
  return slice(x, startIndices, endIndices);
}

export function takeAxis1Slice(x: MxArray, axisIndex: number, context: string): MxArray {
  const [sequenceLength, axisCount, heads, headDim] = x.shape;
  if (
    sequenceLength === undefined ||
    axisCount === undefined ||
    heads === undefined ||
    headDim === undefined ||
    x.shape.length !== 4 ||
    axisIndex < 0 ||
    axisIndex >= axisCount
  ) {
    throw new Error(`${context}: expected [seq, axis, heads, dim], got ${formatShape(x.shape)}.`);
  }

  using view = slice(x, [0, axisIndex, 0, 0], [sequenceLength, axisIndex + 1, heads, headDim]);
  return reshape(view, [sequenceLength, heads, headDim]);
}

function rotateHalf(x: MxArray, context: string): MxArray {
  const lastDimension = x.shape[x.shape.length - 1];
  if (lastDimension === undefined || lastDimension % 2 !== 0) {
    throw new Error(
      `${context}: expected an even last dimension, got ${lastDimension ?? "undefined"}.`,
    );
  }

  const half = lastDimension / 2;
  const startZeros = new Array<number>(x.shape.length - 1).fill(0);
  const leadingDimensions: number[] = [];
  for (let index = 0; index < x.shape.length - 1; index += 1) {
    const dimension = x.shape[index];
    if (dimension === undefined) {
      throw new Error(
        `${context}: expected fully known leading dimensions, got ${formatShape(x.shape)}.`,
      );
    }
    leadingDimensions.push(dimension);
  }
  using left = slice(x, [...startZeros, 0], [...leadingDimensions, half]);
  using right = slice(x, [...startZeros, half], [...leadingDimensions, lastDimension]);
  using negRight = multiply(right, -1);
  return concatenate([negRight, left], x.shape.length - 1);
}

export function applyVisionRotaryPosEmb(
  queries: MxArray,
  keys: MxArray,
  cosEmbeddings: MxArray,
  sinEmbeddings: MxArray,
): { queries: MxArray; keys: MxArray } {
  using cosHeads = expandDims(cosEmbeddings, 1);
  using sinHeads = expandDims(sinEmbeddings, 1);
  using queryRotatedHalf = rotateHalf(queries, "applyVisionRotaryPosEmb");
  using keyRotatedHalf = rotateHalf(keys, "applyVisionRotaryPosEmb");
  using scaledQueries = multiply(queries, cosHeads);
  using scaledQueryRotation = multiply(queryRotatedHalf, sinHeads);
  using scaledKeys = multiply(keys, cosHeads);
  using scaledKeyRotation = multiply(keyRotatedHalf, sinHeads);
  return {
    queries: add(scaledQueries, scaledQueryRotation),
    keys: add(scaledKeys, scaledKeyRotation),
  };
}

export function gridGeometry(
  grid: GridThw,
  spatialMergeSize: number,
  context: string,
): GridGeometry {
  const [frames, height, width] = grid;
  if (height % spatialMergeSize !== 0 || width % spatialMergeSize !== 0) {
    throw new Error(
      `${context}: grid ${grid.join("x")} must be divisible by spatial merge size ${spatialMergeSize}.`,
    );
  }

  return {
    frames,
    height,
    width,
    mergedHeight: height / spatialMergeSize,
    mergedWidth: width / spatialMergeSize,
    frameSize: height * width,
  };
}

function forEachMergedPatch(
  geometry: GridGeometry,
  spatialMergeSize: number,
  visit: (frame: number, row: number, column: number) => void,
): void {
  for (let frame = 0; frame < geometry.frames; frame += 1) {
    for (let blockRow = 0; blockRow < geometry.mergedHeight; blockRow += 1) {
      const rowBase = blockRow * spatialMergeSize;
      for (let blockCol = 0; blockCol < geometry.mergedWidth; blockCol += 1) {
        const columnBase = blockCol * spatialMergeSize;
        for (let intraRow = 0; intraRow < spatialMergeSize; intraRow += 1) {
          for (let intraCol = 0; intraCol < spatialMergeSize; intraCol += 1) {
            visit(frame, rowBase + intraRow, columnBase + intraCol);
          }
        }
      }
    }
  }
}

export function rotaryPairIndices(geometry: GridGeometry, spatialMergeSize: number): number[][] {
  const pairIndices: number[][] = [];
  forEachMergedPatch(geometry, spatialMergeSize, (_frame, row, column) => {
    pairIndices.push([row, column]);
  });
  return pairIndices;
}

export function flattenPairEmbeddings(
  pairEmbeddings: MxArray,
  pairCount: number,
  context: string,
): MxArray {
  const pairAxes = pairEmbeddings.shape[1];
  const pairDim = pairEmbeddings.shape[2];
  if (pairAxes === undefined || pairDim === undefined) {
    throw new Error(
      `${context}: expected pair embeddings with shape [seq, axes, dim], got ${formatShape(pairEmbeddings.shape)}.`,
    );
  }

  return reshape(pairEmbeddings, [pairCount, pairAxes * pairDim]);
}

function bilinearAxis(index: number, size: number, gridSize: number): BilinearAxis {
  const position = size === 1 ? 0 : (index * (gridSize - 1)) / (size - 1);
  const floor = Math.floor(position);
  const ceil = Math.min(floor + 1, gridSize - 1);
  return { floor, ceil, delta: position - floor };
}

function pushBilinearEntry(
  tables: BilinearInterpolationTables,
  row: BilinearAxis,
  column: BilinearAxis,
  gridSize: number,
): void {
  tables.indices[0].push(row.floor * gridSize + column.floor);
  tables.indices[1].push(row.floor * gridSize + column.ceil);
  tables.indices[2].push(row.ceil * gridSize + column.floor);
  tables.indices[3].push(row.ceil * gridSize + column.ceil);

  tables.weights[0].push((1 - row.delta) * (1 - column.delta));
  tables.weights[1].push((1 - row.delta) * column.delta);
  tables.weights[2].push(row.delta * (1 - column.delta));
  tables.weights[3].push(row.delta * column.delta);
}

export function bilinearInterpolationTables(
  height: number,
  width: number,
  gridSize: number,
): BilinearInterpolationTables {
  const tables: BilinearInterpolationTables = {
    indices: [[], [], [], []],
    weights: [[], [], [], []],
  };

  for (let row = 0; row < height; row += 1) {
    const rowAxis = bilinearAxis(row, height, gridSize);
    for (let column = 0; column < width; column += 1) {
      pushBilinearEntry(tables, rowAxis, bilinearAxis(column, width, gridSize), gridSize);
    }
  }

  return tables;
}

export function repeatFrames(embeddings: MxArray, frames: number): MxArray {
  if (frames === 1) {
    return retainArray(embeddings);
  }
  return repeat(embeddings, frames, 0);
}

export function reorderedPatchIndices(
  grid: GridThw,
  spatialMergeSize: number,
  context: string,
): number[] {
  const geometry = gridGeometry(grid, spatialMergeSize, context);
  const indices: number[] = [];
  forEachMergedPatch(geometry, spatialMergeSize, (frame, row, column) => {
    indices.push(frame * geometry.frameSize + row * geometry.width + column);
  });

  return indices;
}
