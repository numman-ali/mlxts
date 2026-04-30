/**
 * Qwen 3.5 image-preprocessing helpers for multimodal prompts.
 * @module
 */

import { array, type MxArray } from "@mlxts/core";

import {
  expectConfigRecord,
  expectInteger,
  optionalString,
} from "../../../infrastructure/config-parsing";
import { inspectSnapshot, resolvePretrainedSnapshot } from "../../../pretrained/snapshot";
import type { LoadSourceOptions } from "../../../pretrained/types";
import { ConfigParseError } from "../../../types";
import type { Qwen3_5ImageGridThw } from "./conditional-support";

export type Qwen3_5VisionPreprocessorConfig = {
  size: {
    shortestEdge: number;
    longestEdge: number;
  };
  patchSize: number;
  temporalPatchSize: number;
  mergeSize: number;
  imageMean: readonly [number, number, number];
  imageStd: readonly [number, number, number];
  processorClass: string | null;
  imageProcessorType: string | null;
};

export type DecodedQwen3_5Image = {
  width: number;
  height: number;
  channels?: 3;
  data: readonly number[] | Uint8Array | Uint8ClampedArray | Float32Array;
};

export type PreparedQwen3_5ImageBatch = {
  pixelValues: MxArray;
  imageGridThw: MxArray;
};

type PreparedImageGrid = {
  gridT: number;
  gridH: number;
  gridW: number;
  mergedGridH: number;
  mergedGridW: number;
};

function expectPositiveInteger(value: number, context: string): number {
  if (value <= 0) {
    throw new ConfigParseError(`${context} must be positive, got ${value}.`);
  }
  return value;
}

function parseRgbTriple(
  record: Record<string, unknown>,
  key: string,
  context: string,
): [number, number, number] {
  const value = record[key];
  if (!Array.isArray(value) || value.length !== 3) {
    throw new ConfigParseError(`${context}.${key} must be an array with 3 numeric entries.`);
  }

  const [red, green, blue] = value;
  if (typeof red !== "number" || typeof green !== "number" || typeof blue !== "number") {
    throw new ConfigParseError(`${context}.${key} must contain only numbers.`);
  }
  return [red, green, blue];
}

/** Parse `preprocessor_config.json` into a typed Qwen 3.5 image preprocessor config. */
export function parseQwen3_5VisionPreprocessorConfig(
  rawConfig: Record<string, unknown>,
): Qwen3_5VisionPreprocessorConfig {
  const config = expectConfigRecord(rawConfig, "Qwen 3.5 preprocessor config");
  const size = expectConfigRecord(config.size, "Qwen 3.5 preprocessor config.size");

  return {
    size: {
      shortestEdge: expectPositiveInteger(
        expectInteger(size, "shortest_edge", "Qwen 3.5 preprocessor config.size"),
        "Qwen 3.5 preprocessor config.size.shortest_edge",
      ),
      longestEdge: expectPositiveInteger(
        expectInteger(size, "longest_edge", "Qwen 3.5 preprocessor config.size"),
        "Qwen 3.5 preprocessor config.size.longest_edge",
      ),
    },
    patchSize: expectPositiveInteger(
      expectInteger(config, "patch_size", "Qwen 3.5 preprocessor config"),
      "Qwen 3.5 preprocessor config.patch_size",
    ),
    temporalPatchSize: expectPositiveInteger(
      expectInteger(config, "temporal_patch_size", "Qwen 3.5 preprocessor config"),
      "Qwen 3.5 preprocessor config.temporal_patch_size",
    ),
    mergeSize: expectPositiveInteger(
      expectInteger(config, "merge_size", "Qwen 3.5 preprocessor config"),
      "Qwen 3.5 preprocessor config.merge_size",
    ),
    imageMean: parseRgbTriple(config, "image_mean", "Qwen 3.5 preprocessor config"),
    imageStd: parseRgbTriple(config, "image_std", "Qwen 3.5 preprocessor config"),
    processorClass:
      optionalString(config, "processor_class", "Qwen 3.5 preprocessor config") ?? null,
    imageProcessorType:
      optionalString(config, "image_processor_type", "Qwen 3.5 preprocessor config") ?? null,
  };
}

/** Load and parse the image preprocessor sidecar for a Qwen 3.5 checkpoint or MLX conversion. */
export async function loadQwen3_5VisionPreprocessor(
  source: string,
  options: LoadSourceOptions = {},
): Promise<Qwen3_5VisionPreprocessorConfig> {
  const snapshot = await resolvePretrainedSnapshot(source, options);
  const inspection = inspectSnapshot(snapshot);
  if (Object.keys(inspection.preprocessorConfig).length === 0) {
    throw new Error(
      `loadQwen3_5VisionPreprocessor: source "${source}" does not provide preprocessor_config.json.`,
    );
  }
  return parseQwen3_5VisionPreprocessorConfig(inspection.preprocessorConfig);
}

/** Match Hugging Face's Qwen smart-resize policy for image-only prompts. */
export function smartResizeQwen3_5Image(
  height: number,
  width: number,
  config: Qwen3_5VisionPreprocessorConfig,
): { height: number; width: number } {
  if (!Number.isInteger(height) || height <= 0) {
    throw new Error(`smartResizeQwen3_5Image: height must be a positive integer, got ${height}.`);
  }
  if (!Number.isInteger(width) || width <= 0) {
    throw new Error(`smartResizeQwen3_5Image: width must be a positive integer, got ${width}.`);
  }

  const aspectRatio = Math.max(height, width) / Math.min(height, width);
  if (aspectRatio > 200) {
    throw new Error(
      `smartResizeQwen3_5Image: absolute aspect ratio must be <= 200, got ${aspectRatio}.`,
    );
  }

  const factor = config.patchSize * config.mergeSize;
  let resizedHeight = Math.round(height / factor) * factor;
  let resizedWidth = Math.round(width / factor) * factor;
  const pixelCount = height * width;

  if (resizedHeight * resizedWidth > config.size.longestEdge) {
    const beta = Math.sqrt(pixelCount / config.size.longestEdge);
    resizedHeight = Math.max(factor, Math.floor(height / beta / factor) * factor);
    resizedWidth = Math.max(factor, Math.floor(width / beta / factor) * factor);
  } else if (resizedHeight * resizedWidth < config.size.shortestEdge) {
    const beta = Math.sqrt(config.size.shortestEdge / pixelCount);
    resizedHeight = Math.ceil((height * beta) / factor) * factor;
    resizedWidth = Math.ceil((width * beta) / factor) * factor;
  }

  return { height: resizedHeight, width: resizedWidth };
}

function pixelValue(
  image: DecodedQwen3_5Image,
  row: number,
  column: number,
  channel: number,
): number {
  const channels = image.channels ?? 3;
  const index = (row * image.width + column) * channels + channel;
  const value = image.data[index];
  if (value === undefined) {
    throw new Error(
      `prepareQwen3_5ImageBatch: missing pixel at row=${row}, column=${column}, channel=${channel}.`,
    );
  }
  const normalized = value > 1 ? value / 255 : value;
  return normalized;
}

function channelNormalization(
  config: Qwen3_5VisionPreprocessorConfig,
  channel: number,
): { readonly mean: number; readonly std: number } {
  switch (channel) {
    case 0:
      return { mean: config.imageMean[0], std: config.imageStd[0] };
    case 1:
      return { mean: config.imageMean[1], std: config.imageStd[1] };
    case 2:
      return { mean: config.imageMean[2], std: config.imageStd[2] };
    default:
      throw new Error(`prepareQwen3_5ImageBatch: expected RGB channel index 0-2, got ${channel}.`);
  }
}

function validateDecodedImage(
  image: DecodedQwen3_5Image,
  config: Qwen3_5VisionPreprocessorConfig,
): void {
  const channels = image.channels ?? 3;
  if (channels !== 3) {
    throw new Error(`prepareQwen3_5ImageBatch: expected 3 RGB channels, got ${channels}.`);
  }
  if (!Number.isInteger(image.height) || image.height <= 0) {
    throw new Error(`prepareQwen3_5ImageBatch: image height must be a positive integer.`);
  }
  if (!Number.isInteger(image.width) || image.width <= 0) {
    throw new Error(`prepareQwen3_5ImageBatch: image width must be a positive integer.`);
  }
  if (image.data.length !== image.height * image.width * channels) {
    throw new Error(
      `prepareQwen3_5ImageBatch: image data length ${image.data.length} does not match ${image.width}x${image.height}x${channels}.`,
    );
  }

  const factor = config.patchSize * config.mergeSize;
  if (image.height % factor !== 0 || image.width % factor !== 0) {
    throw new Error(
      `prepareQwen3_5ImageBatch: image size ${image.height}x${image.width} must be divisible by patch_size*merge_size=${factor}.`,
    );
  }
}

function normalizeImageBatchInput(
  images: DecodedQwen3_5Image | readonly DecodedQwen3_5Image[],
): DecodedQwen3_5Image[] {
  const imageList = Array.isArray(images) ? [...images] : [images];
  if (imageList.length === 0) {
    throw new Error("prepareQwen3_5ImageBatch: at least one decoded image is required.");
  }
  return imageList;
}

function createImageGrid(
  image: DecodedQwen3_5Image,
  config: Qwen3_5VisionPreprocessorConfig,
): PreparedImageGrid {
  const gridT = 1;
  const gridH = image.height / config.patchSize;
  const gridW = image.width / config.patchSize;
  return {
    gridT,
    gridH,
    gridW,
    mergedGridH: gridH / config.mergeSize,
    mergedGridW: gridW / config.mergeSize,
  };
}

function blockPixelStart(
  patchSize: number,
  mergeSize: number,
  blockIndex: number,
  mergeIndex: number,
): number {
  return (blockIndex * mergeSize + mergeIndex) * patchSize;
}

function createPatchToken(
  image: DecodedQwen3_5Image,
  config: Qwen3_5VisionPreprocessorConfig,
  blockRow: number,
  blockColumn: number,
  mergeRow: number,
  mergeColumn: number,
): number[] {
  const token: number[] = [];
  const rowStart = blockPixelStart(config.patchSize, config.mergeSize, blockRow, mergeRow);
  const columnStart = blockPixelStart(config.patchSize, config.mergeSize, blockColumn, mergeColumn);

  for (let temporalPatch = 0; temporalPatch < config.temporalPatchSize; temporalPatch += 1) {
    void temporalPatch;
    for (let patchRow = 0; patchRow < config.patchSize; patchRow += 1) {
      for (let patchColumn = 0; patchColumn < config.patchSize; patchColumn += 1) {
        const row = rowStart + patchRow;
        const column = columnStart + patchColumn;
        for (let channel = 0; channel < 3; channel += 1) {
          const { mean, std } = channelNormalization(config, channel);
          token.push((pixelValue(image, row, column, channel) - mean) / std);
        }
      }
    }
  }

  return token;
}

function appendImagePatchTokens(
  flattenedPatches: number[][],
  image: DecodedQwen3_5Image,
  config: Qwen3_5VisionPreprocessorConfig,
  grid: PreparedImageGrid,
): void {
  for (let blockRow = 0; blockRow < grid.mergedGridH; blockRow += 1) {
    for (let blockColumn = 0; blockColumn < grid.mergedGridW; blockColumn += 1) {
      for (let mergeRow = 0; mergeRow < config.mergeSize; mergeRow += 1) {
        for (let mergeColumn = 0; mergeColumn < config.mergeSize; mergeColumn += 1) {
          flattenedPatches.push(
            createPatchToken(image, config, blockRow, blockColumn, mergeRow, mergeColumn),
          );
        }
      }
    }
  }
}

function appendPreparedImage(
  flattenedPatches: number[][],
  grids: number[][],
  image: DecodedQwen3_5Image,
  config: Qwen3_5VisionPreprocessorConfig,
): void {
  validateDecodedImage(image, config);
  const grid = createImageGrid(image, config);
  grids.push([grid.gridT, grid.gridH, grid.gridW]);
  appendImagePatchTokens(flattenedPatches, image, config, grid);
}

/** Return Qwen image grid metadata for already-decoded RGB image data. */
export function qwen3_5ImageGridThwValues(
  images: DecodedQwen3_5Image | readonly DecodedQwen3_5Image[],
  config: Qwen3_5VisionPreprocessorConfig,
): Qwen3_5ImageGridThw[] {
  return normalizeImageBatchInput(images).map((image) => {
    validateDecodedImage(image, config);
    const grid = createImageGrid(image, config);
    return [grid.gridT, grid.gridH, grid.gridW];
  });
}

/** Patchify already-decoded RGB image data into Qwen 3.5 `pixel_values` and `image_grid_thw`. */
export function prepareQwen3_5ImageBatch(
  images: DecodedQwen3_5Image | readonly DecodedQwen3_5Image[],
  config: Qwen3_5VisionPreprocessorConfig,
): PreparedQwen3_5ImageBatch {
  const imageList = normalizeImageBatchInput(images);
  const flattenedPatches: number[][] = [];
  const grids: number[][] = [];

  for (const image of imageList) {
    appendPreparedImage(flattenedPatches, grids, image, config);
  }

  return {
    pixelValues: array(flattenedPatches, "float32"),
    imageGridThw: array(grids, "int32"),
  };
}
