import { describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

import {
  loadQwen3_5VisionPreprocessor,
  parseQwen3_5VisionPreprocessorConfig,
  prepareQwen3_5ImageBatch,
  qwen3_5ImageGridThwValues,
  smartResizeQwen3_5Image,
} from "./preprocessing";

function createTempDir(prefix: string): string {
  const directory = mkdtempSync(join(tmpdir(), prefix));
  mkdirSync(directory, { recursive: true });
  return directory;
}

describe("Qwen 3.5 preprocessing", () => {
  test("parses and loads preprocessor sidecars", async () => {
    const directory = createTempDir("mlxts-qwen3_5-preprocessor-");
    const rawConfig = {
      size: { shortest_edge: 3136, longest_edge: 50176 },
      patch_size: 16,
      temporal_patch_size: 2,
      merge_size: 2,
      image_mean: [0.5, 0.5, 0.5],
      image_std: [0.5, 0.5, 0.5],
      processor_class: "Qwen3VLProcessor",
      image_processor_type: "Qwen2VLImageProcessorFast",
    };
    writeFileSync(join(directory, "config.json"), JSON.stringify({ model_type: "qwen3_5" }));
    writeFileSync(join(directory, "preprocessor_config.json"), JSON.stringify(rawConfig));

    expect(parseQwen3_5VisionPreprocessorConfig(rawConfig)).toEqual({
      size: { shortestEdge: 3136, longestEdge: 50176 },
      patchSize: 16,
      temporalPatchSize: 2,
      mergeSize: 2,
      imageMean: [0.5, 0.5, 0.5],
      imageStd: [0.5, 0.5, 0.5],
      processorClass: "Qwen3VLProcessor",
      imageProcessorType: "Qwen2VLImageProcessorFast",
    });

    await expect(loadQwen3_5VisionPreprocessor(directory)).resolves.toMatchObject({
      patchSize: 16,
      temporalPatchSize: 2,
      mergeSize: 2,
    });
  });

  test("matches Qwen smart-resize rules", () => {
    const config = parseQwen3_5VisionPreprocessorConfig({
      size: { shortest_edge: 3136, longest_edge: 50176 },
      patch_size: 16,
      temporal_patch_size: 2,
      merge_size: 2,
      image_mean: [0.5, 0.5, 0.5],
      image_std: [0.5, 0.5, 0.5],
    });

    expect(smartResizeQwen3_5Image(32, 32, config)).toEqual({ height: 64, width: 64 });
    expect(smartResizeQwen3_5Image(224, 448, config)).toEqual({ height: 128, width: 288 });
  });

  test("patchifies decoded RGB images into Qwen pixel_values and grid metadata", () => {
    const config = parseQwen3_5VisionPreprocessorConfig({
      size: { shortest_edge: 16, longest_edge: 16 },
      patch_size: 2,
      temporal_patch_size: 2,
      merge_size: 1,
      image_mean: [0, 0, 0],
      image_std: [1, 1, 1],
    });

    const prepared = prepareQwen3_5ImageBatch(
      {
        width: 2,
        height: 2,
        data: [0.1, 0.01, 0.001, 0.2, 0.02, 0.002, 0.3, 0.03, 0.003, 0.4, 0.04, 0.004],
      },
      config,
    );

    try {
      expect(prepared.pixelValues.shape).toEqual([1, 24]);
      expect(prepared.imageGridThw.toList()).toEqual([[1, 1, 1]]);
      expect(
        qwen3_5ImageGridThwValues(
          {
            width: 2,
            height: 2,
            data: [0.1, 0.01, 0.001, 0.2, 0.02, 0.002, 0.3, 0.03, 0.003, 0.4, 0.04, 0.004],
          },
          config,
        ),
      ).toEqual([[1, 1, 1]]);
      expect(prepared.pixelValues.toList()).toEqual([
        [
          expect.closeTo(0.1, 6),
          expect.closeTo(0.01, 6),
          expect.closeTo(0.001, 6),
          expect.closeTo(0.2, 6),
          expect.closeTo(0.02, 6),
          expect.closeTo(0.002, 6),
          expect.closeTo(0.3, 6),
          expect.closeTo(0.03, 6),
          expect.closeTo(0.003, 6),
          expect.closeTo(0.4, 6),
          expect.closeTo(0.04, 6),
          expect.closeTo(0.004, 6),
          expect.closeTo(0.1, 6),
          expect.closeTo(0.01, 6),
          expect.closeTo(0.001, 6),
          expect.closeTo(0.2, 6),
          expect.closeTo(0.02, 6),
          expect.closeTo(0.002, 6),
          expect.closeTo(0.3, 6),
          expect.closeTo(0.03, 6),
          expect.closeTo(0.003, 6),
          expect.closeTo(0.4, 6),
          expect.closeTo(0.04, 6),
          expect.closeTo(0.004, 6),
        ],
      ]);
    } finally {
      prepared.pixelValues.free();
      prepared.imageGridThw.free();
    }
  });
});
