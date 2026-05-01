import { describe, expect, test } from "bun:test";
import { array, zeros } from "@mlxts/core";
import { mkdtempSync, readFileSync, rmSync } from "fs";
import { join } from "path";

import { ltxVideoPreviewFrameToBmpBytes, writeLtxVideoPreviewBmp } from "./video-output";

function temporaryDirectory(): string {
  return mkdtempSync(join(import.meta.dir, ".tmp-bmp-"));
}

describe("LTX-Video preview output", () => {
  test("encodes sampled BFHWC frames as a 24-bit BMP sheet", () => {
    using video = array(
      [
        [
          [
            [
              [1, 0, 0],
              [0, 1, 0],
            ],
          ],
          [
            [
              [0, 0, 1],
              [1, 1, 0],
            ],
          ],
          [
            [
              [1, 0, 1],
              [0, 1, 1],
            ],
          ],
        ],
      ],
      "float32",
    );

    const bytes = ltxVideoPreviewFrameToBmpBytes(video);
    const view = new DataView(bytes.buffer);

    expect(new TextDecoder().decode(bytes.slice(0, 2))).toBe("BM");
    expect(view.getUint32(2, true)).toBe(bytes.byteLength);
    expect(view.getUint32(10, true)).toBe(54);
    expect(view.getInt32(18, true)).toBe(6);
    expect(view.getInt32(22, true)).toBe(1);
    expect(view.getUint16(28, true)).toBe(24);
  });

  test("writes BMP preview files and creates parent directories", () => {
    const directory = temporaryDirectory();
    try {
      using video = array(
        [
          [
            [
              [
                [0, 0, 1],
                [1, 0, 0],
              ],
            ],
          ],
        ],
        "float32",
      );
      const outputPath = join(directory, "nested", "preview.bmp");

      const result = writeLtxVideoPreviewBmp(video, outputPath);

      expect(result.path).toBe(outputPath);
      expect(result.width).toBe(2);
      expect(result.height).toBe(1);
      expect(result.bytes).toBe(62);
      expect(result.sha256).toHaveLength(64);
      expect(result.status).toBe("passed");
      expect(readFileSync(outputPath).byteLength).toBe(62);
    } finally {
      rmSync(directory, { force: true, recursive: true });
    }
  });

  test("rejects shapes outside the preview proof contract", () => {
    using batched = zeros([2, 1, 1, 1, 3]);
    using grayscale = zeros([1, 1, 1, 1, 1]);
    using image = zeros([1, 1, 1, 3]);

    expect(() => ltxVideoPreviewFrameToBmpBytes(batched)).toThrow("one BFHWC RGB video");
    expect(() => ltxVideoPreviewFrameToBmpBytes(grayscale)).toThrow("one BFHWC RGB video");
    expect(() => ltxVideoPreviewFrameToBmpBytes(image)).toThrow("one BFHWC RGB video");
    expect(() => writeLtxVideoPreviewBmp(grayscale, "sample.png")).toThrow("BFHWC");
  });
});
