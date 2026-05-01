import { describe, expect, test } from "bun:test";
import { array, zeros } from "@mlxts/core";
import { mkdtempSync, readFileSync, rmSync } from "fs";
import { join } from "path";

import { stableDiffusion3ImageToBmpBytes, writeStableDiffusion3Bmp } from "./image-output";

function temporaryDirectory(): string {
  return mkdtempSync(join(import.meta.dir, ".tmp-bmp-"));
}

describe("Stable Diffusion 3 image output", () => {
  test("encodes a single NHWC RGB tensor as a 24-bit BMP", () => {
    using image = array(
      [
        [
          [
            [1, 0, 0],
            [0, 1, 0],
          ],
        ],
      ],
      "float32",
    );

    const bytes = stableDiffusion3ImageToBmpBytes(image);
    const view = new DataView(bytes.buffer);

    expect(new TextDecoder().decode(bytes.slice(0, 2))).toBe("BM");
    expect(view.getUint32(2, true)).toBe(bytes.byteLength);
    expect(view.getUint32(10, true)).toBe(54);
    expect(view.getInt32(18, true)).toBe(2);
    expect(view.getInt32(22, true)).toBe(1);
    expect(view.getUint16(28, true)).toBe(24);
    expect(Array.from(bytes.slice(54, 62))).toEqual([0, 0, 255, 0, 255, 0, 0, 0]);
  });

  test("writes BMP files and creates parent directories", () => {
    const directory = temporaryDirectory();
    try {
      using image = array(
        [
          [
            [
              [0, 0, 1],
              [1, 0, 0],
            ],
          ],
        ],
        "float32",
      );
      const outputPath = join(directory, "nested", "sample.bmp");

      const result = writeStableDiffusion3Bmp(image, outputPath);

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

  test("rejects batch and output format shapes outside the proof contract", () => {
    using batched = zeros([2, 1, 1, 3]);
    using grayscale = zeros([1, 1, 1, 1]);

    expect(() => stableDiffusion3ImageToBmpBytes(batched)).toThrow("one NHWC RGB image");
    expect(() => stableDiffusion3ImageToBmpBytes(grayscale)).toThrow("one NHWC RGB image");
    expect(() => writeStableDiffusion3Bmp(grayscale, "sample.png")).toThrow(".bmp");
  });
});
