import { describe, expect, test } from "bun:test";
import { array, zeros } from "@mlxts/core";
import { mkdtempSync, rmSync, writeFileSync } from "fs";
import { join } from "path";

import { verifyImageProofArtifact, writeImageProofBmp } from "./artifact";

function temporaryDirectory(): string {
  return mkdtempSync(join(import.meta.dir, ".tmp-artifact-"));
}

describe("image proof artifact", () => {
  test("writes BMP evidence with artifact integrity checks", () => {
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
      const artifact = writeImageProofBmp(image, join(directory, "sample.bmp"), {
        label: "test image",
      });

      expect(artifact.status).toBe("passed");
      expect(artifact.sha256).toHaveLength(64);
      expect(artifact.width).toBe(2);
      expect(artifact.height).toBe(1);
      expect(artifact.bytes).toBe(62);
      expect(Object.values(verifyImageProofArtifact(artifact)).every(Boolean)).toBe(true);
    } finally {
      rmSync(directory, { force: true, recursive: true });
    }
  });

  test("rejects uniform proof artifacts", () => {
    const directory = temporaryDirectory();
    try {
      using image = zeros([1, 2, 2, 3]);

      expect(() =>
        writeImageProofBmp(image, join(directory, "uniform.bmp"), { label: "uniform image" }),
      ).toThrow("failed checks");
    } finally {
      rmSync(directory, { force: true, recursive: true });
    }
  });

  test("detects truncated BMP files during verification", () => {
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
      const outputPath = join(directory, "sample.bmp");
      const artifact = writeImageProofBmp(image, outputPath, { label: "test image" });
      writeFileSync(outputPath, new Uint8Array([66, 77, 0]));

      const checks = verifyImageProofArtifact(artifact);

      expect(checks.bmpHeaderValid).toBe(false);
      expect(checks.byteLengthMatches).toBe(false);
      expect(checks.sha256Present).toBe(false);
    } finally {
      rmSync(directory, { force: true, recursive: true });
    }
  });
});
