import { afterEach, describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import { array } from "./array";
import { loadSafetensors, saveSafetensors } from "./io";

const tempDirectories: string[] = [];

afterEach(() => {
  while (tempDirectories.length > 0) {
    const directory = tempDirectories.pop();
    if (directory !== undefined) {
      rmSync(directory, { recursive: true, force: true });
    }
  }
});

function createTempPath(name: string): string {
  const directory = mkdtempSync(join(tmpdir(), "mlx-ts-io-"));
  tempDirectories.push(directory);
  return join(directory, name);
}

describe("safetensors I/O", () => {
  test("saveSafetensors and loadSafetensors round-trip tensors and metadata", async () => {
    const path = createTempPath("weights.safetensors");
    const tokenWeights = array(
      [
        [1, 2, 3],
        [4, 5, 6],
      ],
      "int32",
    );
    const bias = array([0.25, -0.5, 1.5], "float32");

    try {
      await saveSafetensors(
        {
          "token_embedding.weight": tokenWeights,
          "lm_head.bias": bias,
        },
        path,
        { source: "unit-test", format: "safetensors" },
      );

      const loaded = await loadSafetensors(path);
      try {
        const loadedWeights = loaded.tensors["token_embedding.weight"];
        const loadedBias = loaded.tensors["lm_head.bias"];

        expect(loaded.metadata).toEqual({ source: "unit-test", format: "safetensors" });
        expect(loadedWeights?.dtype).toBe("int32");
        expect(loadedWeights?.shape).toEqual([2, 3]);
        expect(loadedWeights?.toList()).toEqual([
          [1, 2, 3],
          [4, 5, 6],
        ]);
        expect(loadedBias?.dtype).toBe("float32");
        expect(loadedBias?.toList()).toEqual([0.25, -0.5, 1.5]);
      } finally {
        for (const tensor of Object.values(loaded.tensors)) {
          tensor.free();
        }
      }
    } finally {
      tokenWeights.free();
      bias.free();
    }
  });

  test("loadSafetensors rejects malformed metadata values", async () => {
    const path = createTempPath("invalid-metadata.safetensors");
    const header = {
      __metadata__: { okay: "yes", broken: 123 },
    };
    const headerBytes = new TextEncoder().encode(JSON.stringify(header));
    const prefix = new Uint8Array(8);
    new DataView(prefix.buffer).setBigUint64(0, BigInt(headerBytes.byteLength), true);
    const payload = new Uint8Array(prefix.byteLength + headerBytes.byteLength);
    payload.set(prefix, 0);
    payload.set(headerBytes, prefix.byteLength);
    await Bun.write(path, payload);

    await expect(loadSafetensors(path)).rejects.toThrow(
      'loadSafetensors: metadata value for "broken" must be a string',
    );
  });
});
