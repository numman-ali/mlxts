import { afterEach, describe, expect, test } from "bun:test";
import { array } from "@mlxts/core";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

import { loadGgufCheckpoint, saveGgufCheckpoint } from "./gguf";

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
  const directory = mkdtempSync(join(tmpdir(), "mlxts-quantize-gguf-"));
  tempDirectories.push(directory);
  return join(directory, name);
}

describe("gguf checkpoint helpers", () => {
  test("loadGgufCheckpoint round-trips through the core GGUF bridge", () => {
    const path = createTempPath("fixture.gguf");
    using weight = array(
      [
        [1, 2],
        [3, 4],
      ],
      "float32",
    );

    saveGgufCheckpoint({ "layer.weight": weight }, path, {
      "general.architecture": "llama",
    });

    const loaded = loadGgufCheckpoint(path);
    try {
      expect(loaded.metadata["general.architecture"]).toBe("llama");
      expect(loaded.tensors["layer.weight"]?.toList()).toEqual([
        [1, 2],
        [3, 4],
      ]);
    } finally {
      for (const tensor of Object.values(loaded.tensors)) {
        tensor.free();
      }
    }
  });
});
