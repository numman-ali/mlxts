import { afterEach, describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

import { array } from "./array";
import { loadGguf, parseGgufMetadataJson, saveGguf } from "./io";

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
  const directory = mkdtempSync(join(tmpdir(), "mlxts-core-gguf-"));
  tempDirectories.push(directory);
  return join(directory, name);
}

describe("gguf I/O", () => {
  test("parseGgufMetadataJson accepts scalar and array metadata values", () => {
    expect(
      parseGgufMetadataJson(
        JSON.stringify({
          string: "llama",
          number: 7,
          boolean: true,
          nullish: null,
          list: [1, 2, 3],
          nested: [
            ["a", "b"],
            [false, null],
          ],
        }),
      ),
    ).toEqual({
      string: "llama",
      number: 7,
      boolean: true,
      nullish: null,
      list: [1, 2, 3],
      nested: [
        ["a", "b"],
        [false, null],
      ],
    });
  });

  test("parseGgufMetadataJson accepts an empty payload", () => {
    expect(parseGgufMetadataJson("")).toEqual({});
  });

  test("parseGgufMetadataJson rejects non-object payloads and unsupported values", () => {
    expect(() => parseGgufMetadataJson('"not-an-object"')).toThrow(
      "loadGguf: native GGUF metadata payload must be a JSON object.",
    );
    expect(() =>
      parseGgufMetadataJson(
        JSON.stringify({
          bad: { nested: "object" },
        }),
      ),
    ).toThrow("loadGguf: native GGUF metadata payload must be a JSON object.");
  });

  test("saveGguf and loadGguf round-trip tensors and string metadata", () => {
    const path = createTempPath("weights.gguf");
    using weight = array(
      [
        [1, 2],
        [3, 4],
      ],
      "float32",
    );
    using bias = array([0.5, -1.25], "float16");

    saveGguf(
      {
        "model.layers.0.linear.weight": weight,
        "model.layers.0.linear.bias": bias,
      },
      path,
      {
        "general.architecture": "llama",
        "general.name": "unit-test",
      },
    );

    const loaded = loadGguf(path);
    try {
      const loadedWeight = loaded.tensors["model.layers.0.linear.weight"];
      const loadedBias = loaded.tensors["model.layers.0.linear.bias"];

      expect(loaded.metadata["general.architecture"]).toBe("llama");
      expect(loaded.metadata["general.name"]).toBe("unit-test");
      expect(loadedWeight?.dtype).toBe("float32");
      expect(loadedWeight?.toList()).toEqual([
        [1, 2],
        [3, 4],
      ]);
      expect(loadedBias?.dtype).toBe("float16");
      expect(loadedBias?.toList()).toEqual([0.5, -1.25]);
    } finally {
      for (const tensor of Object.values(loaded.tensors)) {
        tensor.free();
      }
    }
  });

  test("saveGguf and loadGguf support omitted metadata", () => {
    const path = createTempPath("weights-no-metadata.gguf");
    using weight = array([1, 2, 3], "float32");

    saveGguf({ weight }, path);

    const loaded = loadGguf(path);
    try {
      expect(loaded.metadata).toEqual({});
      expect(loaded.tensors.weight?.toList()).toEqual([1, 2, 3]);
    } finally {
      for (const tensor of Object.values(loaded.tensors)) {
        tensor.free();
      }
    }
  });
});
