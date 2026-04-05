import { afterEach, describe, expect, test } from "bun:test";
import { array, saveSafetensors } from "@mlxts/core";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

import { resolveSnapshot } from "./snapshot";
import { iterateSafetensorWeights, loadSafetensorShardSet } from "./weights";

const tempRoots: string[] = [];

function createTempDir(prefix: string): string {
  const directory = mkdtempSync(join(tmpdir(), prefix));
  tempRoots.push(directory);
  return directory;
}

afterEach(() => {
  while (tempRoots.length > 0) {
    const directory = tempRoots.pop();
    if (directory !== undefined) {
      rmSync(directory, { recursive: true, force: true });
    }
  }
});

describe("safetensor shard loading", () => {
  test("iterates and merges safetensor shards", async () => {
    const directory = createTempDir("mlxts-hub-weights-");
    await saveSafetensors(
      {
        "model.embed_tokens.weight": array(
          [
            [1, 2],
            [3, 4],
          ],
          "float32",
        ),
      },
      join(directory, "model-00001-of-00002.safetensors"),
      { family: "llama" },
    );
    await saveSafetensors(
      {
        "lm_head.weight": array(
          [
            [5, 6],
            [7, 8],
          ],
          "float32",
        ),
      },
      join(directory, "model-00002-of-00002.safetensors"),
      { family: "llama" },
    );
    await Bun.write(
      join(directory, "model.safetensors.index.json"),
      JSON.stringify({
        metadata: { total_size: "32" },
        weight_map: {
          "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
          "lm_head.weight": "model-00002-of-00002.safetensors",
        },
      }),
    );

    const snapshot = await resolveSnapshot(directory, {
      include: ["model.safetensors.index.json", "model-*.safetensors"],
    });

    const shardNames: string[] = [];
    const iteratedTensorNames: string[] = [];
    for await (const entry of iterateSafetensorWeights(snapshot)) {
      shardNames.push(entry.shardPath.split("/").at(-1) ?? entry.shardPath);
      iteratedTensorNames.push(entry.name);
      entry.tensor.free();
    }

    expect(shardNames).toEqual([
      "model-00001-of-00002.safetensors",
      "model-00002-of-00002.safetensors",
    ]);
    expect(iteratedTensorNames).toEqual(["model.embed_tokens.weight", "lm_head.weight"]);

    const merged = await loadSafetensorShardSet(snapshot);
    try {
      expect(Object.keys(merged.tensors)).toEqual(["model.embed_tokens.weight", "lm_head.weight"]);
      expect(merged.metadata.family).toBe("llama");
      expect(merged.shardPaths.map((path) => path.split("/").at(-1))).toEqual([
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
      ]);
      expect(merged.tensors["lm_head.weight"]?.toList()).toEqual([
        [5, 6],
        [7, 8],
      ]);
    } finally {
      for (const tensor of Object.values(merged.tensors)) {
        tensor.free();
      }
    }
  });

  test("falls back to direct shard paths and validates safetensor indexes", async () => {
    const directory = createTempDir("mlxts-hub-weights-direct-");
    const shardPath = join(directory, "model.safetensors");
    await saveSafetensors(
      {
        "model.embed_tokens.weight": array(
          [
            [1, 2],
            [3, 4],
          ],
          "float32",
        ),
      },
      shardPath,
      { family: "llama" },
    );

    const directSnapshot = await resolveSnapshot(directory, { include: ["*.safetensors"] });
    const directWeights: string[] = [];
    for await (const entry of iterateSafetensorWeights(directSnapshot)) {
      directWeights.push(entry.name);
      entry.tensor.free();
    }
    expect(directWeights).toEqual(["model.embed_tokens.weight"]);

    await Bun.write(
      join(directory, "model.safetensors.index.json"),
      JSON.stringify({
        weight_map: {
          "model.embed_tokens.weight": 123,
        },
      }),
    );
    const invalidIndexSnapshot = await resolveSnapshot(directory, {
      include: ["model.safetensors.index.json", "*.safetensors"],
    });
    await expect(loadSafetensorShardSet(invalidIndexSnapshot)).rejects.toThrow(
      'inspectSnapshot: safetensors index entry "model.embed_tokens.weight" must point to a shard filename',
    );

    await Bun.write(
      join(directory, "model.safetensors.index.json"),
      JSON.stringify({
        weight_map: {
          "model.embed_tokens.weight": "missing.safetensors",
        },
      }),
    );
    const missingShardSnapshot = await resolveSnapshot(directory, {
      include: ["model.safetensors.index.json", "*.safetensors"],
    });
    await expect(loadSafetensorShardSet(missingShardSnapshot)).rejects.toThrow(
      'iterateSafetensorWeights: missing shard "missing.safetensors" from resolved snapshot',
    );
  });

  test("iterateSafetensorWeights can skip tensors before loading them", async () => {
    const directory = createTempDir("mlxts-hub-weights-filter-");
    await saveSafetensors(
      {
        "model.embed_tokens.weight": array(
          [
            [1, 2],
            [3, 4],
          ],
          "float32",
        ),
        "lm_head.weight": array(
          [
            [5, 6],
            [7, 8],
          ],
          "float32",
        ),
      },
      join(directory, "model.safetensors"),
      { family: "llama" },
    );

    const snapshot = await resolveSnapshot(directory, { include: ["*.safetensors"] });
    const iteratedTensorNames: string[] = [];
    for await (const entry of iterateSafetensorWeights(snapshot, {
      include: (name) => name === "lm_head.weight",
    })) {
      iteratedTensorNames.push(entry.name);
      expect(entry.tensor.toList()).toEqual([
        [5, 6],
        [7, 8],
      ]);
      entry.tensor.free();
    }

    expect(iteratedTensorNames).toEqual(["lm_head.weight"]);
  });
});
