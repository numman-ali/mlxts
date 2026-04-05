import { afterEach, describe, expect, test } from "bun:test";
import { array, saveSafetensors } from "@mlxts/core";
import { mkdtempSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

import { resolvePretrainedSnapshot } from "./snapshot";
import {
  iterateSafetensorWeights,
  listSafetensorShardPaths,
  parseSafetensorIndex,
} from "./weights";

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

describe("pretrained weight helpers", () => {
  test("parseSafetensorIndex returns null without a weight map and rejects non-string shard names", () => {
    expect(parseSafetensorIndex({})).toBeNull();
    expect(() => parseSafetensorIndex({ weight_map: { broken: 123 } })).toThrow(
      'inspectSnapshot: safetensors index entry "broken" must point to a shard filename.',
    );
  });

  test("listSafetensorShardPaths falls back to the discovered safetensors files without an index", async () => {
    const directory = createTempDir("mlxts-transformers-weights-no-index-");
    writeFileSync(join(directory, "config.json"), JSON.stringify({ model_type: "llama" }));
    using first = array([[1, 2]], "float32");
    using second = array([[3, 4]], "float32");
    await saveSafetensors({ first }, join(directory, "model-00002-of-00002.safetensors"));
    await saveSafetensors({ second }, join(directory, "model-00001-of-00002.safetensors"));

    const snapshot = await resolvePretrainedSnapshot(directory);
    expect(listSafetensorShardPaths(snapshot)).toEqual([
      join(directory, "model-00001-of-00002.safetensors"),
      join(directory, "model-00002-of-00002.safetensors"),
    ]);
  });

  test("listSafetensorShardPaths respects the shard index and rejects unresolved shard references", async () => {
    const directory = createTempDir("mlxts-transformers-weights-index-");
    writeFileSync(join(directory, "config.json"), JSON.stringify({ model_type: "llama" }));
    writeFileSync(
      join(directory, "model.safetensors.index.json"),
      JSON.stringify({
        weight_map: {
          "model.layers.0.weight": "model-00002-of-00002.safetensors",
          "model.layers.1.weight": "model-00001-of-00002.safetensors",
          "model.layers.2.weight": "model-00001-of-00002.safetensors",
        },
      }),
    );
    using shard = array([[1, 2]], "float32");
    await saveSafetensors({ shard }, join(directory, "model-00001-of-00002.safetensors"));
    await saveSafetensors({ shard }, join(directory, "model-00002-of-00002.safetensors"));

    const snapshot = await resolvePretrainedSnapshot(directory);
    expect(listSafetensorShardPaths(snapshot)).toEqual([
      join(directory, "model-00001-of-00002.safetensors"),
      join(directory, "model-00002-of-00002.safetensors"),
    ]);

    writeFileSync(
      join(directory, "model.safetensors.index.json"),
      JSON.stringify({ weight_map: { broken: "missing.safetensors" } }),
    );
    const brokenSnapshot = await resolvePretrainedSnapshot(directory);
    expect(() => listSafetensorShardPaths(brokenSnapshot)).toThrow(
      'iterateSafetensorWeights: missing shard "missing.safetensors" from resolved snapshot.',
    );
  });

  test("iterateSafetensorWeights yields shard paths and respects include filters", async () => {
    const directory = createTempDir("mlxts-transformers-weights-iterate-");
    writeFileSync(join(directory, "config.json"), JSON.stringify({ model_type: "llama" }));
    using first = array([[1, 2]], "float32");
    using second = array([[3, 4]], "float32");
    await saveSafetensors({ alpha: first }, join(directory, "model-00001-of-00002.safetensors"));
    await saveSafetensors({ beta: second }, join(directory, "model-00002-of-00002.safetensors"));

    const snapshot = await resolvePretrainedSnapshot(directory);
    const seen: string[] = [];

    for await (const { name, tensor, shardPath } of iterateSafetensorWeights(snapshot, {
      include: (weightName, currentShardPath) =>
        weightName === "beta" && currentShardPath.endsWith("00002.safetensors"),
    })) {
      seen.push(`${name}@${shardPath.split("/").at(-1)}`);
      tensor.free();
    }

    expect(seen).toEqual(["beta@model-00002-of-00002.safetensors"]);
  });
});
