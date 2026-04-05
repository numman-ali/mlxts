import { afterEach, describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

import { inspectSnapshot } from "./inspect";
import { resolveSnapshot } from "./snapshot";

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

describe("inspectSnapshot", () => {
  test("parses config, tokenizer files, and safetensors shard indexes", async () => {
    const directory = createTempDir("mlxts-hub-config-");
    await Bun.write(
      join(directory, "config.json"),
      JSON.stringify({ model_type: "llama", hidden_size: 16 }),
    );
    await Bun.write(join(directory, "generation_config.json"), JSON.stringify({ max_length: 20 }));
    await Bun.write(
      join(directory, "tokenizer.json"),
      JSON.stringify({ model: { type: "BPE", vocab: { hello: 0 }, merges: [] } }),
    );
    await Bun.write(join(directory, "tokenizer_config.json"), JSON.stringify({ bos_token: "<s>" }));
    await Bun.write(
      join(directory, "special_tokens_map.json"),
      JSON.stringify({ eos_token: { content: "</s>" } }),
    );
    await Bun.write(
      join(directory, "model.safetensors.index.json"),
      JSON.stringify({
        metadata: { total_size: "16" },
        weight_map: {
          "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
          "model.layers.0.mlp.gate_proj.weight": "model-00002-of-00002.safetensors",
        },
      }),
    );
    await Bun.write(join(directory, "model-00001-of-00002.safetensors"), new Uint8Array([1]));
    await Bun.write(join(directory, "model-00002-of-00002.safetensors"), new Uint8Array([2]));

    const snapshot = await resolveSnapshot(directory, {
      include: [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "model.safetensors.index.json",
        "model-*.safetensors",
      ],
    });
    const artifacts = inspectSnapshot(snapshot);

    expect(artifacts.config.model_type).toBe("llama");
    expect(artifacts.generationConfig.max_length).toBe(20);
    expect(artifacts.tokenizer.tokenizerJsonPath?.endsWith("tokenizer.json")).toBe(true);
    const weightMap = artifacts.safetensorsIndex.weight_map;
    if (typeof weightMap !== "object" || weightMap === null || Array.isArray(weightMap)) {
      throw new Error("Expected safetensors weight map");
    }
    const normalizedWeightMap = Object.fromEntries(Object.entries(weightMap));
    const embedTensor = normalizedWeightMap["model.embed_tokens.weight"];
    expect(embedTensor).toBe("model-00001-of-00002.safetensors");
    expect(artifacts.model.safetensorPaths.map((path) => path.split("/").at(-1))).toEqual([
      "model-00001-of-00002.safetensors",
      "model-00002-of-00002.safetensors",
    ]);
  });
});
