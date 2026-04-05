import { afterEach, describe, expect, test } from "bun:test";
import type { ParameterTree } from "@mlxts/core";
import { array, type MxArray, saveSafetensors } from "@mlxts/core";
import { type ResolvedSnapshot, resolveSnapshot } from "@mlxts/hub";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

import type { CausalLM, TransformerCache } from "../../types";
import type { Gemma4TextConfig } from "./types";
import { exceptionalGemma4WeightNames, loadExceptionalGemma4Weights } from "./weights";

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

function gemma4Config(overrides: Partial<Gemma4TextConfig> = {}): Gemma4TextConfig {
  return {
    family: "gemma",
    modelType: "gemma4_text",
    rawConfig: {},
    vocabSize: 32,
    vocabSizePerLayerInput: 32,
    hiddenSize: 16,
    intermediateSize: 32,
    numHiddenLayers: 2,
    numAttentionHeads: 4,
    numKeyValueHeads: 2,
    numGlobalKeyValueHeads: null,
    headDim: 4,
    globalHeadDim: 8,
    maxPositionEmbeddings: 128,
    slidingWindow: 64,
    layerTypes: ["sliding_attention", "full_attention"],
    rmsNormEps: 1e-5,
    attentionBias: false,
    tieWordEmbeddings: true,
    hiddenSizePerLayerInput: 4,
    useDoubleWideMLP: false,
    attentionKEqV: false,
    numKvSharedLayers: 0,
    slidingRopeTheta: 10_000,
    fullRopeTheta: 1_000_000,
    fullRotaryDimensions: 2,
    finalLogitSoftcapping: 30,
    embeddingScale: 4,
    ...overrides,
  };
}

function expectTensor(value: MxArray | null, message: string): MxArray {
  if (value === null) {
    throw new Error(message);
  }
  return value;
}

class DummyModel implements CausalLM {
  readonly family = "gemma";
  readonly layerCount = 0;
  readonly config = {
    family: "gemma",
    modelType: "gemma4_text",
    rawConfig: {},
    vocabSize: 32,
    hiddenSize: 16,
    numHiddenLayers: 0,
  } as const;

  forward(): MxArray {
    return array([[[0]]], "float32");
  }

  createCache(): TransformerCache {
    throw new Error("DummyModel.createCache should not be called in Gemma 4 weight tests.");
  }

  parameters(): ParameterTree {
    return {};
  }

  trainableParameters(): ParameterTree {
    return {};
  }

  update(): void {}

  freeze(): this {
    return this;
  }

  unfreeze(): this {
    return this;
  }

  eval(): this {
    return this;
  }

  train(): this {
    return this;
  }

  [Symbol.dispose](): void {}
}

async function resolveLocalSnapshot(directory: string): Promise<ResolvedSnapshot> {
  return resolveSnapshot(directory, {
    include: ["*.json", "*.safetensors"],
  });
}

describe("gemma4 exceptional weight loading", () => {
  test("reports exceptional checkpoint names only when per-layer input is enabled", async () => {
    expect(exceptionalGemma4WeightNames(gemma4Config({ hiddenSizePerLayerInput: 0 }))).toEqual([]);
    expect(exceptionalGemma4WeightNames(gemma4Config({ modelType: "gemma4_text" }))).toEqual([
      "model.embed_tokens_per_layer.weight",
    ]);
    expect(exceptionalGemma4WeightNames(gemma4Config({ modelType: "gemma4" }))).toEqual([
      "model.language_model.embed_tokens_per_layer.weight",
    ]);

    const snapshot: ResolvedSnapshot = {
      source: "local",
      repoType: "model",
      directory: "/tmp/unused",
      files: [],
    };

    let assigned = false;
    await loadExceptionalGemma4Weights({
      snapshot,
      config: gemma4Config({ hiddenSizePerLayerInput: 0 }),
      model: new DummyModel(),
      assignWeight: () => {
        assigned = true;
      },
    });

    expect(assigned).toBe(false);
  });

  test("loads the exceptional tensor directly from a single safetensors shard", async () => {
    const directory = createTempDir("mlxts-gemma4-weights-");
    using tensor = array(
      [
        [1, 2],
        [3, 4],
      ],
      "float32",
    );
    await saveSafetensors(
      {
        "model.embed_tokens_per_layer.weight": tensor,
      },
      join(directory, "model.safetensors"),
    );

    const snapshot = await resolveLocalSnapshot(directory);
    let assignedPath = "";
    let assignedTensor: MxArray | null = null;

    await loadExceptionalGemma4Weights({
      snapshot,
      config: gemma4Config({ modelType: "gemma4_text" }),
      model: new DummyModel(),
      assignWeight: (path, loadedTensor) => {
        assignedPath = path;
        assignedTensor = loadedTensor;
      },
    });

    expect(assignedPath).toBe("model.embedTokensPerLayer.weight");
    using ownedTensor = expectTensor(assignedTensor, "expected exceptional Gemma 4 tensor");
    expect(ownedTensor.toList()).toEqual([
      [1, 2],
      [3, 4],
    ]);
  });

  test("loads the exceptional tensor through the safetensors index for top-level gemma4", async () => {
    const directory = createTempDir("mlxts-gemma4-index-");
    using tensor = array(
      [
        [5, 6],
        [7, 8],
      ],
      "float32",
    );
    await saveSafetensors(
      {
        "model.language_model.embed_tokens_per_layer.weight": tensor,
      },
      join(directory, "model-00001-of-00001.safetensors"),
    );
    await Bun.write(
      join(directory, "model.safetensors.index.json"),
      JSON.stringify({
        weight_map: {
          "model.language_model.embed_tokens_per_layer.weight": "model-00001-of-00001.safetensors",
        },
      }),
    );

    const snapshot = await resolveLocalSnapshot(directory);
    let assignedTensor: MxArray | null = null;

    await loadExceptionalGemma4Weights({
      snapshot,
      config: gemma4Config({ modelType: "gemma4" }),
      model: new DummyModel(),
      assignWeight: (_path, loadedTensor) => {
        assignedTensor = loadedTensor;
      },
    });

    using ownedTensor = expectTensor(assignedTensor, "expected indexed exceptional Gemma 4 tensor");
    expect(ownedTensor.toList()).toEqual([
      [5, 6],
      [7, 8],
    ]);
  });

  test("rejects malformed safetensors index entries for exceptional Gemma 4 weights", async () => {
    const directory = createTempDir("mlxts-gemma4-bad-index-");
    await Bun.write(
      join(directory, "model.safetensors.index.json"),
      JSON.stringify({
        weight_map: {
          "model.embed_tokens_per_layer.weight": 7,
        },
      }),
    );

    const snapshot = await resolveLocalSnapshot(directory);

    await expect(
      loadExceptionalGemma4Weights({
        snapshot,
        config: gemma4Config({ modelType: "gemma4_text" }),
        model: new DummyModel(),
        assignWeight: () => {
          throw new Error("assignWeight should not be called for malformed index entries.");
        },
      }),
    ).rejects.toThrow('safetensors index entry "model.embed_tokens_per_layer.weight"');
  });

  test("rejects missing or unresolved shard mappings for exceptional Gemma 4 weights", async () => {
    const missingKeyDirectory = createTempDir("mlxts-gemma4-missing-key-");
    await Bun.write(
      join(missingKeyDirectory, "model.safetensors.index.json"),
      JSON.stringify({
        weight_map: {
          "model.other.weight": "model-00001-of-00001.safetensors",
        },
      }),
    );

    const missingKeySnapshot = await resolveLocalSnapshot(missingKeyDirectory);
    await expect(
      loadExceptionalGemma4Weights({
        snapshot: missingKeySnapshot,
        config: gemma4Config({ modelType: "gemma4_text" }),
        model: new DummyModel(),
        assignWeight: () => {
          throw new Error("assignWeight should not be called when the index entry is missing.");
        },
      }),
    ).rejects.toThrow('checkpoint index did not contain "model.embed_tokens_per_layer.weight"');

    const missingShardDirectory = createTempDir("mlxts-gemma4-missing-shard-");
    await Bun.write(
      join(missingShardDirectory, "model.safetensors.index.json"),
      JSON.stringify({
        weight_map: {
          "model.embed_tokens_per_layer.weight": "model-00001-of-00001.safetensors",
        },
      }),
    );

    const missingShardSnapshot = await resolveLocalSnapshot(missingShardDirectory);
    await expect(
      loadExceptionalGemma4Weights({
        snapshot: missingShardSnapshot,
        config: gemma4Config({ modelType: "gemma4_text" }),
        model: new DummyModel(),
        assignWeight: () => {
          throw new Error("assignWeight should not be called when the shard file is missing.");
        },
      }),
    ).rejects.toThrow('resolved snapshot is missing shard "model-00001-of-00001.safetensors"');
  });

  test("rejects ambiguous no-index snapshots with multiple safetensors shards", async () => {
    const directory = createTempDir("mlxts-gemma4-multi-shard-");
    using firstTensor = array([[1]], "float32");
    using secondTensor = array([[2]], "float32");
    await saveSafetensors(
      {
        "model.embed_tokens_per_layer.weight": firstTensor,
      },
      join(directory, "model-00001-of-00002.safetensors"),
    );
    await saveSafetensors(
      {
        "model.layers.0.mlp.down_proj.weight": secondTensor,
      },
      join(directory, "model-00002-of-00002.safetensors"),
    );

    const snapshot = await resolveLocalSnapshot(directory);
    await expect(
      loadExceptionalGemma4Weights({
        snapshot,
        config: gemma4Config({ modelType: "gemma4_text" }),
        model: new DummyModel(),
        assignWeight: () => {
          throw new Error("assignWeight should not be called for ambiguous shard layouts.");
        },
      }),
    ).rejects.toThrow('unable to resolve a shard path for "model.embed_tokens_per_layer.weight"');
  });
});
