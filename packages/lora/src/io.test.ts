import { afterEach, describe, expect, test } from "bun:test";
import { array, saveSafetensors } from "@mlxts/core";
import { Linear, LoRALinear, Module } from "@mlxts/nn";
import { mkdtempSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

import { applyLoRAToModule } from "./apply-module";
import { loadLoRAAdapters, saveLoRAAdapters } from "./io";

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

class AttentionBlock extends Module {
  qProjection = new Linear(64, 64);

  forward(): never {
    throw new Error("unused");
  }
}

class ModelBody extends Module {
  layers = [new AttentionBlock(), new AttentionBlock()];

  forward(): never {
    throw new Error("unused");
  }
}

class TinyModel extends Module {
  model = new ModelBody();

  forward(): never {
    throw new Error("unused");
  }
}

describe("LoRA adapter I/O", () => {
  test("saves and reloads adapter wrappers and weights", async () => {
    const directory = createTempDir("mlxts-lora-");
    using sourceModel = new TinyModel();
    using targetModel = new TinyModel();

    applyLoRAToModule(sourceModel, {
      paths: ["model.layers.1.qProjection"],
      rank: 4,
      alpha: 12,
      dropout: 0.1,
    });
    await saveLoRAAdapters(sourceModel, directory);
    await loadLoRAAdapters(targetModel, directory);

    expect(targetModel.model.layers[1]?.qProjection).toBeInstanceOf(LoRALinear);
    const loaded = targetModel.model.layers[1]?.qProjection;
    if (!(loaded instanceof LoRALinear)) {
      throw new Error("expected LoRALinear after loading adapters");
    }
    expect(loaded.rank).toBe(4);
    expect(loaded.alpha).toBe(12);
    expect(loaded.dropoutProbability).toBeCloseTo(0.1, 6);
  });

  test("rejects saves when no LoRA wrappers are active", async () => {
    const directory = createTempDir("mlxts-lora-empty-");
    using model = new TinyModel();

    await expect(saveLoRAAdapters(model, directory)).rejects.toThrow(
      "found no active LoRA wrappers",
    );
  });

  test("rejects malformed adapter configs before loading tensors", async () => {
    const directory = createTempDir("mlxts-lora-invalid-");
    writeFileSync(
      join(directory, "adapter_config.json"),
      JSON.stringify({ format: "wrong", version: 1, targets: [] }),
    );

    using model = new TinyModel();
    await expect(loadLoRAAdapters(model, directory)).rejects.toThrow('format must be "mlxts-lora"');
  });

  test("rejects invalid adapter target fields", async () => {
    const directory = createTempDir("mlxts-lora-invalid-target-");
    writeFileSync(
      join(directory, "adapter_config.json"),
      JSON.stringify({
        format: "mlxts-lora",
        version: 1,
        targets: [{ path: "", rank: 0, alpha: "bad", dropout: 2 }],
      }),
    );

    using model = new TinyModel();
    await expect(loadLoRAAdapters(model, directory)).rejects.toThrow(
      "path must be a non-empty string",
    );
  });

  test("rejects unexpected adapter tensors", async () => {
    const directory = createTempDir("mlxts-lora-unexpected-");
    writeFileSync(
      join(directory, "adapter_config.json"),
      JSON.stringify({
        format: "mlxts-lora",
        version: 1,
        targets: [{ path: "model.layers.1.qProjection", rank: 4, alpha: 8, dropout: 0 }],
      }),
    );
    using extra = array([1, 2, 3, 4], "float32");
    await saveSafetensors(
      { "model.layers.1.qProjection.extra": extra },
      join(directory, "adapters.safetensors"),
    );

    using model = new TinyModel();
    await expect(loadLoRAAdapters(model, directory)).rejects.toThrow(
      'unexpected adapter tensor "model.layers.1.qProjection.extra"',
    );
  });

  test("rejects missing adapter tensors", async () => {
    const directory = createTempDir("mlxts-lora-missing-");
    writeFileSync(
      join(directory, "adapter_config.json"),
      JSON.stringify({
        format: "mlxts-lora",
        version: 1,
        targets: [{ path: "model.layers.1.qProjection", rank: 4, alpha: 8, dropout: 0 }],
      }),
    );
    using onlyA = array(new Float32Array(64 * 4), "float32");
    await saveSafetensors(
      { "model.layers.1.qProjection.loraA": onlyA },
      join(directory, "adapters.safetensors"),
    );

    using model = new TinyModel();
    await expect(loadLoRAAdapters(model, directory)).rejects.toThrow(
      'missing adapter tensor "model.layers.1.qProjection.loraB"',
    );
  });
});
