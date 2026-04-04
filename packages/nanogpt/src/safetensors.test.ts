import { afterEach, describe, expect, test } from "bun:test";
import { CharTokenizer } from "@mlxts/tokenizers";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import { GPT_TINY, resolveConfig } from "./config";
import { GPT } from "./model/gpt";
import { initializeGPT } from "./model/init";
import { loadModelSafetensors, saveModelSafetensors } from "./safetensors";

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
  const directory = mkdtempSync(join(tmpdir(), "nanogpt-safetensors-"));
  tempDirectories.push(directory);
  return join(directory, name);
}

function createTinyModel(vocabSize = 27): GPT {
  const config = resolveConfig(
    { ...GPT_TINY, nLayer: 2, nHead: 2, nEmbd: 32, blockSize: 16, dropout: 0 },
    vocabSize,
  );
  const model = new GPT(config);
  initializeGPT(model, config);
  return model;
}

describe("model safetensors interop", () => {
  test("saveModelSafetensors and loadModelSafetensors round-trip model weights", async () => {
    const tokenizer = CharTokenizer.fromText("to be or not to be");
    const path = createTempPath("model.safetensors");
    const source = createTinyModel(tokenizer.vocabSize);
    const target = createTinyModel(tokenizer.vocabSize);

    try {
      await saveModelSafetensors(source, path, { source: "unit-test" });
      const metadata = await loadModelSafetensors(target, path);

      expect(metadata.metadata).toEqual({ source: "unit-test" });
      expect(source.tokenEmbedding.weight.toList()).toEqual(target.tokenEmbedding.weight.toList());
      expect(source.positionEmbedding.weight.toList()).toEqual(
        target.positionEmbedding.weight.toList(),
      );
    } finally {
      source[Symbol.dispose]();
      target[Symbol.dispose]();
    }
  });

  test("loadModelSafetensors rejects incompatible model parameter shapes", async () => {
    const source = createTinyModel(27);
    const target = createTinyModel(28);
    const path = createTempPath("mismatch.safetensors");

    try {
      await saveModelSafetensors(source, path);
      await expect(loadModelSafetensors(target, path)).rejects.toThrow(
        'loadModelSafetensors: shape mismatch for "tokenEmbedding.weight"',
      );
    } finally {
      source[Symbol.dispose]();
      target[Symbol.dispose]();
    }
  });
});
