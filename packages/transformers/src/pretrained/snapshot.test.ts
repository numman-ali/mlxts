import { afterEach, describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, statSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

import { inspectSnapshot, resolvePretrainedSnapshot, resolvePretrainedSource } from "./snapshot";
import type { PretrainedLoadProgressEvent } from "./types";

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

describe("resolvePretrainedSource", () => {
  test("returns an existing local directory and emits resolve progress", async () => {
    const directory = createTempDir("mlxts-transformers-snapshot-");
    writeFileSync(join(directory, "config.json"), JSON.stringify({ model_type: "llama" }));
    writeFileSync(join(directory, "tokenizer.json"), JSON.stringify({ version: "1.0" }));
    writeFileSync(join(directory, "model.safetensors"), new Uint8Array([1, 2, 3]));
    const totalBytes =
      statSync(join(directory, "config.json")).size +
      statSync(join(directory, "tokenizer.json")).size +
      statSync(join(directory, "model.safetensors")).size;

    const events: PretrainedLoadProgressEvent[] = [];
    const resolved = await resolvePretrainedSource(directory, {
      onProgress: (event) => {
        events.push(event);
      },
    });

    expect(resolved).toBe(directory);
    expect(events).toEqual([
      { stage: "resolve", status: "start", source: directory },
      {
        stage: "resolve",
        status: "complete",
        sourceKind: "local",
        directory,
        fileCount: 3,
        totalBytes,
      },
    ]);
  });

  test("ignores hidden cache directories when scanning a local snapshot", async () => {
    const directory = createTempDir("mlxts-transformers-snapshot-hidden-");
    mkdirSync(join(directory, ".cache", "huggingface"), { recursive: true });
    writeFileSync(join(directory, ".cache", "huggingface", "token"), "ignored");
    writeFileSync(join(directory, "config.json"), JSON.stringify({ model_type: "llama" }));
    writeFileSync(join(directory, "model.safetensors"), new Uint8Array([1, 2, 3]));

    const snapshot = await resolvePretrainedSnapshot(directory);
    expect(snapshot.files.map((file) => file.relativePath)).toEqual([
      "config.json",
      "model.safetensors",
    ]);
  });

  test("rejects a missing local directory", async () => {
    const directory = join(createTempDir("mlxts-transformers-missing-"), "missing");
    await expect(resolvePretrainedSnapshot(directory)).rejects.toThrow(
      `resolvePretrainedSource: local path "${directory}" does not exist.`,
    );
  });

  test("rejects sources that are neither local directories nor repo ids", async () => {
    await expect(resolvePretrainedSnapshot("gemma4")).rejects.toThrow(
      'resolvePretrainedSource: "gemma4" is neither a local directory nor a Hugging Face model id.',
    );
  });
});

describe("inspectSnapshot", () => {
  test("collects model, tokenizer, and chat-template artifacts from a local directory", async () => {
    const directory = createTempDir("mlxts-transformers-inspect-");
    writeFileSync(join(directory, "config.json"), JSON.stringify({ model_type: "llama" }));
    writeFileSync(join(directory, "generation_config.json"), JSON.stringify({ temperature: 0.7 }));
    writeFileSync(join(directory, "processor_config.json"), JSON.stringify({ chat_template: "x" }));
    writeFileSync(join(directory, "preprocessor_config.json"), JSON.stringify({ patch_size: 16 }));
    writeFileSync(
      join(directory, "video_preprocessor_config.json"),
      JSON.stringify({ temporal_patch_size: 2 }),
    );
    writeFileSync(join(directory, "tokenizer.json"), JSON.stringify({ version: "1.0" }));
    writeFileSync(join(directory, "vocab.json"), JSON.stringify({ "<|endoftext|>": 0 }));
    writeFileSync(join(directory, "merges.txt"), "#version: 0.2\n");
    writeFileSync(join(directory, "tokenizer_config.json"), JSON.stringify({ bos_token: "<s>" }));
    writeFileSync(
      join(directory, "special_tokens_map.json"),
      JSON.stringify({ eos_token: { content: "</s>" } }),
    );
    writeFileSync(
      join(directory, "chat_template.jinja"),
      "{{ bos_token }}{{ messages[0]['content'] }}",
    );
    writeFileSync(join(directory, "model.safetensors"), new Uint8Array([1, 2, 3]));

    const snapshot = await resolvePretrainedSnapshot(directory);
    const inspection = inspectSnapshot(snapshot);

    expect(inspection.model.configPath).toBe(join(directory, "config.json"));
    expect(inspection.model.generationConfigPath).toBe(join(directory, "generation_config.json"));
    expect(inspection.model.processorConfigPath).toBe(join(directory, "processor_config.json"));
    expect(inspection.model.preprocessorConfigPath).toBe(
      join(directory, "preprocessor_config.json"),
    );
    expect(inspection.model.videoPreprocessorConfigPath).toBe(
      join(directory, "video_preprocessor_config.json"),
    );
    expect(inspection.model.chatTemplatePath).toBe(join(directory, "chat_template.jinja"));
    expect(inspection.tokenizer.tokenizerJsonPath).toBe(join(directory, "tokenizer.json"));
    expect(inspection.tokenizer.vocabJsonPath).toBe(join(directory, "vocab.json"));
    expect(inspection.tokenizer.mergesTextPath).toBe(join(directory, "merges.txt"));
    expect(inspection.tokenizer.tokenizerConfigPath).toBe(join(directory, "tokenizer_config.json"));
    expect(inspection.config.model_type).toBe("llama");
    expect(inspection.generationConfig.temperature).toBe(0.7);
    expect(inspection.processorConfig.chat_template).toBe("x");
    expect(inspection.preprocessorConfig.patch_size).toBe(16);
    expect(inspection.videoPreprocessorConfig.temporal_patch_size).toBe(2);
    expect(inspection.tokenizerConfig.bos_token).toBe("<s>");
    expect(inspection.specialTokensMap.eos_token).toEqual({ content: "</s>" });
  });

  test("collects tekken and safetensor-index artifacts and rejects non-object JSON sidecars", async () => {
    const directory = createTempDir("mlxts-transformers-inspect-tekken-");
    writeFileSync(join(directory, "config.json"), JSON.stringify({ model_type: "mistral3" }));
    writeFileSync(join(directory, "tekken.json"), JSON.stringify({ version: "1.0" }));
    writeFileSync(
      join(directory, "model.safetensors.index.json"),
      JSON.stringify({
        weight_map: { "model.embed_tokens.weight": "model-00001-of-00002.safetensors" },
      }),
    );
    writeFileSync(join(directory, "model-00001-of-00002.safetensors"), new Uint8Array([1]));
    writeFileSync(join(directory, "tokenizer_config.json"), JSON.stringify(["invalid"]));

    const snapshot = await resolvePretrainedSnapshot(directory);
    expect(() => inspectSnapshot(snapshot)).toThrow(
      `inspectSnapshot: expected "${join(directory, "tokenizer_config.json")}" to contain a JSON object.`,
    );
  });
});
