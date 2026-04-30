import { afterEach, describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from "fs";
import { basename, join } from "path";

import { discoverLocalModelSources } from "./discovery";

const temporaryDirectories: string[] = [];

function createTemporaryDirectory(): string {
  const directory = mkdtempSync(join(Bun.env.TMPDIR ?? "/tmp", "mlxts-serve-discovery-"));
  temporaryDirectories.push(directory);
  return directory;
}

function writeCheckpoint(directory: string, config: Record<string, unknown>): void {
  mkdirSync(directory, { recursive: true });
  writeFileSync(join(directory, "config.json"), JSON.stringify(config));
  writeFileSync(join(directory, "model.safetensors"), "weights");
}

afterEach(() => {
  for (const directory of temporaryDirectories.splice(0)) {
    rmSync(directory, { recursive: true, force: true });
  }
});

describe("discoverLocalModelSources", () => {
  test("discovers root, flat, and org/model checkpoints", () => {
    const root = createTemporaryDirectory();
    writeCheckpoint(root, {
      model_type: "llama",
      architectures: ["LlamaForCausalLM", 7],
    });
    writeCheckpoint(join(root, "flat"), {
      model_type: "gemma",
      architectures: ["GemmaForCausalLM"],
    });
    writeCheckpoint(join(root, "org", "qwen"), {
      model_type: "qwen3_5",
      architectures: ["Qwen3_5ForConditionalGeneration"],
      vision_config: { image_size: 448 },
    });
    writeCheckpoint(join(root, "unsupported"), { model_type: "clip_vision_model" });
    writeCheckpoint(join(root, "missing-type"), { architectures: ["NoModelType"] });
    writeCheckpoint(join(root, ".cache", "hidden"), { model_type: "hidden" });
    writeCheckpoint(join(root, "too", "deep", "model"), { model_type: "deep" });
    mkdirSync(join(root, "org", "no-weights"), { recursive: true });
    writeFileSync(join(root, "org", "no-weights", "config.json"), '{"model_type":"skip"}');

    expect(discoverLocalModelSources(root)).toEqual([
      {
        source: root,
        modelId: basename(root),
        modelType: "llama",
        architectures: ["LlamaForCausalLM"],
        hasVisionConfig: false,
      },
      {
        source: join(root, "flat"),
        modelId: "flat",
        modelType: "gemma",
        architectures: ["GemmaForCausalLM"],
        hasVisionConfig: false,
      },
      {
        source: join(root, "org", "qwen"),
        modelId: "org/qwen",
        modelType: "qwen3_5",
        architectures: ["Qwen3_5ForConditionalGeneration"],
        hasVisionConfig: true,
      },
    ]);
  });

  test("skips checkpoint-shaped directories outside supported autoregressive model types", () => {
    const root = createTemporaryDirectory();
    writeCheckpoint(join(root, "diffusion", "vae"), { model_type: "vae" });
    writeCheckpoint(join(root, "vision", "clip"), { model_type: "clip_vision_model" });
    writeCheckpoint(join(root, "audio"), { model_type: "whisper" });

    expect(discoverLocalModelSources(root)).toEqual([]);
  });

  test("accepts safetensor symlinks from cache-style layouts", () => {
    const root = createTemporaryDirectory();
    const blob = join(root, "blob");
    const model = join(root, "org", "linked");
    mkdirSync(model, { recursive: true });
    writeFileSync(blob, "weights");
    writeFileSync(join(model, "config.json"), JSON.stringify({ model_type: "llama" }));
    symlinkSync(blob, join(model, "model.safetensors"));

    expect(discoverLocalModelSources(root)).toEqual([
      {
        source: model,
        modelId: "org/linked",
        modelType: "llama",
        architectures: [],
        hasVisionConfig: false,
      },
    ]);
  });

  test("rejects missing roots and malformed checkpoint configs", () => {
    const root = createTemporaryDirectory();
    expect(() => discoverLocalModelSources(join(root, "missing"))).toThrow('Model root "');

    const model = join(root, "broken");
    mkdirSync(model, { recursive: true });
    writeFileSync(join(model, "config.json"), "{");
    writeFileSync(join(model, "model.safetensors"), "weights");

    expect(() => discoverLocalModelSources(root)).toThrow("is not valid JSON");
  });

  test("ignores malformed non-checkpoint config files", () => {
    const root = createTemporaryDirectory();
    const directory = join(root, "metadata");
    mkdirSync(directory, { recursive: true });
    writeFileSync(join(directory, "config.json"), "{");

    expect(discoverLocalModelSources(root)).toEqual([]);
  });
});
