import { afterEach, describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from "fs";
import { join } from "path";

import {
  estimateModelLoadMemory,
  MODEL_LOAD_MEMORY_HEADROOM,
  requireModelLoadMemoryBudget,
} from "./memory-preflight";

const tempDirectories: string[] = [];

function tempDirectory(): string {
  const directory = mkdtempSync(join(Bun.env.TMPDIR ?? "/tmp", "mlxts-serve-memory-preflight-"));
  tempDirectories.push(directory);
  return directory;
}

function writeBytes(path: string, bytes: number): void {
  writeFileSync(path, new Uint8Array(bytes));
}

afterEach(() => {
  for (const directory of tempDirectories.splice(0)) {
    rmSync(directory, { recursive: true, force: true });
  }
});

describe("model load memory preflight", () => {
  test("estimates local safetensor files with serving headroom", () => {
    const directory = tempDirectory();
    mkdirSync(join(directory, "shards"));
    writeBytes(join(directory, "model-00001.safetensors"), 100);
    writeBytes(join(directory, "shards", "model-00002.safetensors"), 60);
    writeBytes(join(directory, "tokenizer.json"), 500);

    expect(estimateModelLoadMemory(directory)).toEqual({
      safetensorBytes: 160,
      estimatedBytes: Math.ceil(160 * MODEL_LOAD_MEMORY_HEADROOM),
    });
  });

  test("counts symlinked safetensor files in Hugging Face snapshots", () => {
    const directory = tempDirectory();
    mkdirSync(join(directory, "blobs"));
    mkdirSync(join(directory, "snapshots"));
    writeBytes(join(directory, "blobs", "model-blob"), 128);
    symlinkSync("../blobs/model-blob", join(directory, "snapshots", "model.safetensors"));

    expect(estimateModelLoadMemory(join(directory, "snapshots"))).toEqual({
      safetensorBytes: 128,
      estimatedBytes: Math.ceil(128 * MODEL_LOAD_MEMORY_HEADROOM),
    });
    expect(() =>
      requireModelLoadMemoryBudget({
        modelId: "symlinked",
        source: join(directory, "snapshots"),
        gpuMemoryUtilization: 0.9,
        memory: { activeBytes: 800, cacheBytes: 0, peakBytes: 0, limitBytes: 1000 },
      }),
    ).toThrow('Model "symlinked" is estimated to need');
  });

  test("skips budget checks when telemetry or safetensor files are unavailable", () => {
    const directory = tempDirectory();
    writeBytes(join(directory, "tokenizer.json"), 500);

    expect(() =>
      requireModelLoadMemoryBudget({
        modelId: "tiny",
        source: directory,
        gpuMemoryUtilization: 0.5,
      }),
    ).not.toThrow();
    expect(() =>
      requireModelLoadMemoryBudget({
        modelId: "tiny",
        source: directory,
        gpuMemoryUtilization: 0.5,
        memory: { activeBytes: 0, cacheBytes: 0, peakBytes: 0, limitBytes: 100 },
      }),
    ).not.toThrow();
  });

  test("accepts estimates inside the active MLX memory budget", () => {
    const directory = tempDirectory();
    writeBytes(join(directory, "model.safetensors"), 100);

    expect(() =>
      requireModelLoadMemoryBudget({
        modelId: "tiny",
        source: directory,
        gpuMemoryUtilization: 0.9,
        memory: { activeBytes: 700, cacheBytes: 0, peakBytes: 0, limitBytes: 1000 },
      }),
    ).not.toThrow();
  });

  test("rejects estimates that exceed the active MLX memory budget", () => {
    const directory = tempDirectory();
    writeBytes(join(directory, "model.safetensors"), 100);

    expect(() =>
      requireModelLoadMemoryBudget({
        modelId: "too-large",
        source: directory,
        gpuMemoryUtilization: 0.9,
        memory: { activeBytes: 850, cacheBytes: 0, peakBytes: 0, limitBytes: 1000 },
      }),
    ).toThrow('Model "too-large" is estimated to need');
  });
});
