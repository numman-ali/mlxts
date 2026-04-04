import { describe, expect, test } from "bun:test";
import { mkdtempSync, writeFileSync } from "fs";
import { mxEval } from "mlx-ts";
import { tmpdir } from "os";
import { join } from "path";
import { createRandomSource, getBatch, loadText, prepareData } from "./data";

describe("data", () => {
  test("prepareData splits tokens into train/val", () => {
    const tokens = Array.from({ length: 100 }, (_, i) => i);
    const { trainTokens, valTokens } = prepareData(tokens, 0.9);
    expect(trainTokens.length).toBe(90);
    expect(valTokens.length).toBe(10);
  });

  test("getBatch returns correct shapes", () => {
    const tokens = new Int32Array(Array.from({ length: 200 }, (_, i) => i % 65));
    const { input, target } = getBatch(tokens, 4, 16);
    mxEval(input, target);
    expect(input.shape).toEqual([4, 16]);
    expect(target.shape).toEqual([4, 16]);
    expect(input.dtype).toBe("int32");
    expect(target.dtype).toBe("int32");
    input.free();
    target.free();
  });

  test("target is input shifted by 1", () => {
    // Use sequential tokens so we can verify the shift
    const tokens = new Int32Array(Array.from({ length: 50 }, (_, i) => i));
    const { input, target } = getBatch(tokens, 1, 10);
    mxEval(input, target);
    const inputList = input.toTypedArray();
    const targetList = target.toTypedArray();
    // Each target[i] should be input[i] + 1
    for (let i = 0; i < 10; i++) {
      expect(targetList[i]).toBe((inputList[i] ?? 0) + 1);
    }
    input.free();
    target.free();
  });

  test("getBatch is deterministic with a seeded random source", () => {
    const tokens = new Int32Array(Array.from({ length: 50 }, (_, i) => i));
    const rngA = createRandomSource(42);
    const rngB = createRandomSource(42);
    const first = getBatch(tokens, 2, 8, rngA);
    const second = getBatch(tokens, 2, 8, rngB);
    try {
      mxEval(first.input, first.target, second.input, second.target);
      expect(first.input.toList()).toEqual(second.input.toList());
      expect(first.target.toList()).toEqual(second.target.toList());
    } finally {
      first.input.free();
      first.target.free();
      second.input.free();
      second.target.free();
    }
  });

  test("prepareData rejects invalid trainSplit", () => {
    expect(() => prepareData([1, 2, 3], 0)).toThrow("trainSplit");
    expect(() => prepareData([1, 2, 3], 1)).toThrow("trainSplit");
  });

  test("getBatch rejects invalid shapes", () => {
    const tokens = new Int32Array(Array.from({ length: 20 }, (_, i) => i));
    expect(() => getBatch(tokens, 0, 4)).toThrow("batchSize");
    expect(() => getBatch(tokens, 1, 0)).toThrow("blockSize");
    expect(() => getBatch(tokens, 1, 20)).toThrow("too short");
  });

  test("loadText prefers an explicit local path", async () => {
    const directory = mkdtempSync(join(tmpdir(), "nanogpt-data-"));
    const filePath = join(directory, "sample.txt");
    writeFileSync(filePath, "to be or not to be", "utf-8");
    await expect(loadText({ path: filePath })).resolves.toBe("to be or not to be");
  });

  test("loadText uses the cached file when present", async () => {
    const cacheDir = mkdtempSync(join(tmpdir(), "nanogpt-cache-"));
    writeFileSync(join(cacheDir, "shakespeare.txt"), "cached text", "utf-8");
    await expect(loadText({ cacheDir })).resolves.toBe("cached text");
  });

  test("loadText downloads and caches when no local source exists", async () => {
    const cacheDir = mkdtempSync(join(tmpdir(), "nanogpt-download-"));
    const originalFetch = globalThis.fetch;
    const mockFetch: typeof fetch = Object.assign(
      async (_input: string | URL | Request, _init?: RequestInit): Promise<Response> =>
        new Response("downloaded text", { status: 200 }),
      { preconnect: originalFetch.preconnect },
    );
    globalThis.fetch = mockFetch;

    try {
      await expect(loadText({ cacheDir })).resolves.toBe("downloaded text");
      await expect(loadText({ cacheDir })).resolves.toBe("downloaded text");
    } finally {
      globalThis.fetch = originalFetch;
    }
  });
});
