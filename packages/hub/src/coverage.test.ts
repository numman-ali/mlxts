import { afterEach, describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join, resolve } from "path";

import { parseGgufHeader } from "./gguf";
import { fetchFile, fetchJson, resolveFileUrl } from "./http";
import { inspectSnapshot } from "./inspect";
import {
  defaultCacheDir,
  ensureDirectory,
  expandHome,
  localSnapshotFiles,
  refsDir,
  repoCacheDir,
  snapshotDir,
} from "./paths";
import { shouldIncludePath } from "./patterns";

const tempRoots: string[] = [];

function createTempDir(prefix: string): string {
  const directory = mkdtempSync(join(tmpdir(), prefix));
  tempRoots.push(directory);
  return directory;
}

function withFetchStub(
  originalFetch: typeof fetch,
  impl: (input: URL | RequestInfo, init?: RequestInit | BunFetchRequestInit) => Promise<Response>,
): typeof fetch {
  return Object.assign(impl, {
    preconnect: originalFetch.preconnect.bind(originalFetch),
  });
}

afterEach(() => {
  while (tempRoots.length > 0) {
    const directory = tempRoots.pop();
    if (directory !== undefined) {
      rmSync(directory, { recursive: true, force: true });
    }
  }
});

function pushU8(bytes: number[], value: number): void {
  bytes.push(value & 0xff);
}

function pushU32(bytes: number[], value: number): void {
  const buffer = new Uint8Array(4);
  new DataView(buffer.buffer).setUint32(0, value, true);
  bytes.push(...buffer);
}

function pushU64(bytes: number[], value: bigint): void {
  const buffer = new Uint8Array(8);
  new DataView(buffer.buffer).setBigUint64(0, value, true);
  bytes.push(...buffer);
}

function pushF64(bytes: number[], value: number): void {
  const buffer = new Uint8Array(8);
  new DataView(buffer.buffer).setFloat64(0, value, true);
  bytes.push(...buffer);
}

function pushString(bytes: number[], value: string): void {
  const encoded = new TextEncoder().encode(value);
  pushU64(bytes, BigInt(encoded.byteLength));
  bytes.push(...encoded);
}

describe("hub coverage", () => {
  test("path helpers expand cache locations, walk local snapshots, and filter patterns", async () => {
    const home = Bun.env.HOME;
    Bun.env.HOME = "/tmp/mlxts-home";

    const root = createTempDir("mlxts-hub-paths-");
    const nested = join(root, "nested");
    ensureDirectory(nested);
    writeFileSync(join(root, "config.json"), "{}");
    writeFileSync(join(nested, "model.safetensors"), "weights");

    try {
      expect(expandHome("~/cache")).toBe("/tmp/mlxts-home/cache");
      expect(expandHome("~")).toBe("/tmp/mlxts-home");
      expect(defaultCacheDir("~/.cache/mlxts")).toBe(resolve("/tmp/mlxts-home/.cache/mlxts"));
      expect(repoCacheDir("/cache", "model", "meta-llama/Llama-3.2-1B")).toBe(
        "/cache/models--meta-llama--Llama-3.2-1B",
      );
      expect(snapshotDir("/cache", "dataset", "openai/gsm8k", "abc123")).toBe(
        "/cache/datasets--openai--gsm8k/snapshots/abc123",
      );
      expect(refsDir("/cache", "model", "mlx-community/Llama-3.2-1B")).toBe(
        "/cache/models--mlx-community--Llama-3.2-1B/refs",
      );
      expect(localSnapshotFiles(root).map((file) => file.relativePath)).toEqual([
        "config.json",
        "nested/model.safetensors",
      ]);

      const filePath = join(root, "config.json");
      expect(() => localSnapshotFiles(join(root, "missing"))).toThrow("does not exist");
      expect(() => localSnapshotFiles(filePath)).toThrow("must be a directory");

      expect(shouldIncludePath("config.json", ["tokenizer.json"], [], [])).toBe(false);
      expect(
        shouldIncludePath(
          "model-00001-of-00002.safetensors",
          undefined,
          ["model*.safetensors"],
          [],
        ),
      ).toBe(true);
      expect(
        shouldIncludePath(
          "model-00001-of-00002.safetensors",
          undefined,
          ["model*.safetensors"],
          ["model-00001*"],
        ),
      ).toBe(false);
      expect(shouldIncludePath("model-a.safetensors", undefined, ["model-?.safetensors"], [])).toBe(
        true,
      );
      expect(shouldIncludePath("notes.txt", undefined, ["model-*.safetensors"], [])).toBe(false);
      expect(shouldIncludePath("config.json", undefined, [], [])).toBe(true);
    } finally {
      if (home === undefined) {
        delete Bun.env.HOME;
      } else {
        Bun.env.HOME = home;
      }
    }
  });

  test("inspectSnapshot collects JSON artifacts and rejects malformed JSON objects", () => {
    const directory = createTempDir("mlxts-hub-inspect-");
    const configPath = join(directory, "config.json");
    const generationConfigPath = join(directory, "generation_config.json");
    const tokenizerJsonPath = join(directory, "tokenizer.json");
    const tekkenJsonPath = join(directory, "tekken.json");
    const tokenizerConfigPath = join(directory, "tokenizer_config.json");
    const specialTokensMapPath = join(directory, "special_tokens_map.json");
    const safetensorsIndexPath = join(directory, "model.safetensors.index.json");
    const ggufPath = join(directory, "model.gguf");
    const safetensorsPath = join(directory, "model.safetensors");

    writeFileSync(configPath, JSON.stringify({ model_type: "llama" }));
    writeFileSync(generationConfigPath, JSON.stringify({ temperature: 0.8 }));
    writeFileSync(tokenizerJsonPath, JSON.stringify({ model: { type: "BPE" } }));
    writeFileSync(tekkenJsonPath, JSON.stringify({ config: { pattern: "\\p{L}+" } }));
    writeFileSync(tokenizerConfigPath, JSON.stringify({ bos_token: "<bos>" }));
    writeFileSync(specialTokensMapPath, JSON.stringify({ eos_token: "<eos>" }));
    writeFileSync(safetensorsIndexPath, JSON.stringify({ metadata: { total_size: 1 } }));
    writeFileSync(ggufPath, "");
    writeFileSync(safetensorsPath, "");

    const snapshot = {
      source: "local" as const,
      repoType: "model" as const,
      directory,
      files: [
        { relativePath: "config.json", localPath: configPath, size: 1 },
        { relativePath: "generation_config.json", localPath: generationConfigPath, size: 1 },
        { relativePath: "tokenizer.json", localPath: tokenizerJsonPath, size: 1 },
        { relativePath: "tekken.json", localPath: tekkenJsonPath, size: 1 },
        { relativePath: "tokenizer_config.json", localPath: tokenizerConfigPath, size: 1 },
        { relativePath: "special_tokens_map.json", localPath: specialTokensMapPath, size: 1 },
        { relativePath: "model.safetensors.index.json", localPath: safetensorsIndexPath, size: 1 },
        { relativePath: "model.safetensors", localPath: safetensorsPath, size: 1 },
        { relativePath: "model.gguf", localPath: ggufPath, size: 1 },
      ],
    };

    const inspection = inspectSnapshot(snapshot);
    expect(inspection.model.configPath).toBe(configPath);
    expect(inspection.model.generationConfigPath).toBe(generationConfigPath);
    expect(inspection.model.ggufPaths).toEqual([ggufPath]);
    expect(inspection.model.safetensorPaths).toEqual([safetensorsPath]);
    expect(inspection.tokenizer.tokenizerJsonPath).toBe(tokenizerJsonPath);
    expect(inspection.tokenizer.tekkenJsonPath).toBe(tekkenJsonPath);
    expect(inspection.config.model_type).toBe("llama");
    expect(inspection.generationConfig.temperature).toBe(0.8);
    expect(inspection.tokenizerConfig.bos_token).toBe("<bos>");
    expect(inspection.specialTokensMap.eos_token).toBe("<eos>");
    expect(inspection.safetensorsIndex.metadata).toEqual({ total_size: 1 });

    writeFileSync(generationConfigPath, JSON.stringify(["bad"]));
    expect(() => inspectSnapshot(snapshot)).toThrow('expected "');
  });

  test("parseGgufHeader reads metadata variants and rejects invalid files", async () => {
    const directory = createTempDir("mlxts-hub-gguf-");
    const path = join(directory, "rich.gguf");
    const bytes: number[] = [];

    pushU32(bytes, 0x46554747);
    pushU32(bytes, 3);
    pushU64(bytes, 1n);
    pushU64(bytes, 4n);

    pushString(bytes, "general.architecture");
    pushU32(bytes, 8);
    pushString(bytes, "llama");

    pushString(bytes, "general.alignment");
    pushU32(bytes, 4);
    pushU32(bytes, 32);

    pushString(bytes, "general.is_quantized");
    pushU32(bytes, 7);
    pushU8(bytes, 1);

    pushString(bytes, "general.scales");
    pushU32(bytes, 9);
    pushU32(bytes, 12);
    pushU64(bytes, 2n);
    pushF64(bytes, 1.5);
    pushF64(bytes, 2.5);

    pushString(bytes, "blk.0.attn_q.weight");
    pushU32(bytes, 2);
    pushU64(bytes, 4n);
    pushU64(bytes, 8n);
    pushU32(bytes, 0);
    pushU64(bytes, 128n);

    await Bun.write(path, new Uint8Array(bytes));

    const header = parseGgufHeader(path);
    expect(header.version).toBe(3);
    expect(header.metadata["general.architecture"]).toBe("llama");
    expect(header.metadata["general.alignment"]).toBe(32);
    expect(header.metadata["general.is_quantized"]).toBe(true);
    expect(header.metadata["general.scales"]).toEqual([1.5, 2.5]);
    expect(header.tensors).toEqual([
      {
        name: "blk.0.attn_q.weight",
        dimensions: [4, 8],
        type: 0,
        offset: 128,
      },
    ]);

    const tinyPath = join(directory, "tiny.gguf");
    await Bun.write(tinyPath, new Uint8Array([1, 2, 3]));
    expect(() => parseGgufHeader(tinyPath)).toThrow("too small");

    const badMagicPath = join(directory, "bad-magic.gguf");
    await Bun.write(
      badMagicPath,
      new Uint8Array([0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    );
    expect(() => parseGgufHeader(badMagicPath)).toThrow('expected magic "GGUF"');

    const badTypePath = join(directory, "bad-type.gguf");
    const badTypeBytes: number[] = [];
    pushU32(badTypeBytes, 0x46554747);
    pushU32(badTypeBytes, 3);
    pushU64(badTypeBytes, 0n);
    pushU64(badTypeBytes, 1n);
    pushString(badTypeBytes, "broken");
    pushU32(badTypeBytes, 99);
    await Bun.write(badTypePath, new Uint8Array(badTypeBytes));
    expect(() => parseGgufHeader(badTypePath)).toThrow("unsupported GGUF metadata value type 99");
  });

  test("parseGgufHeader covers scalar metadata variants and rejects unsafe uint64 values", async () => {
    const directory = createTempDir("mlxts-hub-gguf-scalars-");
    const path = join(directory, "scalars.gguf");
    const bytes: number[] = [];

    pushU32(bytes, 0x46554747);
    pushU32(bytes, 3);
    pushU64(bytes, 0n);
    pushU64(bytes, 7n);

    pushString(bytes, "u8");
    pushU32(bytes, 0);
    pushU8(bytes, 255);

    pushString(bytes, "i8");
    pushU32(bytes, 1);
    pushU8(bytes, 0xfb);

    pushString(bytes, "u16");
    pushU32(bytes, 2);
    bytes.push(0xff, 0x00);

    pushString(bytes, "i16");
    pushU32(bytes, 3);
    bytes.push(0xfe, 0xff);

    pushString(bytes, "f32");
    pushU32(bytes, 6);
    const f32 = new Uint8Array(4);
    new DataView(f32.buffer).setFloat32(0, 1.25, true);
    bytes.push(...f32);

    pushString(bytes, "u64");
    pushU32(bytes, 10);
    pushU64(bytes, 42n);

    pushString(bytes, "i64");
    pushU32(bytes, 11);
    const i64 = new Uint8Array(8);
    new DataView(i64.buffer).setBigInt64(0, -7n, true);
    bytes.push(...i64);

    await Bun.write(path, new Uint8Array(bytes));

    const header = parseGgufHeader(path);
    expect(header.metadata).toEqual({
      u8: 255,
      i8: -5,
      u16: 255,
      i16: -2,
      f32: 1.25,
      u64: 42,
      i64: -7,
    });

    const unsafePath = join(directory, "unsafe.gguf");
    const unsafeBytes: number[] = [];
    pushU32(unsafeBytes, 0x46554747);
    pushU32(unsafeBytes, 3);
    pushU64(unsafeBytes, 0n);
    pushU64(unsafeBytes, 1n);
    pushString(unsafeBytes, "too-big");
    pushU32(unsafeBytes, 10);
    pushU64(unsafeBytes, 9007199254740992n);
    await Bun.write(unsafePath, new Uint8Array(unsafeBytes));

    expect(() => parseGgufHeader(unsafePath)).toThrow("larger than Number.MAX_SAFE_INTEGER");
  });

  test("http helpers build URLs and surface fetch failures clearly", async () => {
    expect(resolveFileUrl("test-org/test-model", "dataset", "main", "config.json")).toBe(
      "https://huggingface.co/datasets/test-org/test-model/resolve/main/config.json",
    );

    const originalFetch = globalThis.fetch;
    globalThis.fetch = withFetchStub(
      originalFetch,
      async () => new Response("nope", { status: 500, statusText: "Boom" }),
    );

    try {
      await expect(fetchJson("test-org/test-model", "model", "main", "")).rejects.toThrow(
        "HuggingFace API request failed",
      );
      await expect(fetchFile("https://huggingface.co/test", undefined, "HEAD")).rejects.toThrow(
        "downloadSnapshot: HEAD https://huggingface.co/test failed",
      );

      globalThis.fetch = withFetchStub(
        originalFetch,
        async () => new Response(JSON.stringify(["bad-payload"]), { status: 200 }),
      );
      await expect(fetchJson("test-org/test-model", "model", "main", undefined)).rejects.toThrow(
        "returned a non-object payload",
      );
    } finally {
      globalThis.fetch = originalFetch;
    }
  });
});
