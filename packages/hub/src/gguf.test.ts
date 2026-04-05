import { afterEach, describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

import { parseGgufHeader } from "./gguf";

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

function pushString(bytes: number[], value: string): void {
  const encoded = new TextEncoder().encode(value);
  pushU64(bytes, BigInt(encoded.byteLength));
  bytes.push(...encoded);
}

describe("parseGgufHeader", () => {
  test("reads GGUF metadata and tensor descriptors", async () => {
    const directory = createTempDir("mlxts-hub-gguf-");
    const path = join(directory, "tiny.gguf");
    const bytes: number[] = [];

    pushU32(bytes, 0x46554747);
    pushU32(bytes, 3);
    pushU64(bytes, 1n);
    pushU64(bytes, 2n);

    pushString(bytes, "general.architecture");
    pushU32(bytes, 8);
    pushString(bytes, "llama");

    pushString(bytes, "llama.attention.head_count");
    pushU32(bytes, 4);
    pushU32(bytes, 8);

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
    expect(header.metadata["llama.attention.head_count"]).toBe(8);
    expect(header.tensors).toEqual([
      {
        name: "blk.0.attn_q.weight",
        dimensions: [4, 8],
        type: 0,
        offset: 128,
      },
    ]);
  });
});
