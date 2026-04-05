import { describe, expect, test } from "bun:test";
import { mkdtempSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

import { array } from "./array";
import { loadSafetensors, saveSafetensors } from "./io";

function writeFixturePath(name: string): string {
  return join(mkdtempSync(join(tmpdir(), `${name}-`)), "fixture.safetensors");
}

function encodeFixture(
  header: Record<string, unknown>,
  body: Uint8Array = new Uint8Array(),
): Uint8Array {
  const encodedHeader = new TextEncoder().encode(JSON.stringify(header));
  const prefix = new Uint8Array(8);
  new DataView(prefix.buffer).setBigUint64(0, BigInt(encodedHeader.byteLength), true);
  const payload = new Uint8Array(8 + encodedHeader.byteLength + body.byteLength);
  payload.set(prefix, 0);
  payload.set(encodedHeader, 8);
  payload.set(body, 8 + encodedHeader.byteLength);
  return payload;
}

describe("io extra coverage", () => {
  test("saveSafetensors round-trips float16 tensors", async () => {
    const path = writeFixturePath("float16-roundtrip");
    using tensor = array([1, 2, 3], "float16");

    await saveSafetensors({ half: tensor }, path);
    const bytes = new Uint8Array(await Bun.file(path).arrayBuffer());
    expect(new TextDecoder().decode(bytes)).toContain('"dtype":"F16"');

    const loaded = await loadSafetensors(path);
    using restored = loaded.tensors.half;
    expect(restored).toBeDefined();
    expect(restored?.dtype).toBe("float16");
    expect(restored?.toList()).toEqual([1, 2, 3]);
  });

  test("saveSafetensors round-trips bfloat16 tensors", async () => {
    const path = writeFixturePath("bfloat16-roundtrip");
    using tensor = array([1, 2, 3], "bfloat16");

    await saveSafetensors({ half: tensor }, path);
    const bytes = new Uint8Array(await Bun.file(path).arrayBuffer());
    expect(new TextDecoder().decode(bytes)).toContain('"dtype":"BF16"');

    const loaded = await loadSafetensors(path);
    using restored = loaded.tensors.half;
    expect(restored).toBeDefined();
    expect(restored?.dtype).toBe("bfloat16");
    expect(restored?.toList()).toEqual([1, 2, 3]);
  });

  test("loadSafetensors rejects malformed tensor headers", async () => {
    const path = writeFixturePath("malformed-header");
    await Bun.write(
      path,
      encodeFixture({
        bad: {
          dtype: "F32",
          shape: [2],
          data_offsets: [0],
        },
      }),
    );

    await expect(loadSafetensors(path)).rejects.toThrow('header entry "bad" is malformed');
  });

  test("loadSafetensors rejects tensors that extend past the file size", async () => {
    const path = writeFixturePath("past-eof");
    await Bun.write(
      path,
      encodeFixture({
        bad: {
          dtype: "F32",
          shape: [2],
          data_offsets: [0, 8],
        },
      }),
    );

    await expect(loadSafetensors(path)).rejects.toThrow("extends past the end of the file");
  });
});
