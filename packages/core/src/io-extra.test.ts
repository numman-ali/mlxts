import { describe, expect, test } from "bun:test";
import { closeSync, ftruncateSync, mkdtempSync, openSync, writeSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

import { array } from "./array";
import {
  iterateSafetensors,
  iterateSafetensorTensorChunks,
  loadSafetensors,
  saveSafetensors,
} from "./io";

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

  test("loadSafetensors reads tensors from sparse files with offsets past 4GB", async () => {
    const path = writeFixturePath("large-sparse-offset");
    const tensorOffset = 4_294_967_296;
    const tensorBytes = new Uint8Array(new Float32Array([1.25, -2.5]).buffer.slice(0));
    const headerPayload = encodeFixture({
      large: {
        dtype: "F32",
        shape: [2],
        data_offsets: [tensorOffset, tensorOffset + tensorBytes.byteLength],
      },
    });
    const absoluteTensorOffset = headerPayload.byteLength + tensorOffset;

    const fileDescriptor = openSync(path, "w");
    try {
      writeSync(fileDescriptor, headerPayload);
      ftruncateSync(fileDescriptor, absoluteTensorOffset + tensorBytes.byteLength);
      writeSync(fileDescriptor, tensorBytes, 0, tensorBytes.byteLength, absoluteTensorOffset);
    } finally {
      closeSync(fileDescriptor);
    }

    const loaded = await loadSafetensors(path);
    using restored = loaded.tensors.large;
    expect(restored).toBeDefined();
    expect(restored?.dtype).toBe("float32");
    expect(restored?.toList()).toEqual([1.25, -2.5]);
  });

  test("iterateSafetensors yields tensors lazily and can skip entries by name", async () => {
    const path = writeFixturePath("iterates-lazily");
    using left = array([1, 2], "float32");
    using right = array([3, 4], "float32");
    await saveSafetensors(
      {
        left,
        right,
      },
      path,
    );

    const iteratedNames: string[] = [];
    for await (const entry of iterateSafetensors(path, { include: (name) => name !== "left" })) {
      iteratedNames.push(entry.name);
      expect(entry.tensor.toList()).toEqual([3, 4]);
      entry.tensor.free();
    }

    expect(iteratedNames).toEqual(["right"]);
  });

  test("iterateSafetensors can iterate every tensor in a safetensors file", async () => {
    const path = writeFixturePath("iterates-all");
    using left = array([1, 2], "float32");
    using right = array([3, 4], "float32");
    await saveSafetensors(
      {
        left,
        right,
      },
      path,
      { source: "iterate-all" },
    );

    const seen = new Map<string, number[]>();
    for await (const entry of iterateSafetensors(path)) {
      seen.set(entry.name, entry.tensor.toList() as number[]);
      entry.tensor.free();
    }

    expect([...seen.keys()]).toEqual(["left", "right"]);
    expect(seen.get("left")).toEqual([1, 2]);
    expect(seen.get("right")).toEqual([3, 4]);
  });

  test("iterateSafetensorTensorChunks reads a tensor in bounded first-axis slices", async () => {
    const path = writeFixturePath("chunked-tensor");
    using matrix = array(
      [
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
      ],
      "float32",
    );
    await saveSafetensors({ matrix }, path);

    const chunks: Array<{ startIndex: number; values: number[][] }> = [];
    for await (const entry of iterateSafetensorTensorChunks(path, "matrix", {
      maxBytesPerChunk: 16,
    })) {
      chunks.push({
        startIndex: entry.startIndex,
        values: entry.tensor.toList() as number[][],
      });
      entry.tensor.free();
    }

    expect(chunks).toEqual([
      {
        startIndex: 0,
        values: [
          [1, 2],
          [3, 4],
        ],
      },
      {
        startIndex: 2,
        values: [
          [5, 6],
          [7, 8],
        ],
      },
    ]);
  });
});
