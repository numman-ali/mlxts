import { describe, expect, test } from "bun:test";

import {
  decodeResizedImageBytes,
  parseBmp,
  readImageBytesSize,
  readImageSourceBytes,
} from "./image";

function uint16le(value: number): number[] {
  return [value & 0xff, (value >> 8) & 0xff];
}

function int32le(value: number): number[] {
  return [value & 0xff, (value >> 8) & 0xff, (value >> 16) & 0xff, (value >> 24) & 0xff];
}

function uint32le(value: number): number[] {
  return int32le(value >>> 0);
}

function bmpBytes(width: number, height: number, pixels: readonly number[]): Uint8Array {
  const bytesPerPixel = 3;
  const rowStride = Math.ceil((width * bytesPerPixel) / 4) * 4;
  const pixelArraySize = rowStride * height;
  const fileSize = 54 + pixelArraySize;
  const header = [
    0x42,
    0x4d,
    ...uint32le(fileSize),
    0,
    0,
    0,
    0,
    ...uint32le(54),
    ...uint32le(40),
    ...int32le(width),
    ...int32le(-height),
    ...uint16le(1),
    ...uint16le(24),
    ...uint32le(0),
    ...uint32le(pixelArraySize),
    ...uint32le(0),
    ...uint32le(0),
    ...uint32le(0),
    ...uint32le(0),
  ];

  const pixelBytes: number[] = [];
  for (let row = 0; row < height; row += 1) {
    for (let column = 0; column < width; column += 1) {
      const index = (row * width + column) * 3;
      const red = pixels[index];
      const green = pixels[index + 1];
      const blue = pixels[index + 2];
      if (red === undefined || green === undefined || blue === undefined) {
        throw new Error("bmpBytes: missing RGB pixel data.");
      }
      pixelBytes.push(blue, green, red);
    }
    while (pixelBytes.length % rowStride !== 0) {
      pixelBytes.push(0);
    }
  }

  return new Uint8Array([...header, ...pixelBytes]);
}

function bmpBitfieldBytes(width: number, height: number, pixels: readonly number[]): Uint8Array {
  const bytesPerPixel = 4;
  const rowStride = width * bytesPerPixel;
  const pixelArraySize = rowStride * height;
  const dibHeaderSize = 124;
  const pixelOffset = 14 + dibHeaderSize;
  const fileSize = pixelOffset + pixelArraySize;
  const dibHeader = [
    ...uint32le(dibHeaderSize),
    ...int32le(width),
    ...int32le(-height),
    ...uint16le(1),
    ...uint16le(32),
    ...uint32le(3),
    ...uint32le(pixelArraySize),
    ...uint32le(0),
    ...uint32le(0),
    ...uint32le(0),
    ...uint32le(0),
    ...uint32le(0x00ff0000),
    ...uint32le(0x0000ff00),
    ...uint32le(0x000000ff),
    ...uint32le(0xff000000),
    ...uint32le(0x73524742),
  ];
  while (dibHeader.length < dibHeaderSize) {
    dibHeader.push(0);
  }

  const header = [
    0x42,
    0x4d,
    ...uint32le(fileSize),
    0,
    0,
    0,
    0,
    ...uint32le(pixelOffset),
    ...dibHeader,
  ];

  const pixelBytes: number[] = [];
  for (let row = 0; row < height; row += 1) {
    for (let column = 0; column < width; column += 1) {
      const index = (row * width + column) * 3;
      const red = pixels[index];
      const green = pixels[index + 1];
      const blue = pixels[index + 2];
      if (red === undefined || green === undefined || blue === undefined) {
        throw new Error("bmpBitfieldBytes: missing RGB pixel data.");
      }
      pixelBytes.push(blue, green, red, 255);
    }
  }

  return new Uint8Array([...header, ...pixelBytes]);
}

function base64Bytes(value: string): Uint8Array {
  const binary = atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return bytes;
}

describe("media image helpers", () => {
  test("parses 24-bit BMP payloads into RGB bytes", () => {
    const image = parseBmp(bmpBytes(2, 1, [255, 0, 0, 0, 255, 0]));

    expect(image.width).toBe(2);
    expect(image.height).toBe(1);
    expect(image.channels).toBe(3);
    expect(Array.from(image.data)).toEqual([255, 0, 0, 0, 255, 0]);
  });

  test("parses bottom-up BMP rows into top-down RGB bytes", () => {
    const topDown = bmpBytes(1, 2, [255, 0, 0, 0, 255, 0]);
    const bottomUp = new Uint8Array(topDown);
    bottomUp[22] = 2;
    bottomUp[23] = 0;
    bottomUp[24] = 0;
    bottomUp[25] = 0;

    const image = parseBmp(bottomUp);

    expect(Array.from(image.data)).toEqual([0, 255, 0, 255, 0, 0]);
  });

  test("parses 32-bit bitfield BMP payloads into RGB bytes", () => {
    const image = parseBmp(bmpBitfieldBytes(2, 1, [255, 0, 0, 0, 255, 0]));

    expect(image.width).toBe(2);
    expect(image.height).toBe(1);
    expect(image.channels).toBe(3);
    expect(Array.from(image.data)).toEqual([255, 0, 0, 0, 255, 0]);
  });

  test("rejects malformed BMP metadata", () => {
    expect(() => parseBmp(new Uint8Array([0x50, 0x36]))).toThrow("expected a BMP payload");
    const unsupportedHeader = bmpBytes(1, 1, [255, 0, 0]);
    unsupportedHeader[14] = 12;
    expect(() => parseBmp(unsupportedHeader)).toThrow("unsupported BMP DIB header");
    const compressed = bmpBytes(1, 1, [255, 0, 0]);
    compressed[30] = 1;
    expect(() => parseBmp(compressed)).toThrow("expected an uncompressed");
    const nonContiguousMask = bmpBitfieldBytes(1, 1, [255, 0, 0]);
    nonContiguousMask.set(uint32le(0x00f000f0), 54);
    expect(() => parseBmp(nonContiguousMask)).toThrow("expected contiguous BMP color masks");
    const overlappingHeader = bmpBitfieldBytes(1, 1, [255, 0, 0]);
    overlappingHeader.set(uint32le(66), 10);
    expect(() => parseBmp(overlappingHeader)).toThrow("BMP pixel data overlaps bitfield metadata");
    const invalidSize = bmpBytes(1, 1, [255, 0, 0]);
    invalidSize[18] = 0;
    expect(() => parseBmp(invalidSize)).toThrow("expected positive BMP width");
    const truncated = bmpBytes(2, 2, [255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0]).slice(0, 56);
    expect(() => parseBmp(truncated)).toThrow("BMP pixel data is truncated");
  });

  test("loads image data URLs as bytes", async () => {
    const bytes = await readImageSourceBytes({
      kind: "data",
      mediaType: "image/png",
      data: btoa("abc"),
    });

    expect(Array.from(bytes)).toEqual([97, 98, 99]);
  });

  test("rejects non-image data URLs", async () => {
    await expect(
      readImageSourceBytes({
        kind: "data",
        mediaType: "text/plain",
        data: btoa("abc"),
      }),
    ).rejects.toThrow("image/*");
  });

  test("rejects oversized data URL payloads", async () => {
    await expect(
      readImageSourceBytes(
        {
          kind: "data",
          mediaType: "image/png",
          data: btoa("abc"),
        },
        { maxBytes: 2 },
      ),
    ).rejects.toThrow("exceeding the 2 byte limit");
    await expect(
      readImageSourceBytes(
        {
          kind: "data",
          mediaType: "image/png",
          data: "A".repeat(64),
        },
        { maxBytes: 2 },
      ),
    ).rejects.toThrow("at least");
  });

  test("loads remote image URLs through the injected transport policy", async () => {
    const bytes = await readImageSourceBytes(
      { kind: "url", url: "https://example.com/image.png" },
      {
        remoteImageHosts: ["example.com"],
        remoteResolve: async () => [{ address: "93.184.216.34", family: 4 }],
        remoteFetch: async () =>
          new Response("abc", {
            headers: { "content-type": "image/png", "content-length": "3" },
          }),
      },
    );

    expect(Array.from(bytes)).toEqual([97, 98, 99]);
  });

  test("rejects file-id image sources and aborted image reads", async () => {
    await expect(readImageSourceBytes({ kind: "file", fileId: "file-1" })).rejects.toThrow(
      "File-id image inputs are not supported",
    );
    const controller = new AbortController();
    controller.abort();
    await expect(
      readImageSourceBytes(
        { kind: "data", mediaType: "image/png", data: btoa("abc") },
        { signal: controller.signal },
      ),
    ).rejects.toThrow("cancelled");
  });

  test("reads and resizes BMP bytes through the platform decoder", async () => {
    const bytes = bmpBytes(2, 1, [255, 0, 0, 0, 255, 0]);

    const size = await readImageBytesSize(bytes);
    const resized = await decodeResizedImageBytes(bytes, { width: 1, height: 1 });

    expect(size).toEqual({ width: 2, height: 1 });
    expect(resized.width).toBe(1);
    expect(resized.height).toBe(1);
    expect(resized.channels).toBe(3);
    expect(resized.data.length).toBe(3);
  });

  test("reads and resizes PNG bytes through the platform decoder", async () => {
    const redRgbPng = base64Bytes(
      "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAAAXNSR0IArs4c6QAAAERlWElmTU0AKgAAAAgAAYdpAAQAAAABAAAAGgAAAAAAA6ABAAMAAAABAAEAAKACAAQAAAABAAAAAaADAAQAAAABAAAAAQAAAAD5Ip3+AAAADElEQVQIHWP4z8AAAAMBAQBb2/lEAAAAAElFTkSuQmCC",
    );
    const redRgbaPng = base64Bytes(
      "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DQAAAEgQGALFXOsAAAAABJRU5ErkJggg==",
    );

    const size = await readImageBytesSize(redRgbPng);
    const resized = await decodeResizedImageBytes(redRgbPng, { width: 1, height: 1 });
    const rgbaResized = await decodeResizedImageBytes(redRgbaPng, { width: 1, height: 1 });

    expect(size).toEqual({ width: 1, height: 1 });
    expect(resized.width).toBe(1);
    expect(resized.height).toBe(1);
    expect(resized.channels).toBe(3);
    expect(Array.from(resized.data)).toEqual([255, 0, 0]);
    expect(Array.from(rgbaResized.data)).toEqual([255, 0, 0]);
  });
});
