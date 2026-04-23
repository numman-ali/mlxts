import { describe, expect, test } from "bun:test";

import { parseBmp } from "./image-io";

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

describe("parseBmp", () => {
  test("parses a 24-bit BMP payload into RGB bytes", () => {
    const bytes = bmpBytes(2, 1, [255, 0, 0, 0, 255, 0]);

    const image = parseBmp(bytes);

    expect(image.width).toBe(2);
    expect(image.height).toBe(1);
    expect(image.channels).toBe(3);
    expect(Array.from(image.data)).toEqual([255, 0, 0, 0, 255, 0]);
  });

  test("rejects non-BMP payloads", () => {
    expect(() => parseBmp(new Uint8Array([0x50, 0x36]))).toThrow("expected a BMP payload");
  });
});
