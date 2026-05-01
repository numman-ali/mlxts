import type { MxArray } from "@mlxts/core";
import { mkdirSync, writeFileSync } from "fs";
import { dirname } from "path";

export type ZImageBmpWriteResult = {
  path: string;
  width: number;
  height: number;
  bytes: number;
};

function clampByte(value: number): number {
  if (!Number.isFinite(value) || value <= 0) {
    return 0;
  }
  if (value >= 1) {
    return 255;
  }
  return Math.round(value * 255);
}

function assertSingleRgbImage(image: MxArray): { height: number; width: number; channels: number } {
  const [batch, height, width, channels] = image.shape;
  if (
    image.shape.length !== 4 ||
    batch !== 1 ||
    height === undefined ||
    width === undefined ||
    channels === undefined ||
    channels < 3
  ) {
    throw new Error(`Z-Image output expects one NHWC RGB image, got [${image.shape.join(",")}].`);
  }
  return { height, width, channels };
}

function writeAscii(view: DataView, offset: number, value: string): void {
  for (let index = 0; index < value.length; index += 1) {
    const code = value.charCodeAt(index);
    view.setUint8(offset + index, code);
  }
}

/** Convert one NHWC RGB image tensor in 0..1 range into an uncompressed 24-bit BMP. */
export function zImageToBmpBytes(image: MxArray): Uint8Array {
  const { height, width, channels } = assertSingleRgbImage(image);
  image.eval();
  const pixels = image.toTypedArray();
  const headerBytes = 54;
  const rowStride = Math.ceil((width * 3) / 4) * 4;
  const pixelBytes = rowStride * height;
  const bytes = new Uint8Array(headerBytes + pixelBytes);
  const view = new DataView(bytes.buffer);

  writeAscii(view, 0, "BM");
  view.setUint32(2, bytes.byteLength, true);
  view.setUint32(10, headerBytes, true);
  view.setUint32(14, 40, true);
  view.setInt32(18, width, true);
  view.setInt32(22, height, true);
  view.setUint16(26, 1, true);
  view.setUint16(28, 24, true);
  view.setUint32(30, 0, true);
  view.setUint32(34, pixelBytes, true);
  view.setInt32(38, 2835, true);
  view.setInt32(42, 2835, true);

  for (let y = 0; y < height; y += 1) {
    const sourceY = height - 1 - y;
    const rowOffset = headerBytes + y * rowStride;
    for (let x = 0; x < width; x += 1) {
      const sourceOffset = ((sourceY * width + x) * channels) | 0;
      const targetOffset = rowOffset + x * 3;
      bytes[targetOffset] = clampByte(Number(pixels[sourceOffset + 2]));
      bytes[targetOffset + 1] = clampByte(Number(pixels[sourceOffset + 1]));
      bytes[targetOffset + 2] = clampByte(Number(pixels[sourceOffset]));
    }
  }

  return bytes;
}

/** Write one generated Z-Image tensor as an uncompressed BMP artifact. */
export function writeZImageBmp(image: MxArray, outputPath: string): ZImageBmpWriteResult {
  if (!outputPath.toLowerCase().endsWith(".bmp")) {
    throw new Error("Z-Image output path must end with .bmp.");
  }
  const bytes = zImageToBmpBytes(image);
  const { height, width } = assertSingleRgbImage(image);
  mkdirSync(dirname(outputPath), { recursive: true });
  writeFileSync(outputPath, bytes);
  return {
    path: outputPath,
    width,
    height,
    bytes: bytes.byteLength,
  };
}
