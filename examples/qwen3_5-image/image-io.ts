import type { DecodedQwen3_5Image } from "@mlxts/transformers";
import { rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

type ImageSize = {
  width: number;
  height: number;
};

function decodeCommandText(output: Uint8Array, context: string): string {
  const text = new TextDecoder().decode(output).trim();
  if (text === "") {
    throw new Error(`${context}: command did not produce any output.`);
  }
  return text;
}

function runSips(args: string[], context: string): string {
  const result = Bun.spawnSync(["sips", ...args], {
    stdout: "pipe",
    stderr: "pipe",
  });
  if (result.exitCode !== 0) {
    const stderr = new TextDecoder().decode(result.stderr).trim();
    throw new Error(stderr === "" ? `${context}: sips failed.` : `${context}: ${stderr}`);
  }
  return decodeCommandText(result.stdout, context);
}

function parsePositiveInteger(value: string | undefined, context: string): number {
  if (value === undefined) {
    throw new Error(`${context}: missing integer value.`);
  }
  const parsed = Number.parseInt(value, 10);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`${context}: expected a positive integer, got "${value}".`);
  }
  return parsed;
}

function expectAvailable(
  bytes: Uint8Array,
  offset: number,
  length: number,
  context: string,
): DataView {
  if (offset < 0 || offset + length > bytes.length) {
    throw new Error(`${context}: unexpected end of BMP data.`);
  }
  return new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
}

function readInt32(bytes: Uint8Array, offset: number, context: string): number {
  return expectAvailable(bytes, offset, 4, context).getInt32(offset, true);
}

function readUint16(bytes: Uint8Array, offset: number, context: string): number {
  return expectAvailable(bytes, offset, 2, context).getUint16(offset, true);
}

function readUint32(bytes: Uint8Array, offset: number, context: string): number {
  return expectAvailable(bytes, offset, 4, context).getUint32(offset, true);
}

type BmpMetadata = {
  width: number;
  height: number;
  pixelOffset: number;
  bytesPerPixel: number;
  rowStride: number;
  topDown: boolean;
};

function readBmpMetadata(bytes: Uint8Array, context: string): BmpMetadata {
  if (bytes.length < 54 || bytes[0] !== 0x42 || bytes[1] !== 0x4d) {
    throw new Error(`${context}: expected a BMP payload.`);
  }

  const pixelOffset = readUint32(bytes, 10, context);
  const dibHeaderSize = readUint32(bytes, 14, context);
  if (dibHeaderSize < 40) {
    throw new Error(`${context}: unsupported BMP DIB header size ${dibHeaderSize}.`);
  }

  const width = readInt32(bytes, 18, context);
  const signedHeight = readInt32(bytes, 22, context);
  const planes = readUint16(bytes, 26, context);
  const bitsPerPixel = readUint16(bytes, 28, context);
  const compression = readUint32(bytes, 30, context);
  if (planes !== 1) {
    throw new Error(`${context}: expected BMP planes=1, got ${planes}.`);
  }
  if (compression !== 0) {
    throw new Error(`${context}: expected an uncompressed BMP, got compression=${compression}.`);
  }
  if (bitsPerPixel !== 24 && bitsPerPixel !== 32) {
    throw new Error(`${context}: expected 24-bit or 32-bit BMP data, got ${bitsPerPixel}.`);
  }
  if (width <= 0 || signedHeight === 0) {
    throw new Error(
      `${context}: expected positive BMP width/height, got ${width}x${signedHeight}.`,
    );
  }

  const height = Math.abs(signedHeight);
  const bytesPerPixel = bitsPerPixel / 8;
  const rowStride = Math.ceil((bitsPerPixel * width) / 32) * 4;
  if (pixelOffset + rowStride * height > bytes.length) {
    throw new Error(`${context}: BMP pixel data is truncated.`);
  }

  return {
    width,
    height,
    pixelOffset,
    bytesPerPixel,
    rowStride,
    topDown: signedHeight < 0,
  };
}

function readBmpRgb(
  bytes: Uint8Array,
  pixelOffset: number,
  context: string,
): readonly [number, number, number] {
  const blue = bytes[pixelOffset];
  const green = bytes[pixelOffset + 1];
  const red = bytes[pixelOffset + 2];
  if (red === undefined || green === undefined || blue === undefined) {
    throw new Error(`${context}: BMP pixel data is truncated.`);
  }
  return [red, green, blue];
}

function decodeBmpPixels(bytes: Uint8Array, metadata: BmpMetadata, context: string): Uint8Array {
  const { width, height, pixelOffset, bytesPerPixel, rowStride, topDown } = metadata;

  const data = new Uint8Array(width * height * 3);
  let cursor = 0;
  for (let row = 0; row < height; row += 1) {
    const sourceRow = topDown ? row : height - 1 - row;
    const rowOffset = pixelOffset + sourceRow * rowStride;
    for (let column = 0; column < width; column += 1) {
      const pixelOffsetInRow = rowOffset + column * bytesPerPixel;
      const [red, green, blue] = readBmpRgb(bytes, pixelOffsetInRow, context);
      data[cursor] = red;
      data[cursor + 1] = green;
      data[cursor + 2] = blue;
      cursor += 3;
    }
  }

  return data;
}

export function parseBmp(bytes: Uint8Array, context = "parseBmp"): DecodedQwen3_5Image {
  const metadata = readBmpMetadata(bytes, context);

  return {
    width: metadata.width,
    height: metadata.height,
    channels: 3,
    data: decodeBmpPixels(bytes, metadata, context),
  };
}

export function readImageSize(imagePath: string): ImageSize {
  const output = runSips(["-g", "pixelWidth", "-g", "pixelHeight", imagePath], "readImageSize");
  const widthMatch = output.match(/pixelWidth:\s+(\d+)/);
  const heightMatch = output.match(/pixelHeight:\s+(\d+)/);
  if (widthMatch === null || heightMatch === null) {
    throw new Error(`readImageSize: could not parse sips output for "${imagePath}".`);
  }
  return {
    width: parsePositiveInteger(widthMatch[1], "readImageSize: width"),
    height: parsePositiveInteger(heightMatch[1], "readImageSize: height"),
  };
}

export async function decodeResizedImage(
  imagePath: string,
  size: ImageSize,
): Promise<DecodedQwen3_5Image> {
  const temporaryPath = join(tmpdir(), `mlxts-qwen3_5-${crypto.randomUUID()}.bmp`);
  try {
    runSips(
      [
        "-z",
        String(size.height),
        String(size.width),
        "-s",
        "format",
        "bmp",
        imagePath,
        "--out",
        temporaryPath,
      ],
      "decodeResizedImage",
    );
    const bytes = new Uint8Array(await Bun.file(temporaryPath).arrayBuffer());
    return parseBmp(bytes, "decodeResizedImage");
  } finally {
    rmSync(temporaryPath, { force: true });
  }
}
