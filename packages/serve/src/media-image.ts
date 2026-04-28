/**
 * Image byte loading and macOS-backed RGB decoding for serving adapters.
 * @module
 */

import { ServeError } from "./errors";
import type { GenerationMediaSource } from "./types";

/** Default decoded image input byte cap for local serving media payloads. */
export const DEFAULT_IMAGE_SOURCE_MAX_BYTES = 32 * 1024 * 1024;

/** Pixel dimensions reported by the host image decoder. */
export type ImageSize = {
  width: number;
  height: number;
};

/** Decoded row-major RGB image data owned by the serving media layer. */
export type DecodedRgbImage = ImageSize & {
  channels: 3;
  data: Uint8Array;
};

/** Safety controls for host-side image reads and platform decoding. */
export type ImageReadOptions = {
  signal?: AbortSignal;
  maxBytes?: number;
};

function temporaryPath(extension: string): string {
  const directory = Bun.env.TMPDIR ?? "/tmp";
  return `${directory.replace(/\/$/, "")}/mlxts-serve-${crypto.randomUUID()}.${extension}`;
}

function throwIfAborted(signal: AbortSignal | undefined, context: string): void {
  if (signal?.aborted === true) {
    throw new DOMException(`${context}: image read was cancelled.`, "AbortError");
  }
}

function decodeCommandText(output: Uint8Array, context: string): string {
  const text = new TextDecoder().decode(output).trim();
  if (text === "") {
    throw new Error(`${context}: command did not produce any output.`);
  }
  return text;
}

async function runSips(
  args: string[],
  context: string,
  signal: AbortSignal | undefined,
): Promise<string> {
  throwIfAborted(signal, context);
  const process = Bun.spawn(["sips", ...args], {
    stdout: "pipe",
    stderr: "pipe",
  });

  let cancelled = false;
  function abort(): void {
    cancelled = true;
    process.kill();
  }

  signal?.addEventListener("abort", abort, { once: true });
  try {
    const [exitCode, stdout, stderr] = await Promise.all([
      process.exited,
      new Response(process.stdout).bytes(),
      new Response(process.stderr).bytes(),
    ]);
    if (cancelled || signal?.aborted === true) {
      throw new DOMException(`${context}: image decode was cancelled.`, "AbortError");
    }
    if (exitCode !== 0) {
      const stderrText = new TextDecoder().decode(stderr).trim();
      throw new Error(stderrText === "" ? `${context}: sips failed.` : `${context}: ${stderrText}`);
    }
    return decodeCommandText(stdout, context);
  } finally {
    signal?.removeEventListener("abort", abort);
  }
}

function runRemoval(path: string): void {
  const result = Bun.spawnSync(["rm", "-f", path]);
  if (result.exitCode !== 0) {
    const stderr = new TextDecoder().decode(result.stderr).trim();
    throw new Error(stderr === "" ? `removeTemporaryFile: failed to remove ${path}.` : stderr);
  }
}

function removeTemporaryFile(path: string): void {
  runRemoval(path);
}

function readPositiveInteger(value: string | undefined, context: string): number {
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
  if (planes !== 1 || compression !== 0 || (bitsPerPixel !== 24 && bitsPerPixel !== 32)) {
    throw new Error(`${context}: expected an uncompressed 24-bit or 32-bit BMP payload.`);
  }
  if (width <= 0 || signedHeight === 0) {
    throw new Error(`${context}: expected positive BMP width/height.`);
  }

  const height = Math.abs(signedHeight);
  const bytesPerPixel = bitsPerPixel / 8;
  const rowStride = Math.ceil((bitsPerPixel * width) / 32) * 4;
  if (pixelOffset + rowStride * height > bytes.length) {
    throw new Error(`${context}: BMP pixel data is truncated.`);
  }

  return { width, height, pixelOffset, bytesPerPixel, rowStride, topDown: signedHeight < 0 };
}

function readBmpRgb(
  bytes: Uint8Array,
  pixelOffset: number,
  context: string,
): { red: number; green: number; blue: number } {
  const blue = bytes[pixelOffset];
  const green = bytes[pixelOffset + 1];
  const red = bytes[pixelOffset + 2];
  if (red === undefined || green === undefined || blue === undefined) {
    throw new Error(`${context}: BMP pixel data is truncated.`);
  }
  return { red, green, blue };
}

function decodeBmpPixels(bytes: Uint8Array, metadata: BmpMetadata, context: string): Uint8Array {
  const data = new Uint8Array(metadata.width * metadata.height * 3);
  let cursor = 0;
  for (let row = 0; row < metadata.height; row += 1) {
    const sourceRow = metadata.topDown ? row : metadata.height - 1 - row;
    const rowOffset = metadata.pixelOffset + sourceRow * metadata.rowStride;
    for (let column = 0; column < metadata.width; column += 1) {
      const pixelOffset = rowOffset + column * metadata.bytesPerPixel;
      const pixel = readBmpRgb(bytes, pixelOffset, context);
      data[cursor] = pixel.red;
      data[cursor + 1] = pixel.green;
      data[cursor + 2] = pixel.blue;
      cursor += 3;
    }
  }
  return data;
}

/** Parse a BMP payload into row-major RGB bytes. */
export function parseBmp(bytes: Uint8Array, context = "parseBmp"): DecodedRgbImage {
  const metadata = readBmpMetadata(bytes, context);
  return {
    width: metadata.width,
    height: metadata.height,
    channels: 3,
    data: decodeBmpPixels(bytes, metadata, context),
  };
}

function decodedBase64ByteLength(value: string): number {
  let encodedLength = 0;
  let padding = 0;
  for (const char of value) {
    if (/\s/.test(char)) {
      continue;
    }
    encodedLength += 1;
    if (char === "=") {
      padding += 1;
    }
  }
  return Math.max(0, Math.floor((encodedLength * 3) / 4) - Math.min(padding, 2));
}

function rejectOversizedBase64(value: string, maxBytes: number, context: string): void {
  const estimatedBytes = decodedBase64ByteLength(value);
  if (estimatedBytes > maxBytes) {
    throw new ServeError(
      `${context}: image payload is at least ${estimatedBytes} bytes, exceeding the ${maxBytes} byte limit.`,
      { code: "unsupported_input", param: "messages" },
    );
  }
}

function decodeBase64(value: string, context: string, maxBytes: number): Uint8Array {
  rejectOversizedBase64(value, maxBytes, context);
  try {
    const binary = atob(value.replace(/\s+/g, ""));
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index += 1) {
      bytes[index] = binary.charCodeAt(index);
    }
    return bytes;
  } catch {
    throw new ServeError(`${context}: invalid base64 media payload.`, { param: "messages" });
  }
}

function enforceImageByteLimit(bytes: Uint8Array, maxBytes: number, context: string): Uint8Array {
  if (bytes.byteLength > maxBytes) {
    throw new ServeError(
      `${context}: image payload is ${bytes.byteLength} bytes, exceeding the ${maxBytes} byte limit.`,
      { code: "unsupported_input", param: "messages" },
    );
  }
  return bytes;
}

/** Load image bytes from a normalized media source. */
export async function readImageSourceBytes(
  source: GenerationMediaSource,
  options: ImageReadOptions = {},
): Promise<Uint8Array> {
  throwIfAborted(options.signal, "readImageSourceBytes");
  const maxBytes = options.maxBytes ?? DEFAULT_IMAGE_SOURCE_MAX_BYTES;
  switch (source.kind) {
    case "data":
      if (!source.mediaType.toLowerCase().startsWith("image/")) {
        throw new ServeError("Image data URLs must use an image/* media type.", {
          param: "messages",
        });
      }
      return enforceImageByteLimit(
        decodeBase64(source.data, "Image data URL", maxBytes),
        maxBytes,
        "Image data URL",
      );
    case "url":
      throw new ServeError(
        "Remote image URLs are not supported by local serving yet. Send image data URLs instead.",
        { code: "unsupported_input", param: "messages" },
      );
    case "file":
      throw new ServeError("File-id image inputs are not supported by local serving yet.", {
        code: "unsupported_input",
        param: "messages",
      });
  }
}

/** Read image dimensions using the platform image decoder. */
export async function readImageSize(
  path: string,
  options: ImageReadOptions = {},
): Promise<ImageSize> {
  const output = await runSips(
    ["-g", "pixelWidth", "-g", "pixelHeight", path],
    "readImageSize",
    options.signal,
  );
  const widthMatch = output.match(/pixelWidth:\s+(\d+)/);
  const heightMatch = output.match(/pixelHeight:\s+(\d+)/);
  return {
    width: readPositiveInteger(widthMatch?.[1], "readImageSize: width"),
    height: readPositiveInteger(heightMatch?.[1], "readImageSize: height"),
  };
}

/** Decode and resize image bytes into row-major RGB bytes. */
export async function decodeResizedImageBytes(
  bytes: Uint8Array,
  size: ImageSize,
  options: ImageReadOptions = {},
): Promise<DecodedRgbImage> {
  const inputPath = temporaryPath("image");
  const outputPath = temporaryPath("bmp");
  try {
    throwIfAborted(options.signal, "decodeResizedImageBytes");
    await Bun.write(inputPath, bytes);
    throwIfAborted(options.signal, "decodeResizedImageBytes");
    await runSips(
      [
        "-z",
        String(size.height),
        String(size.width),
        "-s",
        "format",
        "bmp",
        inputPath,
        "--out",
        outputPath,
      ],
      "decodeResizedImageBytes",
      options.signal,
    );
    throwIfAborted(options.signal, "decodeResizedImageBytes");
    const decoded = new Uint8Array(await Bun.file(outputPath).arrayBuffer());
    throwIfAborted(options.signal, "decodeResizedImageBytes");
    return parseBmp(decoded, "decoded image");
  } finally {
    removeTemporaryFile(inputPath);
    removeTemporaryFile(outputPath);
  }
}

/** Decode enough metadata to choose a model-native resize target. */
export async function readImageBytesSize(
  bytes: Uint8Array,
  options: ImageReadOptions = {},
): Promise<ImageSize> {
  const inputPath = temporaryPath("image");
  try {
    throwIfAborted(options.signal, "readImageBytesSize");
    await Bun.write(inputPath, bytes);
    throwIfAborted(options.signal, "readImageBytesSize");
    return await readImageSize(inputPath, options);
  } finally {
    removeTemporaryFile(inputPath);
  }
}
