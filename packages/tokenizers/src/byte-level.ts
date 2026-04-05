/**
 * Byte-level helpers shared by GPT-2/Phi-style BPE tokenizers.
 * @module
 */

const BYTE_LEVEL_REGEX =
  /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

type ByteMaps = {
  byteToUnicode: string[];
  unicodeToByte: Map<string, number>;
};

function range(start: number, end: number): number[] {
  const values: number[] = [];
  for (let value = start; value <= end; value += 1) {
    values.push(value);
  }
  return values;
}

function createByteMaps(): ByteMaps {
  const byteValues = [...range(33, 126), ...range(161, 172), ...range(174, 255)];
  const unicodeValues = [...byteValues];

  let nextCodePoint = 256;
  for (let byte = 0; byte < 256; byte += 1) {
    if (byteValues.includes(byte)) {
      continue;
    }

    byteValues.push(byte);
    unicodeValues.push(nextCodePoint);
    nextCodePoint += 1;
  }

  const byteToUnicode: string[] = [];
  const unicodeToByte = new Map<string, number>();

  for (let index = 0; index < byteValues.length; index += 1) {
    const byteValue = byteValues[index];
    const unicodeValue = unicodeValues[index];
    if (byteValue === undefined || unicodeValue === undefined) {
      continue;
    }

    const char = String.fromCodePoint(unicodeValue);
    byteToUnicode[byteValue] = char;
    unicodeToByte.set(char, byteValue);
  }

  return { byteToUnicode, unicodeToByte };
}

const BYTE_MAPS = createByteMaps();

/** Convert raw bytes into the reversible GPT-2 byte-level alphabet. */
export function encodeByteLevelBytes(bytes: Uint8Array): string {
  const chars: string[] = [];
  for (const byte of bytes) {
    const mapped = BYTE_MAPS.byteToUnicode[byte];
    if (mapped === undefined) {
      throw new Error(`encodeByteLevelBytes: missing byte-level mapping for byte ${byte}`);
    }
    chars.push(mapped);
  }
  return chars.join("");
}

/** Convert UTF-8 bytes into the reversible GPT-2 byte-level alphabet. */
export function encodeByteLevelSegment(text: string): string {
  return encodeByteLevelBytes(new TextEncoder().encode(text));
}

/** Convert GPT-2 byte-level characters back into a string. */
export function decodeByteLevelTokens(tokens: string[]): string {
  const bytes: number[] = [];
  for (const token of tokens) {
    const byteFallbackMatch = token.match(/^<0x([0-9A-F]{2})>$/);
    if (byteFallbackMatch !== null) {
      const hex = byteFallbackMatch[1];
      if (hex !== undefined) {
        bytes.push(Number.parseInt(hex, 16));
      }
      continue;
    }

    for (const char of token) {
      const byte = BYTE_MAPS.unicodeToByte.get(char);
      if (byte === undefined) {
        throw new Error(`decodeByteLevelTokens: missing byte mapping for "${char}"`);
      }
      bytes.push(byte);
    }
  }
  return new TextDecoder().decode(new Uint8Array(bytes));
}

/** Split text using the canonical GPT-2 byte-level regex. */
export function splitByteLevelText(
  text: string,
  useRegex: boolean,
  pattern?: string,
): Array<[segment: string, start: number, end: number]> {
  if (!useRegex) {
    return text === "" ? [] : [[text, 0, text.length]];
  }

  const segments: Array<[string, number, number]> = [];
  const regex =
    pattern === undefined
      ? new RegExp(BYTE_LEVEL_REGEX.source, BYTE_LEVEL_REGEX.flags)
      : new RegExp(pattern, "gu");
  let match = regex.exec(text);
  while (match !== null) {
    const segment = match[0];
    const start = match.index;
    const end = start + segment.length;
    segments.push([segment, start, end]);
    match = regex.exec(text);
  }
  return segments;
}
