/**
 * Character-level tokenizer.
 *
 * Maps each unique character in the training text to an integer ID.
 * Simple, deterministic, and sufficient for Shakespeare training.
 *
 * @module
 */

import type {
  BatchEncoding,
  DecodeOptions,
  EncodeOptions,
  Encoding,
  Offset,
  Tokenizer,
} from "./tokenizer";

function sanitizeOffsets(offsets: Offset[]): Offset[] | undefined {
  return offsets.length === 0 ? undefined : offsets;
}

function createEncoding(ids: number[], offsets: Offset[] | undefined): Encoding {
  const encoding: Encoding = {
    ids,
    specialTokensMask: ids.map(() => 0),
  };
  if (offsets !== undefined) {
    encoding.offsets = offsets;
  }
  return encoding;
}

/** Character-level tokenizer: one token per character. */
export class CharTokenizer implements Tokenizer {
  #charToIndex: Map<string, number>;
  #indexToChar: string[];

  private constructor(chars: string[]) {
    this.#indexToChar = chars;
    this.#charToIndex = new Map();
    for (let i = 0; i < chars.length; i++) {
      const c = chars[i];
      if (c !== undefined) {
        this.#charToIndex.set(c, i);
      }
    }
  }

  /** Build a tokenizer from training text. Characters are sorted for determinism. */
  static fromText(text: string): CharTokenizer {
    const unique = [...new Set(text)].sort();
    return new CharTokenizer(unique);
  }

  /** Restore a tokenizer from a saved vocabulary (e.g., from a checkpoint). */
  static fromVocab(chars: string[]): CharTokenizer {
    return new CharTokenizer([...chars]);
  }

  /** Encode a string to an array of integer token IDs. */
  encode(text: string, _options: EncodeOptions = {}): number[] {
    const tokens: number[] = [];
    for (const [index, char] of Array.from(text).entries()) {
      const id = this.#charToIndex.get(char);
      if (id === undefined) {
        throw new Error(
          `CharTokenizer.encode: unknown character '${char}' (code ${char.codePointAt(0) ?? 0}) at position ${index}`,
        );
      }
      tokens.push(id);
    }
    return tokens;
  }

  /** Encode text and optionally keep per-token character offsets. */
  encodeWithOffsets(text: string, options: EncodeOptions = {}): Encoding {
    const ids = this.encode(text, options);
    const offsets: Offset[] = [];
    if (options.returnOffsets === true) {
      let offset = 0;
      for (const char of text) {
        const width = char.length;
        offsets.push({ start: offset, end: offset + width });
        offset += width;
      }
    }
    return createEncoding(ids, sanitizeOffsets(offsets));
  }

  /** Encode a batch of strings. */
  encodeBatch(texts: string[], options: EncodeOptions = {}): BatchEncoding {
    return texts.map((text) => this.encodeWithOffsets(text, options));
  }

  /** Decode an array of token IDs back to a string. */
  decode(tokens: number[], _options: DecodeOptions = {}): string {
    const chars: string[] = [];
    for (let i = 0; i < tokens.length; i++) {
      const id = tokens[i];
      if (id === undefined) continue;
      const char = this.#indexToChar[id];
      if (char === undefined) {
        throw new Error(
          `CharTokenizer.decode: token ID ${id} is out of range [0, ${this.#indexToChar.length})`,
        );
      }
      chars.push(char);
    }
    return chars.join("");
  }

  /** Decode a batch of token ID arrays. */
  decodeBatch(batch: number[][], options: DecodeOptions = {}): string[] {
    return batch.map((tokens) => this.decode(tokens, options));
  }

  /** Number of unique tokens in the vocabulary. */
  get vocabSize(): number {
    return this.#indexToChar.length;
  }

  /** Character tokenizer does not define a BOS token. */
  get bosTokenId(): number | undefined {
    return undefined;
  }

  /** Character tokenizer does not define EOS tokens. */
  get eosTokenIds(): number[] {
    return [];
  }

  /** Character tokenizer does not define a PAD token. */
  get padTokenId(): number | undefined {
    return undefined;
  }

  /** Ordered character list for serialization. */
  get vocab(): string[] {
    return [...this.#indexToChar];
  }
}
