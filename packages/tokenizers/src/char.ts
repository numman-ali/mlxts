/**
 * Character-level tokenizer.
 *
 * Maps each unique character in the training text to an integer ID.
 * Simple, deterministic, and sufficient for Shakespeare training.
 *
 * @module
 */

/** Character-level tokenizer: one token per character. */
export class CharTokenizer {
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
  encode(text: string): number[] {
    const tokens: number[] = [];
    for (let i = 0; i < text.length; i++) {
      const char = text[i];
      if (char === undefined) continue;
      const id = this.#charToIndex.get(char);
      if (id === undefined) {
        throw new Error(
          `CharTokenizer.encode: unknown character '${char}' (code ${char.charCodeAt(0)}) at position ${i}`,
        );
      }
      tokens.push(id);
    }
    return tokens;
  }

  /** Decode an array of token IDs back to a string. */
  decode(tokens: number[]): string {
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

  /** Number of unique tokens in the vocabulary. */
  get vocabSize(): number {
    return this.#indexToChar.length;
  }

  /** Ordered character list for serialization. */
  get vocab(): string[] {
    return [...this.#indexToChar];
  }
}
