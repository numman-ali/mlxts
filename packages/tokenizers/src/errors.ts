/**
 * Tokenizer-specific error types.
 * @module
 */

/** Clear error for tokenizer files that exist but are outside the supported Phase 7 subset. */
export class UnsupportedTokenizerError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "UnsupportedTokenizerError";
  }
}
