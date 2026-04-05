/**
 * Shared tokenizer interfaces and result types.
 * @module
 */

export type Offset = {
  start: number;
  end: number;
};

export type EncodeOptions = {
  addSpecialTokens?: boolean;
  returnOffsets?: boolean;
};

export type DecodeOptions = {
  skipSpecialTokens?: boolean;
};

export type Encoding = {
  ids: number[];
  offsets?: Offset[];
  specialTokensMask?: number[];
};

export type BatchEncoding = Encoding[];

/**
 * Shared tokenizer contract for decoder-style language models.
 *
 * The compact `encode()` / `decode()` helpers stay friendly for existing
 * call sites while `encodeWithOffsets()` and `encodeBatch()` expose the richer
 * metadata Phase 7 needs.
 */
export interface Tokenizer {
  readonly vocabSize: number;
  readonly bosTokenId: number | undefined;
  readonly eosTokenIds: number[];
  readonly padTokenId: number | undefined;

  encode(text: string, options?: EncodeOptions): number[];
  encodeWithOffsets(text: string, options?: EncodeOptions): Encoding;
  encodeBatch(texts: string[], options?: EncodeOptions): BatchEncoding;
  decode(tokenIds: number[], options?: DecodeOptions): string;
  decodeBatch(batch: number[][], options?: DecodeOptions): string[];
}
