/**
 * Minimal SentencePiece unigram tokenizer for Phase 7.
 * @module
 */

import { UnsupportedTokenizerError } from "./errors";
import type { SentencePieceEntry } from "./sentencepiece-proto";
import { parseSentencePieceModel } from "./sentencepiece-proto";
import type {
  BatchEncoding,
  DecodeOptions,
  EncodeOptions,
  Encoding,
  Offset,
  Tokenizer,
} from "./tokenizer";

type SentencePieceConfig = {
  pieces: SentencePieceEntry[];
  byteFallback: boolean;
  unkId?: number;
  bosId?: number;
  eosId?: number;
  padId?: number;
  addDummyPrefix?: boolean;
  removeExtraWhitespaces?: boolean;
  escapeWhitespaces?: boolean;
};

type IndexedPiece = {
  id: number;
  piece: string;
  score: number;
};

function normalizeWhitespace(text: string): string {
  return text.replace(/\s+/g, " ").trim();
}

function splitToChars(text: string): string[] {
  return Array.from(text);
}

function sanitizeOffsets(offsets: Offset[]): Offset[] | undefined {
  return offsets.length === 0 ? undefined : offsets;
}

function createEncoding(
  ids: number[],
  offsets: Offset[] | undefined,
  specialTokensMask: number[],
): Encoding {
  const encoding: Encoding = {
    ids,
    specialTokensMask,
  };
  if (offsets !== undefined) {
    encoding.offsets = offsets;
  }
  return encoding;
}

/** SentencePiece unigram tokenizer with byte fallback support. */
export class SentencePieceTokenizer implements Tokenizer {
  #pieces: string[];
  #pieceIds: Map<string, number>;
  #piecesByFirstChar: Map<string, IndexedPiece[]>;
  #specialIds: Set<number>;
  #byteFallback: boolean;
  #unkId: number | undefined;
  #bosId: number | undefined;
  #eosIds: number[];
  #padId: number | undefined;
  #addDummyPrefix: boolean;
  #removeExtraWhitespaces: boolean;
  #escapeWhitespaces: boolean;

  constructor(config: SentencePieceConfig) {
    this.#pieces = config.pieces.map((entry) => entry.piece);
    this.#pieceIds = new Map();
    this.#piecesByFirstChar = new Map();

    for (let id = 0; id < config.pieces.length; id += 1) {
      const entry = config.pieces[id];
      if (entry === undefined) {
        continue;
      }
      this.#pieceIds.set(entry.piece, id);
      const firstChar = Array.from(entry.piece)[0];
      if (firstChar === undefined) {
        continue;
      }

      const indexed: IndexedPiece = { id, piece: entry.piece, score: entry.score };
      const bucket = this.#piecesByFirstChar.get(firstChar);
      if (bucket === undefined) {
        this.#piecesByFirstChar.set(firstChar, [indexed]);
      } else {
        bucket.push(indexed);
      }
    }

    for (const bucket of this.#piecesByFirstChar.values()) {
      bucket.sort((left, right) => right.piece.length - left.piece.length);
    }

    this.#specialIds = new Set(
      config.pieces.flatMap((entry, id) =>
        entry.type === 2 || entry.type === 3 || entry.type === 4 ? [id] : [],
      ),
    );
    this.#byteFallback = config.byteFallback;
    this.#unkId = config.unkId;
    this.#bosId = config.bosId;
    this.#eosIds = config.eosId === undefined ? [] : [config.eosId];
    this.#padId = config.padId;
    this.#addDummyPrefix = config.addDummyPrefix ?? true;
    this.#removeExtraWhitespaces = config.removeExtraWhitespaces ?? true;
    this.#escapeWhitespaces = config.escapeWhitespaces ?? true;
  }

  static fromModelBytes(bytes: Uint8Array): SentencePieceTokenizer {
    const model = parseSentencePieceModel(bytes);
    if (model.modelType !== undefined && model.modelType !== 1) {
      throw new UnsupportedTokenizerError(
        `SentencePiece model_type ${model.modelType} is not supported; only unigram models are supported`,
      );
    }

    const config: SentencePieceConfig = {
      pieces: model.pieces,
      byteFallback: model.byteFallback,
      addDummyPrefix: model.addDummyPrefix,
      removeExtraWhitespaces: model.removeExtraWhitespaces,
      escapeWhitespaces: model.escapeWhitespaces,
    };
    if (model.unkId !== undefined) {
      config.unkId = model.unkId;
    }
    if (model.bosId !== undefined) {
      config.bosId = model.bosId;
    }
    if (model.eosId !== undefined) {
      config.eosId = model.eosId;
    }
    if (model.padId !== undefined) {
      config.padId = model.padId;
    }
    return new SentencePieceTokenizer(config);
  }

  get vocabSize(): number {
    return this.#pieces.length;
  }

  get bosTokenId(): number | undefined {
    return this.#bosId;
  }

  get eosTokenIds(): number[] {
    return [...this.#eosIds];
  }

  get padTokenId(): number | undefined {
    return this.#padId;
  }

  encode(text: string, options: EncodeOptions = {}): number[] {
    return this.encodeWithOffsets(text, options).ids;
  }

  private appendSpecialToken(
    ids: number[],
    offsets: Offset[],
    specialTokensMask: number[],
    tokenId: number,
    returnOffsets: boolean,
  ): void {
    ids.push(tokenId);
    specialTokensMask.push(1);
    if (returnOffsets) {
      offsets.push({ start: 0, end: 0 });
    }
  }

  private appendTokenIds(
    ids: number[],
    offsets: Offset[],
    specialTokensMask: number[],
    tokenIds: readonly number[],
    start: number,
    consumed: number,
    returnOffsets: boolean,
  ): void {
    for (const tokenId of tokenIds) {
      ids.push(tokenId);
      specialTokensMask.push(0);
      if (returnOffsets) {
        offsets.push({
          start: Math.max(0, start - 1),
          end: Math.max(0, start + consumed - 1),
        });
      }
    }
  }

  encodeWithOffsets(text: string, options: EncodeOptions = {}): Encoding {
    const normalizedText = this.normalize(text);
    const chars = splitToChars(normalizedText);
    const ids: number[] = [];
    const offsets: Offset[] = [];
    const specialTokensMask: number[] = [];
    const addSpecialTokens = options.addSpecialTokens ?? true;
    const returnOffsets = options.returnOffsets === true;

    if (addSpecialTokens && this.#bosId !== undefined) {
      this.appendSpecialToken(ids, offsets, specialTokensMask, this.#bosId, returnOffsets);
    }

    let position = 0;
    while (position < chars.length) {
      const [tokenIds, consumed] = this.segment(chars, position);
      this.appendTokenIds(
        ids,
        offsets,
        specialTokensMask,
        tokenIds,
        position,
        consumed,
        returnOffsets,
      );
      position += consumed;
    }

    if (addSpecialTokens) {
      for (const eosId of this.#eosIds) {
        this.appendSpecialToken(ids, offsets, specialTokensMask, eosId, returnOffsets);
      }
    }

    return createEncoding(ids, sanitizeOffsets(offsets), specialTokensMask);
  }

  encodeBatch(texts: string[], options: EncodeOptions = {}): BatchEncoding {
    return texts.map((text) => this.encodeWithOffsets(text, options));
  }

  decode(tokenIds: number[], options: DecodeOptions = {}): string {
    const tokens: string[] = [];
    for (const tokenId of tokenIds) {
      if (options.skipSpecialTokens === true && this.#specialIds.has(tokenId)) {
        continue;
      }
      const piece = this.#pieces[tokenId];
      if (piece === undefined) {
        throw new Error(`SentencePieceTokenizer.decode: token ID ${tokenId} is out of range`);
      }
      tokens.push(piece);
    }

    const byteBuffer: number[] = [];
    const decodedPieces: string[] = [];
    for (const token of tokens) {
      const byteFallback = token.match(/^<0x([0-9A-F]{2})>$/);
      if (byteFallback !== null) {
        const hex = byteFallback[1];
        if (hex !== undefined) {
          byteBuffer.push(Number.parseInt(hex, 16));
        }
        continue;
      }

      if (byteBuffer.length > 0) {
        decodedPieces.push(new TextDecoder().decode(new Uint8Array(byteBuffer)));
        byteBuffer.length = 0;
      }
      decodedPieces.push(token);
    }

    if (byteBuffer.length > 0) {
      decodedPieces.push(new TextDecoder().decode(new Uint8Array(byteBuffer)));
    }

    return decodedPieces.join("").replaceAll("▁", " ").replace(/^ /, "");
  }

  decodeBatch(batch: number[][], options: DecodeOptions = {}): string[] {
    return batch.map((entry) => this.decode(entry, options));
  }

  private normalize(text: string): string {
    const base = this.#removeExtraWhitespaces ? normalizeWhitespace(text) : text;
    const withPrefix = this.#addDummyPrefix ? ` ${base}` : base;
    return this.#escapeWhitespaces ? withPrefix.replaceAll(" ", "▁") : withPrefix;
  }

  private candidateBucket(chars: string[], start: number): IndexedPiece[] {
    const firstChar = chars[start];
    return firstChar === undefined ? [] : (this.#piecesByFirstChar.get(firstChar) ?? []);
  }

  private candidateMatches(chars: string[], start: number, pieceChars: string[]): boolean {
    if (pieceChars.length === 0 || start + pieceChars.length > chars.length) {
      return false;
    }

    for (let index = 0; index < pieceChars.length; index += 1) {
      if (chars[start + index] !== pieceChars[index]) {
        return false;
      }
    }
    return true;
  }

  private bestCandidate(chars: string[], start: number): [number | undefined, number] {
    let bestId: number | undefined;
    let bestScore = Number.NEGATIVE_INFINITY;
    let bestLength = 0;

    for (const candidate of this.candidateBucket(chars, start)) {
      const pieceChars = splitToChars(candidate.piece);
      if (!this.candidateMatches(chars, start, pieceChars)) {
        continue;
      }

      if (candidate.score > bestScore) {
        bestId = candidate.id;
        bestScore = candidate.score;
        bestLength = pieceChars.length;
      }
    }

    return [bestId, bestLength];
  }

  private segment(chars: string[], start: number): [number[], number] {
    const [bestId, bestLength] = this.bestCandidate(chars, start);
    if (bestId !== undefined && bestLength > 0) {
      return [[bestId], bestLength];
    }

    const byteTokens = this.byteFallbackIds(chars[start]);
    if (byteTokens.length > 0) {
      return [byteTokens, 1];
    }

    if (this.#unkId === undefined) {
      throw new Error(
        "SentencePieceTokenizer.encode: encountered an unknown token but unk_id is missing",
      );
    }
    return [[this.#unkId], 1];
  }

  private byteFallbackIds(char: string | undefined): number[] {
    if (!this.#byteFallback || char === undefined) {
      return [];
    }

    const ids: number[] = [];
    const bytes = new TextEncoder().encode(char);
    for (const byte of bytes) {
      const token = `<0x${byte.toString(16).toUpperCase().padStart(2, "0")}>`;
      const id = this.#pieceIds.get(token);
      if (id === undefined) {
        return [];
      }
      ids.push(id);
    }
    return ids;
  }
}
