/**
 * HuggingFace-compatible BPE tokenizers for the supported Phase 7 subset.
 * @module
 */

import { sortAddedTokenMatches, splitInputByAddedTokens } from "./bpe-added-tokens";
import { createMergeKey, findBestMerge, mergeWordPieces } from "./bpe-merges";
import { decodeByteLevelTokens, encodeByteLevelSegment, splitByteLevelText } from "./byte-level";
import type {
  BatchEncoding,
  DecodeOptions,
  EncodeOptions,
  Encoding,
  Offset,
  Tokenizer,
} from "./tokenizer";

export type AddedToken = {
  content: string;
  id: number;
  special: boolean;
};

export type BPEVariant = "bytelevel" | "sentencepiece";

export type BPEConfig = {
  vocab: Record<string, number>;
  merges: Array<[string, string]>;
  variant: BPEVariant;
  addedTokens: AddedToken[];
  bosTokenId?: number;
  eosTokenIds?: number[];
  padTokenId?: number;
  addBosToken?: boolean;
  addEosToken?: boolean;
  addPrefixSpace?: boolean;
  useRegex?: boolean;
  splitPattern?: string;
  byteFallback?: boolean;
  unkTokenId?: number;
};

function denormalizeSentencePieceText(text: string): string {
  return text.replaceAll("▁", " ").replace(/^ /, "");
}

function normalizeSentencePieceSegment(text: string, isLeadingSegment: boolean): string {
  const replaced = text.replaceAll(" ", "▁");
  return isLeadingSegment ? `▁${replaced}` : replaced;
}

function addedTokenMap(addedTokens: AddedToken[]): Map<string, AddedToken> {
  const result = new Map<string, AddedToken>();
  for (const token of addedTokens) {
    result.set(token.content, token);
  }
  return result;
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

/** Supported BPE tokenizer for ByteLevel and sentencepiece-style JSON tokenizers. */
export class BPETokenizer implements Tokenizer {
  #variant: BPEVariant;
  #vocab: Map<string, number>;
  #idToToken: string[];
  #merges: Map<string, number>;
  #addedTokens: Map<string, AddedToken>;
  #addedTokenMatches: AddedToken[];
  #specialIds: Set<number>;
  #bosTokenId: number | undefined;
  #eosTokenIds: number[];
  #padTokenId: number | undefined;
  #addBosToken: boolean;
  #addEosToken: boolean;
  #addPrefixSpace: boolean;
  #useRegex: boolean;
  #splitPattern: string | undefined;
  #byteFallback: boolean;
  #unkTokenId: number | undefined;

  constructor(config: BPEConfig) {
    this.#variant = config.variant;
    this.#vocab = new Map(Object.entries(config.vocab));
    this.#idToToken = [];
    for (const [token, id] of Object.entries(config.vocab)) {
      this.#idToToken[id] = token;
    }
    for (const token of config.addedTokens) {
      this.#idToToken[token.id] = token.content;
    }
    this.#merges = new Map();
    for (let index = 0; index < config.merges.length; index += 1) {
      const merge = config.merges[index];
      if (merge === undefined) {
        continue;
      }
      const left = merge[0];
      const right = merge[1];
      if (left === undefined || right === undefined) {
        continue;
      }
      this.#merges.set(createMergeKey(left, right), index);
    }
    this.#addedTokens = addedTokenMap(config.addedTokens);
    this.#addedTokenMatches = sortAddedTokenMatches(config.addedTokens);
    this.#specialIds = new Set(
      config.addedTokens.filter((token) => token.special).map((token) => token.id),
    );
    this.#bosTokenId = config.bosTokenId;
    this.#eosTokenIds = config.eosTokenIds ?? [];
    this.#padTokenId = config.padTokenId;
    this.#addBosToken = config.addBosToken ?? false;
    this.#addEosToken = config.addEosToken ?? false;
    this.#addPrefixSpace = config.addPrefixSpace ?? false;
    this.#useRegex = config.useRegex ?? true;
    this.#splitPattern = config.splitPattern;
    this.#byteFallback = config.byteFallback ?? false;
    this.#unkTokenId = config.unkTokenId;
  }

  get vocabSize(): number {
    return this.#idToToken.length;
  }

  get bosTokenId(): number | undefined {
    return this.#bosTokenId;
  }

  get eosTokenIds(): number[] {
    return [...this.#eosTokenIds];
  }

  get padTokenId(): number | undefined {
    return this.#padTokenId;
  }

  encode(text: string, options: EncodeOptions = {}): number[] {
    return this.encodeWithOffsets(text, options).ids;
  }

  private appendToken(
    ids: number[],
    offsets: Offset[],
    specialTokensMask: number[],
    tokenId: number,
    returnOffsets: boolean,
    special: boolean,
    start: number,
    end: number,
  ): void {
    ids.push(tokenId);
    specialTokensMask.push(special ? 1 : 0);
    if (returnOffsets) {
      offsets.push({ start, end });
    }
  }

  private encodeByteLevel(
    text: string,
    textStart: number,
    ids: number[],
    offsets: Offset[],
    specialTokensMask: number[],
    returnOffsets: boolean,
  ): void {
    if (text === "") {
      return;
    }

    const normalized =
      this.#addPrefixSpace && textStart === 0 && !/^\s/.test(text) ? ` ${text}` : text;
    const leadingPrefixSize = normalized.length - text.length;
    const segments = splitByteLevelText(normalized, this.#useRegex, this.#splitPattern);
    for (const [segment, start, end] of segments) {
      const encoded = encodeByteLevelSegment(segment);
      const tokens = this.segmentToTokens(encoded);
      const adjustedStart = Math.max(textStart, textStart + start - leadingPrefixSize);
      const adjustedEnd = Math.max(adjustedStart, textStart + end - leadingPrefixSize);
      for (const tokenId of tokens) {
        this.appendToken(
          ids,
          offsets,
          specialTokensMask,
          tokenId,
          returnOffsets,
          false,
          adjustedStart,
          adjustedEnd,
        );
      }
    }
  }

  private encodeSentencePiece(
    text: string,
    textStart: number,
    ids: number[],
    offsets: Offset[],
    specialTokensMask: number[],
    returnOffsets: boolean,
  ): void {
    if (text === "") {
      return;
    }

    const normalized = normalizeSentencePieceSegment(text, textStart === 0);
    const leadingPrefixSize = textStart === 0 ? 1 : 0;
    const chars = Array.from(normalized);
    let cursor = 0;
    while (cursor < chars.length) {
      const [tokenStrings, consumed] = this.segmentSentencePiece(chars, cursor);
      const adjustedStart = Math.max(textStart, textStart + cursor - leadingPrefixSize);
      const adjustedEnd = Math.max(
        adjustedStart,
        textStart + cursor + consumed - leadingPrefixSize,
      );
      for (const tokenString of tokenStrings) {
        this.appendToken(
          ids,
          offsets,
          specialTokensMask,
          this.lookupTokenId(tokenString),
          returnOffsets,
          false,
          adjustedStart,
          adjustedEnd,
        );
      }
      cursor += consumed;
    }
  }

  encodeWithOffsets(text: string, options: EncodeOptions = {}): Encoding {
    const ids: number[] = [];
    const offsets: Offset[] = [];
    const specialTokensMask: number[] = [];
    const returnOffsets = options.returnOffsets === true;
    const addSpecialTokens = options.addSpecialTokens ?? true;

    if (addSpecialTokens) {
      this.pushLeadingSpecialTokens(ids, offsets, specialTokensMask, returnOffsets);
    }

    const chunks = splitInputByAddedTokens(text, this.#addedTokenMatches);
    for (const chunk of chunks) {
      if (chunk.kind === "added-token") {
        this.appendToken(
          ids,
          offsets,
          specialTokensMask,
          chunk.token.id,
          returnOffsets,
          chunk.token.special,
          chunk.start,
          chunk.end,
        );
        continue;
      }

      if (this.#variant === "bytelevel") {
        this.encodeByteLevel(
          chunk.text,
          chunk.start,
          ids,
          offsets,
          specialTokensMask,
          returnOffsets,
        );
      } else {
        this.encodeSentencePiece(
          chunk.text,
          chunk.start,
          ids,
          offsets,
          specialTokensMask,
          returnOffsets,
        );
      }
    }

    if (addSpecialTokens) {
      this.pushTrailingSpecialTokens(ids, offsets, specialTokensMask, returnOffsets);
    }

    return createEncoding(ids, sanitizeOffsets(offsets), specialTokensMask);
  }

  encodeBatch(texts: string[], options: EncodeOptions = {}): BatchEncoding {
    return texts.map((text) => this.encodeWithOffsets(text, options));
  }

  decode(tokenIds: number[], options: DecodeOptions = {}): string {
    const tokenStrings: string[] = [];
    for (const tokenId of tokenIds) {
      if (options.skipSpecialTokens === true && this.#specialIds.has(tokenId)) {
        continue;
      }

      const token = this.#idToToken[tokenId];
      if (token === undefined) {
        throw new Error(`BPETokenizer.decode: token ID ${tokenId} is out of range`);
      }
      tokenStrings.push(token);
    }

    if (this.#variant === "bytelevel") {
      return decodeByteLevelTokens(tokenStrings);
    }

    return denormalizeSentencePieceText(tokenStrings.join(""));
  }

  decodeBatch(batch: number[][], options: DecodeOptions = {}): string[] {
    return batch.map((entry) => this.decode(entry, options));
  }

  private pushLeadingSpecialTokens(
    ids: number[],
    offsets: Offset[],
    mask: number[],
    returnOffsets: boolean,
  ): void {
    if (this.#addBosToken && this.#bosTokenId !== undefined) {
      this.appendToken(ids, offsets, mask, this.#bosTokenId, returnOffsets, true, 0, 0);
    }
  }

  private pushTrailingSpecialTokens(
    ids: number[],
    offsets: Offset[],
    mask: number[],
    returnOffsets: boolean,
  ): void {
    if (!this.#addEosToken) {
      return;
    }

    for (const eosTokenId of this.#eosTokenIds) {
      this.appendToken(ids, offsets, mask, eosTokenId, returnOffsets, true, 0, 0);
    }
  }

  private segmentToTokens(segment: string): number[] {
    const pieces = this.segmentToPieceStrings(segment);
    return pieces.map((piece) => this.lookupTokenId(piece));
  }

  private segmentToPieceStrings(segment: string): string[] {
    if (segment === "") {
      return [];
    }

    if (this.#vocab.has(segment)) {
      return [segment];
    }

    let word = Array.from(segment);
    let bestMerge = findBestMerge(word, this.#merges);
    while (bestMerge !== null) {
      word = mergeWordPieces(word, bestMerge);
      bestMerge = findBestMerge(word, this.#merges);
    }

    return word;
  }

  private segmentSentencePiece(chars: string[], start: number): [string[], number] {
    let bestToken: string | undefined;
    let bestLength = 0;

    for (let end = chars.length; end > start; end -= 1) {
      const candidate = chars.slice(start, end).join("");
      if (this.#vocab.has(candidate)) {
        bestToken = candidate;
        bestLength = end - start;
        break;
      }
    }

    if (bestToken !== undefined) {
      return [[bestToken], bestLength];
    }

    const fallbackTokens = this.byteFallbackTokens(chars[start]);
    if (fallbackTokens.length > 0) {
      return [fallbackTokens, 1];
    }

    if (this.#unkTokenId === undefined) {
      throw new Error("BPETokenizer.encode: encountered unknown segment without an unk token");
    }

    const unkToken = this.#idToToken[this.#unkTokenId];
    if (unkToken === undefined) {
      throw new Error(
        `BPETokenizer.encode: unk token ID ${this.#unkTokenId} is not in the vocabulary`,
      );
    }
    return [[unkToken], 1];
  }

  private byteFallbackTokens(char: string | undefined): string[] {
    if (!this.#byteFallback || char === undefined) {
      return [];
    }

    const bytes = new TextEncoder().encode(char);
    const pieces: string[] = [];
    for (const byte of bytes) {
      const token = `<0x${byte.toString(16).toUpperCase().padStart(2, "0")}>`;
      if (!this.#vocab.has(token)) {
        return [];
      }
      pieces.push(token);
    }
    return pieces;
  }

  private lookupTokenId(token: string): number {
    const added = this.#addedTokens.get(token);
    if (added !== undefined) {
      return added.id;
    }

    const tokenId = this.#vocab.get(token);
    if (tokenId !== undefined) {
      return tokenId;
    }

    const fallbackTokens = this.byteFallbackTokens(token);
    if (fallbackTokens.length === 1) {
      return this.lookupTokenId(fallbackTokens[0] ?? token);
    }

    if (this.#unkTokenId !== undefined) {
      return this.#unkTokenId;
    }

    throw new Error(`BPETokenizer.encode: token "${token}" is not in the vocabulary`);
  }
}
