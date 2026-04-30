/**
 * CLIP vocab/merges BPE tokenizer.
 * @module
 */

import { UnsupportedTokenizerError } from "../errors";
import type {
  BatchEncoding,
  DecodeOptions,
  EncodeOptions,
  Encoding,
  Offset,
  Tokenizer,
} from "../tokenizer";
import { createMergeKey, findBestMerge, mergeWordPieces } from "./bpe-merges";
import { decodeByteLevelTokens, encodeByteLevelSegment } from "./byte-level";

const CLIP_SPLIT_REGEX =
  /<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+/giu;
const DEFAULT_BOS_TOKEN = "<|startoftext|>";
const DEFAULT_EOS_TOKEN = "<|endoftext|>";
const DEFAULT_MODEL_MAX_LENGTH = 77;
const END_OF_WORD_SUFFIX = "</w>";

export type CLIPTokenizerConfig = {
  vocab: Record<string, number>;
  merges: Array<[string, string]>;
  bosToken?: string;
  eosToken?: string;
  padToken?: string;
  unkToken?: string;
  modelMaxLength?: number;
};

export type CLIPTokenizerLoadOptions = {
  tokenizerConfig?: Record<string, unknown>;
  specialTokensMap?: Record<string, unknown>;
};

export type CLIPTextInput = {
  inputIds: number[];
  attentionMask: number[];
  truncated: boolean;
};

export type EncodeCLIPTextInputOptions = {
  maxLength?: number;
};

function expectRecord(value: unknown, context: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new UnsupportedTokenizerError(`${context} must be an object`);
  }
  return Object.fromEntries(Object.entries(value));
}

/** Parse a CLIP `vocab.json` payload into a token-to-id vocabulary. */
export function parseCLIPVocabJson(value: unknown): Record<string, number> {
  const raw = expectRecord(value, "vocab.json");
  const vocab: Record<string, number> = {};
  for (const [token, id] of Object.entries(raw)) {
    if (typeof id !== "number" || !Number.isInteger(id) || id < 0) {
      throw new UnsupportedTokenizerError(`vocab.json["${token}"] must be a non-negative integer`);
    }
    vocab[token] = id;
  }
  return vocab;
}

function tokenContent(value: unknown): string | undefined {
  if (typeof value === "string") {
    return value;
  }
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return undefined;
  }
  const raw = Object.fromEntries(Object.entries(value));
  return typeof raw.content === "string" ? raw.content : undefined;
}

function configuredToken(
  tokenizerConfig: Record<string, unknown>,
  specialTokensMap: Record<string, unknown>,
  key: "bos_token" | "eos_token" | "pad_token" | "unk_token",
  fallback: string,
): string {
  return tokenContent(tokenizerConfig[key]) ?? tokenContent(specialTokensMap[key]) ?? fallback;
}

function configuredModelMaxLength(tokenizerConfig: Record<string, unknown>): number {
  const value = tokenizerConfig.model_max_length;
  if (typeof value !== "number" || !Number.isInteger(value) || value <= 0) {
    return DEFAULT_MODEL_MAX_LENGTH;
  }
  return value > 1_000_000 ? DEFAULT_MODEL_MAX_LENGTH : value;
}

function requireTokenId(vocab: Map<string, number>, token: string, context: string): number {
  const id = vocab.get(token);
  if (id === undefined) {
    throw new UnsupportedTokenizerError(`${context} token "${token}" is not present in vocab.json`);
  }
  return id;
}

function normalizeCLIPText(text: string): string {
  return text.normalize("NFC").replace(/\s+/gu, " ").toLowerCase();
}

function createEncoding(
  ids: number[],
  offsets: Offset[] | undefined,
  specialTokensMask: number[],
): Encoding {
  const encoding: Encoding = { ids, specialTokensMask };
  if (offsets !== undefined) {
    encoding.offsets = offsets;
  }
  return encoding;
}

function parseMergeLine(line: string, lineNumber: number): [string, string] {
  const parts = line.trim().split(/\s+/u);
  const left = parts[0];
  const right = parts[1];
  if (left === undefined || right === undefined || parts.length !== 2) {
    throw new UnsupportedTokenizerError(`merges.txt line ${lineNumber} must contain two tokens`);
  }
  return [left, right];
}

/** Parse a CLIP `merges.txt` payload into ranked BPE pairs. */
export function parseCLIPMergesText(text: string): Array<[string, string]> {
  const merges: Array<[string, string]> = [];
  const lines = text.split(/\r?\n/u);
  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    if (line === undefined) {
      continue;
    }
    const trimmed = line.trim();
    if (trimmed === "" || trimmed.startsWith("#")) {
      continue;
    }
    merges.push(parseMergeLine(trimmed, index + 1));
  }
  return merges;
}

/** Load a CLIP tokenizer from parsed `vocab.json` and raw `merges.txt` content. */
export function loadCLIPTokenizer(
  vocabJson: unknown,
  mergesText: string,
  options: CLIPTokenizerLoadOptions = {},
): CLIPTokenizer {
  const vocab = parseCLIPVocabJson(vocabJson);
  const tokenizerConfig = options.tokenizerConfig ?? {};
  const specialTokensMap = options.specialTokensMap ?? {};
  return new CLIPTokenizer({
    vocab,
    merges: parseCLIPMergesText(mergesText),
    bosToken: configuredToken(tokenizerConfig, specialTokensMap, "bos_token", DEFAULT_BOS_TOKEN),
    eosToken: configuredToken(tokenizerConfig, specialTokensMap, "eos_token", DEFAULT_EOS_TOKEN),
    padToken: configuredToken(tokenizerConfig, specialTokensMap, "pad_token", DEFAULT_EOS_TOKEN),
    unkToken: configuredToken(tokenizerConfig, specialTokensMap, "unk_token", DEFAULT_EOS_TOKEN),
    modelMaxLength: configuredModelMaxLength(tokenizerConfig),
  });
}

/** Fixed-length CLIP prompt encoding used by diffusion text-conditioning paths. */
export function encodeCLIPTextInput(
  tokenizer: CLIPTokenizer,
  text: string,
  options: EncodeCLIPTextInputOptions = {},
): CLIPTextInput {
  const maxLength = options.maxLength ?? tokenizer.modelMaxLength;
  if (!Number.isInteger(maxLength) || maxLength < 2) {
    throw new Error("encodeCLIPTextInput: maxLength must be an integer greater than 1");
  }

  const eosTokenId = tokenizer.eosTokenIds[0];
  const padTokenId = tokenizer.padTokenId;
  if (eosTokenId === undefined || padTokenId === undefined) {
    throw new Error("encodeCLIPTextInput: tokenizer must define EOS and PAD token IDs");
  }

  const rawIds = tokenizer.encode(text, { addSpecialTokens: true });
  const truncated = rawIds.length > maxLength;
  const inputIds = truncated ? rawIds.slice(0, maxLength) : [...rawIds];
  if (truncated) {
    inputIds[maxLength - 1] = eosTokenId;
  }

  const attentionMask = inputIds.map(() => 1);
  while (inputIds.length < maxLength) {
    inputIds.push(padTokenId);
    attentionMask.push(0);
  }

  return { inputIds, attentionMask, truncated };
}

/** CLIP byte-level BPE tokenizer backed by Diffusers-style vocab/merges files. */
export class CLIPTokenizer implements Tokenizer {
  #vocab: Map<string, number>;
  #idToToken: string[];
  #merges: Map<string, number>;
  #specialIds: Set<number>;
  #bosToken: string;
  #eosToken: string;
  #padToken: string;
  #unkToken: string;
  #bosTokenId: number;
  #eosTokenId: number;
  #padTokenId: number;
  #unkTokenId: number;
  #modelMaxLength: number;
  #cache = new Map<string, string[]>();

  constructor(config: CLIPTokenizerConfig) {
    this.#vocab = new Map(Object.entries(config.vocab));
    this.#idToToken = [];
    for (const [token, id] of Object.entries(config.vocab)) {
      this.#idToToken[id] = token;
    }
    this.#merges = new Map();
    for (let index = 0; index < config.merges.length; index += 1) {
      const merge = config.merges[index];
      if (merge === undefined) {
        continue;
      }
      this.#merges.set(createMergeKey(merge[0], merge[1]), index);
    }

    this.#bosToken = config.bosToken ?? DEFAULT_BOS_TOKEN;
    this.#eosToken = config.eosToken ?? DEFAULT_EOS_TOKEN;
    this.#padToken = config.padToken ?? this.#eosToken;
    this.#unkToken = config.unkToken ?? this.#eosToken;
    this.#bosTokenId = requireTokenId(this.#vocab, this.#bosToken, "BOS");
    this.#eosTokenId = requireTokenId(this.#vocab, this.#eosToken, "EOS");
    this.#padTokenId = requireTokenId(this.#vocab, this.#padToken, "PAD");
    this.#unkTokenId = requireTokenId(this.#vocab, this.#unkToken, "UNK");
    this.#modelMaxLength = config.modelMaxLength ?? DEFAULT_MODEL_MAX_LENGTH;
    this.#specialIds = new Set([
      this.#bosTokenId,
      this.#eosTokenId,
      this.#padTokenId,
      this.#unkTokenId,
    ]);
  }

  get vocabSize(): number {
    return this.#idToToken.length;
  }

  get bosTokenId(): number {
    return this.#bosTokenId;
  }

  get eosTokenIds(): number[] {
    return [this.#eosTokenId];
  }

  get padTokenId(): number {
    return this.#padTokenId;
  }

  get modelMaxLength(): number {
    return this.#modelMaxLength;
  }

  encode(text: string, options: EncodeOptions = {}): number[] {
    return this.encodeWithOffsets(text, options).ids;
  }

  encodeWithOffsets(text: string, options: EncodeOptions = {}): Encoding {
    const ids: number[] = [];
    const offsets: Offset[] = [];
    const specialTokensMask: number[] = [];
    const returnOffsets = options.returnOffsets === true;
    const addSpecialTokens = options.addSpecialTokens ?? true;

    if (addSpecialTokens) {
      this.appendToken(
        ids,
        offsets,
        specialTokensMask,
        this.#bosTokenId,
        returnOffsets,
        true,
        0,
        0,
      );
    }

    const normalized = normalizeCLIPText(text);
    for (const match of normalized.matchAll(CLIP_SPLIT_REGEX)) {
      const segment = match[0];
      const start = match.index;
      const end = start + segment.length;
      const specialId = this.specialSegmentId(segment);
      if (specialId !== undefined) {
        this.appendToken(
          ids,
          offsets,
          specialTokensMask,
          specialId,
          returnOffsets,
          true,
          start,
          end,
        );
        continue;
      }

      for (const piece of this.segmentToPieces(segment)) {
        this.appendToken(
          ids,
          offsets,
          specialTokensMask,
          this.lookupPiece(piece),
          returnOffsets,
          false,
          start,
          end,
        );
      }
    }

    if (addSpecialTokens) {
      this.appendToken(
        ids,
        offsets,
        specialTokensMask,
        this.#eosTokenId,
        returnOffsets,
        true,
        0,
        0,
      );
    }

    return createEncoding(ids, returnOffsets ? offsets : undefined, specialTokensMask);
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
      const token = this.#idToToken[tokenId];
      if (token === undefined) {
        throw new Error(`CLIPTokenizer.decode: token ID ${tokenId} is out of range`);
      }
      tokens.push(token);
    }
    return decodeByteLevelTokens(tokens).replaceAll(END_OF_WORD_SUFFIX, " ").trim();
  }

  decodeBatch(batch: number[][], options: DecodeOptions = {}): string[] {
    return batch.map((entry) => this.decode(entry, options));
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

  private specialSegmentId(segment: string): number | undefined {
    if (segment === this.#bosToken) {
      return this.#bosTokenId;
    }
    if (segment === this.#eosToken) {
      return this.#eosTokenId;
    }
    if (segment === this.#padToken) {
      return this.#padTokenId;
    }
    if (segment === this.#unkToken) {
      return this.#unkTokenId;
    }
    return undefined;
  }

  private segmentToPieces(segment: string): string[] {
    const encoded = encodeByteLevelSegment(segment);
    const cached = this.#cache.get(encoded);
    if (cached !== undefined) {
      return [...cached];
    }

    const chars = Array.from(encoded);
    if (chars.length === 0) {
      return [];
    }

    const last = chars[chars.length - 1];
    if (last === undefined) {
      return [];
    }
    const word = [...chars.slice(0, -1), `${last}${END_OF_WORD_SUFFIX}`];
    let pieces = word;
    let bestMerge = findBestMerge(pieces, this.#merges);
    while (bestMerge !== null) {
      pieces = mergeWordPieces(pieces, bestMerge);
      bestMerge = findBestMerge(pieces, this.#merges);
    }
    this.#cache.set(encoded, pieces);
    return [...pieces];
  }

  private lookupPiece(piece: string): number {
    return this.#vocab.get(piece) ?? this.#unkTokenId;
  }
}
