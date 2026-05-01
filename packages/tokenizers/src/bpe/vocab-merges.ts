/**
 * Byte-level vocab/merges BPE loading for GPT/Qwen-style tokenizers.
 * @module
 */

import { UnsupportedTokenizerError } from "../errors";
import { type AddedToken, type BPEConfig, BPETokenizer } from "./bpe-base";

const QWEN2_PRETOKENIZE_REGEX = String.raw`(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`;

export type ByteLevelBPEVocabMergesLoadOptions = {
  tokenizerConfig?: Record<string, unknown>;
  specialTokensMap?: Record<string, unknown>;
  addedTokens?: Record<string, unknown>;
};

function expectRecord(value: unknown, context: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new UnsupportedTokenizerError(`${context} must be an object`);
  }
  return Object.fromEntries(Object.entries(value));
}

function expectString(value: unknown, context: string): string {
  if (typeof value !== "string") {
    throw new UnsupportedTokenizerError(`${context} must be a string`);
  }
  return value;
}

function parseVocab(value: unknown): Record<string, number> {
  const raw = expectRecord(value, "tokenizer.model.vocab");
  const vocab: Record<string, number> = {};
  for (const [token, id] of Object.entries(raw)) {
    if (typeof id !== "number" || !Number.isInteger(id) || id < 0) {
      throw new UnsupportedTokenizerError(
        `tokenizer.model.vocab["${token}"] must be a non-negative integer`,
      );
    }
    vocab[token] = id;
  }
  return vocab;
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

function parseMergesText(text: string): Array<[string, string]> {
  const merges: Array<[string, string]> = [];
  const lines = text.split(/\r?\n/u);
  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    if (line === undefined) {
      continue;
    }
    const trimmed = line.trim();
    if (trimmed !== "" && !trimmed.startsWith("#")) {
      merges.push(parseMergeLine(trimmed, index + 1));
    }
  }
  return merges;
}

function parseSpecialTokenId(
  tokenValue: unknown,
  vocab: Record<string, number>,
  addedTokens: readonly AddedToken[],
): number | undefined {
  const addedByContent = new Map<string, number>();
  for (const token of addedTokens) {
    addedByContent.set(token.content, token.id);
  }
  if (typeof tokenValue === "string") {
    return addedByContent.get(tokenValue) ?? vocab[tokenValue];
  }
  if (typeof tokenValue === "number" && Number.isInteger(tokenValue) && tokenValue >= 0) {
    return tokenValue;
  }
  if (typeof tokenValue !== "object" || tokenValue === null || Array.isArray(tokenValue)) {
    return undefined;
  }
  const raw = Object.fromEntries(Object.entries(tokenValue));
  return typeof raw.content === "string"
    ? (addedByContent.get(raw.content) ?? vocab[raw.content])
    : undefined;
}

function configuredSpecialTokenId(
  tokenizerConfig: Record<string, unknown>,
  specialTokensMap: Record<string, unknown>,
  vocab: Record<string, number>,
  addedTokens: readonly AddedToken[],
  key: "bos_token" | "eos_token" | "pad_token" | "unk_token",
): number | undefined {
  return (
    parseSpecialTokenId(tokenizerConfig[key], vocab, addedTokens) ??
    parseSpecialTokenId(tokenizerConfig[`${key}_id`], vocab, addedTokens) ??
    parseSpecialTokenId(specialTokensMap[key], vocab, addedTokens)
  );
}

function specialTokenContent(value: unknown): string | undefined {
  if (typeof value === "string") {
    return value;
  }
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return undefined;
  }
  const raw = Object.fromEntries(Object.entries(value));
  return typeof raw.content === "string" ? raw.content : undefined;
}

function collectSpecialTokenContents(
  tokenizerConfig: Record<string, unknown>,
  specialTokensMap: Record<string, unknown>,
): Set<string> {
  const contents = new Set<string>();
  for (const source of [tokenizerConfig, specialTokensMap]) {
    for (const key of ["bos_token", "eos_token", "pad_token", "unk_token"]) {
      const content = specialTokenContent(source[key]);
      if (content !== undefined) {
        contents.add(content);
      }
    }
    const additional = source.additional_special_tokens;
    if (Array.isArray(additional)) {
      for (const entry of additional) {
        const content = specialTokenContent(entry);
        if (content !== undefined) {
          contents.add(content);
        }
      }
    }
  }
  return contents;
}

function parseAddedTokensDecoder(value: unknown): AddedToken[] {
  if (value === undefined || value === null) {
    return [];
  }
  const raw = expectRecord(value, "tokenizer_config.json.added_tokens_decoder");
  const tokens: AddedToken[] = [];
  for (const [idText, entry] of Object.entries(raw)) {
    const id = Number(idText);
    if (!Number.isInteger(id) || id < 0) {
      throw new UnsupportedTokenizerError(
        `tokenizer_config.json.added_tokens_decoder["${idText}"] must use a non-negative integer key`,
      );
    }
    const record = expectRecord(entry, `tokenizer_config.json.added_tokens_decoder["${idText}"]`);
    tokens.push({
      content: expectString(
        record.content,
        `tokenizer_config.json.added_tokens_decoder["${idText}"].content`,
      ),
      id,
      special: record.special === true,
    });
  }
  return tokens;
}

function configuredAddedTokens(
  value: unknown,
  tokenizerConfig: Record<string, unknown>,
  specialTokensMap: Record<string, unknown>,
): AddedToken[] {
  const specialContents = collectSpecialTokenContents(tokenizerConfig, specialTokensMap);
  const byContent = new Map<string, AddedToken>();
  for (const token of parseAddedTokensDecoder(tokenizerConfig.added_tokens_decoder)) {
    byContent.set(token.content, token);
  }

  if (value !== undefined) {
    const raw = expectRecord(value, "added_tokens.json");
    for (const [content, id] of Object.entries(raw)) {
      if (typeof id !== "number" || !Number.isInteger(id) || id < 0) {
        throw new UnsupportedTokenizerError(
          `added_tokens.json["${content}"] must be a non-negative integer`,
        );
      }
      const decoded = byContent.get(content);
      byContent.set(content, {
        content,
        id,
        special: decoded?.special ?? specialContents.has(content),
      });
    }
  }
  return [...byContent.values()].sort((left, right) => left.id - right.id);
}

function isConfiguredQwen2Tokenizer(tokenizerConfig: Record<string, unknown>): boolean {
  const tokenizerClass = tokenizerConfig.tokenizer_class;
  return tokenizerClass === "Qwen2Tokenizer" || tokenizerClass === "Qwen2TokenizerFast";
}

/** Load a GPT/Qwen-style byte-level BPE tokenizer from `vocab.json` and `merges.txt`. */
export function loadByteLevelBPEFromVocabMerges(
  vocabJson: unknown,
  mergesText: string,
  options: ByteLevelBPEVocabMergesLoadOptions = {},
): BPETokenizer {
  const vocab = parseVocab(vocabJson);
  const tokenizerConfig = options.tokenizerConfig ?? {};
  const specialTokensMap = options.specialTokensMap ?? {};
  const addedTokens = configuredAddedTokens(options.addedTokens, tokenizerConfig, specialTokensMap);
  const eosCandidate = configuredSpecialTokenId(
    tokenizerConfig,
    specialTokensMap,
    vocab,
    addedTokens,
    "eos_token",
  );
  const config: BPEConfig = {
    vocab,
    merges: parseMergesText(mergesText),
    variant: "bytelevel",
    addedTokens,
    eosTokenIds: eosCandidate === undefined ? [] : [eosCandidate],
    addBosToken: tokenizerConfig.add_bos_token === true,
    addEosToken: tokenizerConfig.add_eos_token === true,
    addPrefixSpace: tokenizerConfig.add_prefix_space === true,
    useRegex: tokenizerConfig.use_regex !== false,
    ...(isConfiguredQwen2Tokenizer(tokenizerConfig)
      ? { splitPattern: QWEN2_PRETOKENIZE_REGEX }
      : {}),
    byteFallback: tokenizerConfig.byte_fallback === true,
  };
  const bosTokenId = configuredSpecialTokenId(
    tokenizerConfig,
    specialTokensMap,
    vocab,
    addedTokens,
    "bos_token",
  );
  const padTokenId = configuredSpecialTokenId(
    tokenizerConfig,
    specialTokensMap,
    vocab,
    addedTokens,
    "pad_token",
  );
  const unkTokenId = configuredSpecialTokenId(
    tokenizerConfig,
    specialTokensMap,
    vocab,
    addedTokens,
    "unk_token",
  );
  if (bosTokenId !== undefined) {
    config.bosTokenId = bosTokenId;
  }
  if (padTokenId !== undefined) {
    config.padTokenId = padTokenId;
  }
  if (unkTokenId !== undefined) {
    config.unkTokenId = unkTokenId;
  }
  return new BPETokenizer(config);
}
