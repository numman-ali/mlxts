/**
 * tokenizer.json parsing for the supported BPE subset.
 * @module
 */

import { type AddedToken, type BPEConfig, BPETokenizer, type BPEVariant } from "./bpe-base";
import { UnsupportedTokenizerError } from "./errors";

type VariantSettings = {
  variant: BPEVariant;
  addPrefixSpace: boolean;
  useRegex: boolean;
};

type ByteLevelSettings = {
  addPrefixSpace: boolean;
  useRegex: boolean;
};

function isStringPattern(
  value: unknown,
  expected: string,
  context: string,
): value is Record<"String", string> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new UnsupportedTokenizerError(`${context} must be an object`);
  }
  const record = Object.fromEntries(Object.entries(value));
  return record.String === expected;
}

function expectRecord(value: unknown, context: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new UnsupportedTokenizerError(`${context} must be an object`);
  }
  return Object.fromEntries(Object.entries(value));
}

function expectArray(value: unknown, context: string): unknown[] {
  if (!Array.isArray(value)) {
    throw new UnsupportedTokenizerError(`${context} must be an array`);
  }
  return value;
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

function parseMerges(value: unknown): Array<[string, string]> {
  const merges = expectArray(value, "tokenizer.model.merges");
  const parsed: Array<[string, string]> = [];
  for (const entry of merges) {
    if (typeof entry === "string") {
      const [left, right] = entry.split(" ");
      if (left === undefined || right === undefined) {
        throw new UnsupportedTokenizerError(`legacy merge entry "${entry}" is malformed`);
      }
      parsed.push([left, right]);
      continue;
    }

    if (Array.isArray(entry) && entry.length === 2) {
      const [left, right] = entry;
      if (typeof left !== "string" || typeof right !== "string") {
        throw new UnsupportedTokenizerError("merge entries must contain two strings");
      }
      parsed.push([left, right]);
      continue;
    }

    throw new UnsupportedTokenizerError("tokenizer.model.merges must contain string pairs");
  }
  return parsed;
}

function parseAddedTokens(value: unknown): AddedToken[] {
  const entries = expectArray(value, "tokenizer.added_tokens");
  const tokens: AddedToken[] = [];
  for (const entry of entries) {
    const raw = expectRecord(entry, "tokenizer.added_tokens[]");
    const content = expectString(raw.content, "tokenizer.added_tokens[].content");
    const id = raw.id;
    if (typeof id !== "number" || !Number.isInteger(id) || id < 0) {
      throw new UnsupportedTokenizerError(
        `tokenizer.added_tokens["${content}"].id must be a non-negative integer`,
      );
    }
    tokens.push({
      content,
      id,
      special: raw.special === true,
    });
  }
  return tokens;
}

function parseSpecialTokenId(
  tokenValue: unknown,
  vocab: Record<string, number>,
  addedTokens: AddedToken[],
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

function byteLevelSettings(value: Record<string, unknown>): ByteLevelSettings {
  return {
    addPrefixSpace: value.add_prefix_space === true,
    useRegex: value.use_regex !== false,
  };
}

function sequenceByteLevelSettings(value: Record<string, unknown>): ByteLevelSettings | null {
  const pretokenizers = value.pretokenizers;
  if (!Array.isArray(pretokenizers)) {
    throw new UnsupportedTokenizerError(
      "tokenizer.json.pre_tokenizer.pretokenizers must be an array for Sequence pre-tokenizers",
    );
  }

  let byteLevel: ByteLevelSettings | null = null;
  let sawSplitRegex = false;

  for (const entry of pretokenizers) {
    const pretokenizer = expectRecord(entry, "tokenizer.json.pre_tokenizer.pretokenizers[]");
    if (pretokenizer.type === "ByteLevel") {
      byteLevel = byteLevelSettings(pretokenizer);
      continue;
    }
    if (pretokenizer.type !== "Split") {
      continue;
    }

    const pattern = pretokenizer.pattern;
    if (typeof pattern !== "object" || pattern === null || Array.isArray(pattern)) {
      continue;
    }
    const rawPattern = Object.fromEntries(Object.entries(pattern));
    if (typeof rawPattern.Regex === "string" && rawPattern.Regex !== "") {
      sawSplitRegex = true;
    }
  }

  if (byteLevel === null) {
    return null;
  }
  return {
    addPrefixSpace: byteLevel.addPrefixSpace,
    useRegex: byteLevel.useRegex || sawSplitRegex,
  };
}

function resolveVariant(preTokenizer: unknown, decoder: unknown): VariantSettings {
  if (preTokenizer === undefined || preTokenizer === null) {
    const decoderRecord =
      decoder === undefined ? undefined : expectRecord(decoder, "tokenizer.json.decoder");
    if (decoderRecord?.type !== "Sequence") {
      throw new UnsupportedTokenizerError(
        "BPE tokenizer without a pre_tokenizer is only supported for sentencepiece-style decoder sequences",
      );
    }
    return { variant: "sentencepiece", addPrefixSpace: false, useRegex: true };
  }

  const pre = expectRecord(preTokenizer, "tokenizer.json.pre_tokenizer");
  if (pre.type === "ByteLevel") {
    const settings = byteLevelSettings(pre);
    return {
      variant: "bytelevel",
      addPrefixSpace: settings.addPrefixSpace,
      useRegex: settings.useRegex,
    };
  }

  if (pre.type === "Sequence") {
    const settings = sequenceByteLevelSettings(pre);
    if (settings !== null) {
      return {
        variant: "bytelevel",
        addPrefixSpace: settings.addPrefixSpace,
        useRegex: settings.useRegex,
      };
    }
  }

  if (
    pre.type === "Split" &&
    pre.behavior === "MergedWithPrevious" &&
    pre.invert === false &&
    isStringPattern(pre.pattern, " ", "tokenizer.json.pre_tokenizer.pattern")
  ) {
    const decoderRecord = expectRecord(decoder, "tokenizer.json.decoder");
    if (decoderRecord.type !== "Sequence") {
      throw new UnsupportedTokenizerError(
        'tokenizer.json.decoder.type must be "Sequence" for sentencepiece-style Split tokenizers',
      );
    }

    return {
      variant: "sentencepiece",
      addPrefixSpace: false,
      useRegex: true,
    };
  }

  throw new UnsupportedTokenizerError(
    `tokenizer.json.pre_tokenizer.type "${String(pre.type)}" is not supported for BPE tokenizers`,
  );
}

function configuredSpecialTokenId(
  tokenizerConfig: Record<string, unknown>,
  specialTokensMap: Record<string, unknown>,
  vocab: Record<string, number>,
  addedTokens: AddedToken[],
  key: "bos_token" | "eos_token" | "pad_token" | "unk_token",
): number | undefined {
  return (
    parseSpecialTokenId(tokenizerConfig[key], vocab, addedTokens) ??
    parseSpecialTokenId(tokenizerConfig[`${key}_id`], vocab, addedTokens) ??
    parseSpecialTokenId(specialTokensMap[key], vocab, addedTokens)
  );
}

/** Load the supported tokenizer.json BPE subset into a `BPETokenizer`. */
export function loadBPEFromTokenizerJson(
  tokenizerJson: unknown,
  tokenizerConfig: Record<string, unknown> = {},
  specialTokensMap: Record<string, unknown> = {},
): BPETokenizer {
  const root = expectRecord(tokenizerJson, "tokenizer.json");
  const model = expectRecord(root.model, "tokenizer.json.model");
  const modelType = expectString(model.type, "tokenizer.json.model.type");
  if (modelType !== "BPE") {
    throw new UnsupportedTokenizerError(
      `tokenizer.json.model.type "${modelType}" is not supported`,
    );
  }

  const vocab = parseVocab(model.vocab);
  const merges = parseMerges(model.merges);
  const addedTokens = root.added_tokens === undefined ? [] : parseAddedTokens(root.added_tokens);
  const variant = resolveVariant(root.pre_tokenizer, root.decoder);
  const bosTokenId = configuredSpecialTokenId(
    tokenizerConfig,
    specialTokensMap,
    vocab,
    addedTokens,
    "bos_token",
  );
  const eosCandidate = configuredSpecialTokenId(
    tokenizerConfig,
    specialTokensMap,
    vocab,
    addedTokens,
    "eos_token",
  );
  const padTokenId = configuredSpecialTokenId(
    tokenizerConfig,
    specialTokensMap,
    vocab,
    addedTokens,
    "pad_token",
  );
  const unkTokenId =
    parseSpecialTokenId(model.unk_token, vocab, addedTokens) ??
    configuredSpecialTokenId(tokenizerConfig, specialTokensMap, vocab, addedTokens, "unk_token");

  const config: BPEConfig = {
    vocab,
    merges,
    variant: variant.variant,
    addedTokens,
    eosTokenIds: eosCandidate === undefined ? [] : [eosCandidate],
    addBosToken: tokenizerConfig.add_bos_token === true,
    addEosToken: tokenizerConfig.add_eos_token === true,
    addPrefixSpace: variant.addPrefixSpace,
    useRegex: variant.useRegex,
    byteFallback: model.byte_fallback === true,
  };
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
