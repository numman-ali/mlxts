/**
 * Mistral Tekken tokenizer loading.
 * @module
 */

import { type AddedToken, BPETokenizer } from "./bpe-base";
import { encodeByteLevelBytes } from "./byte-level";
import { UnsupportedTokenizerError } from "./errors";

type TekkenSpecialToken = {
  rank: number;
  token: string;
};

type TekkenVocabToken = {
  rank: number;
  bytes: Uint8Array;
};

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

function expectInteger(value: unknown, context: string): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value < 0) {
    throw new UnsupportedTokenizerError(`${context} must be a non-negative integer`);
  }
  return value;
}

function decodeBase64Bytes(value: string, context: string): Uint8Array {
  try {
    const decoded = atob(value);
    const bytes = new Uint8Array(decoded.length);
    for (let index = 0; index < decoded.length; index += 1) {
      bytes[index] = decoded.charCodeAt(index);
    }
    return bytes;
  } catch (error) {
    throw new UnsupportedTokenizerError(
      `${context} must be a valid base64 string: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
}

function bytesKey(bytes: Uint8Array): string {
  return bytes.join(",");
}

function parseSpecialTokens(value: unknown): TekkenSpecialToken[] {
  const entries = expectArray(value, "tekken.special_tokens");
  const tokens = entries.map((entry, index) => {
    const record = expectRecord(entry, `tekken.special_tokens[${index}]`);
    return {
      rank: expectInteger(record.rank, `tekken.special_tokens[${index}].rank`),
      token: expectString(record.token_str, `tekken.special_tokens[${index}].token_str`),
    };
  });

  return tokens.sort((left, right) => left.rank - right.rank);
}

function parseVocabEntries(value: unknown): TekkenVocabToken[] {
  const entries = expectArray(value, "tekken.vocab");
  const tokens = entries.map((entry, index) => {
    const record = expectRecord(entry, `tekken.vocab[${index}]`);
    return {
      rank: expectInteger(record.rank, `tekken.vocab[${index}].rank`),
      bytes: decodeBase64Bytes(
        expectString(record.token_bytes, `tekken.vocab[${index}].token_bytes`),
        `tekken.vocab[${index}].token_bytes`,
      ),
    };
  });

  return tokens.sort((left, right) => left.rank - right.rank);
}

function parseSpecialTokenId(
  value: unknown,
  vocab: Record<string, number>,
  addedTokens: AddedToken[],
): number | undefined {
  if (typeof value === "number" && Number.isInteger(value) && value >= 0) {
    return value;
  }

  const addedByContent = new Map(addedTokens.map((token) => [token.content, token.id]));
  if (typeof value === "string") {
    return addedByContent.get(value) ?? vocab[value];
  }

  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return undefined;
  }

  const record = Object.fromEntries(Object.entries(value));
  return typeof record.content === "string"
    ? (addedByContent.get(record.content) ?? vocab[record.content])
    : undefined;
}

function resolveConfiguredSpecialTokenId(
  tokenizerConfig: Record<string, unknown>,
  specialTokensMap: Record<string, unknown>,
  vocab: Record<string, number>,
  addedTokens: AddedToken[],
  key: "bos_token" | "eos_token" | "pad_token" | "unk_token",
  fallbackTokens: readonly string[],
): number | undefined {
  return (
    parseSpecialTokenId(tokenizerConfig[key], vocab, addedTokens) ??
    parseSpecialTokenId(tokenizerConfig[`${key}_id`], vocab, addedTokens) ??
    parseSpecialTokenId(specialTokensMap[key], vocab, addedTokens) ??
    fallbackTokens.map((token) => vocab[token]).find((tokenId) => tokenId !== undefined)
  );
}

function specialTokenOffset(
  config: Record<string, unknown>,
  specialTokens: readonly TekkenSpecialToken[],
): number {
  const highestSpecialRank =
    specialTokens.length === 0 ? 0 : Math.max(...specialTokens.map((token) => token.rank + 1));
  const configuredCount = config.default_num_special_tokens;
  if (configuredCount === undefined) {
    return highestSpecialRank;
  }
  const parsedCount = expectInteger(configuredCount, "tekken.config.default_num_special_tokens");
  if (parsedCount < highestSpecialRank) {
    throw new UnsupportedTokenizerError(
      `tekken.config.default_num_special_tokens (${parsedCount}) must be >= the highest special-token rank (${highestSpecialRank - 1})`,
    );
  }
  return parsedCount;
}

function buildRegularVocabAndMerges(
  entries: readonly TekkenVocabToken[],
  offset: number,
): { vocab: Record<string, number>; merges: Array<[string, string]> } {
  const vocab: Record<string, number> = {};
  const tokenByKey = new Map<string, { rank: number; token: string }>();

  for (const entry of entries) {
    const token = encodeByteLevelBytes(entry.bytes);
    vocab[token] = offset + entry.rank;
    tokenByKey.set(bytesKey(entry.bytes), { rank: entry.rank, token });
  }

  const mergeCandidates: Array<{ left: string; right: string; rank: number }> = [];
  for (const entry of entries) {
    if (entry.bytes.length <= 1) {
      continue;
    }

    for (let splitIndex = 1; splitIndex < entry.bytes.length; splitIndex += 1) {
      const left = tokenByKey.get(bytesKey(entry.bytes.subarray(0, splitIndex)));
      const right = tokenByKey.get(bytesKey(entry.bytes.subarray(splitIndex)));
      if (left === undefined || right === undefined) {
        continue;
      }

      mergeCandidates.push({
        left: left.token,
        right: right.token,
        rank: entry.rank,
      });
    }
  }

  mergeCandidates.sort((left, right) => left.rank - right.rank);
  return {
    vocab,
    merges: mergeCandidates.map((entry) => [entry.left, entry.right]),
  };
}

/** Load a Tekken tokenizer JSON file into the shared BPE tokenizer surface. */
export function loadTekkenJson(
  tekkenJson: unknown,
  tokenizerConfig: Record<string, unknown> = {},
  specialTokensMap: Record<string, unknown> = {},
): BPETokenizer {
  const root = expectRecord(tekkenJson, "tekken");
  const config = expectRecord(root.config, "tekken.config");
  const pattern = expectString(config.pattern, "tekken.config.pattern");
  const specialTokens = parseSpecialTokens(root.special_tokens);
  const specialOffset = specialTokenOffset(config, specialTokens);
  const addedTokens: AddedToken[] = specialTokens.map((token) => ({
    content: token.token,
    id: token.rank,
    special: true,
  }));

  const vocab: Record<string, number> = {};
  for (const token of specialTokens) {
    vocab[token.token] = token.rank;
  }

  const regular = buildRegularVocabAndMerges(parseVocabEntries(root.vocab), specialOffset);
  Object.assign(vocab, regular.vocab);

  const bosTokenId = resolveConfiguredSpecialTokenId(
    tokenizerConfig,
    specialTokensMap,
    vocab,
    addedTokens,
    "bos_token",
    ["<s>", "<bos>"],
  );
  const eosTokenId = resolveConfiguredSpecialTokenId(
    tokenizerConfig,
    specialTokensMap,
    vocab,
    addedTokens,
    "eos_token",
    ["</s>", "<eos>"],
  );
  const padTokenId = resolveConfiguredSpecialTokenId(
    tokenizerConfig,
    specialTokensMap,
    vocab,
    addedTokens,
    "pad_token",
    ["<pad>"],
  );
  const unkTokenId = resolveConfiguredSpecialTokenId(
    tokenizerConfig,
    specialTokensMap,
    vocab,
    addedTokens,
    "unk_token",
    ["<unk>"],
  );

  const tokenizerOptions = {
    vocab,
    merges: regular.merges,
    variant: "bytelevel" as const,
    addedTokens,
    eosTokenIds: eosTokenId === undefined ? [] : [eosTokenId],
    addBosToken:
      tokenizerConfig.add_bos_token === true ||
      (tokenizerConfig.add_bos_token === undefined && bosTokenId !== undefined),
    addEosToken: tokenizerConfig.add_eos_token === true,
    addPrefixSpace: false,
    useRegex: true,
    splitPattern: pattern,
    byteFallback: false,
  };

  return new BPETokenizer({
    ...tokenizerOptions,
    ...(bosTokenId === undefined ? {} : { bosTokenId }),
    ...(padTokenId === undefined ? {} : { padTokenId }),
    ...(unkTokenId === undefined ? {} : { unkTokenId }),
  });
}
