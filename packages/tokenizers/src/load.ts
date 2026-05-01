/**
 * Tokenizer file loading and format detection.
 * @module
 */

import { existsSync, readFileSync, statSync } from "fs";
import { basename, dirname, join } from "path";
import {
  type BPETokenizer,
  loadBPEFromTokenizerJson,
  loadByteLevelBPEFromVocabMerges,
} from "./bpe/bpe";
import { type CLIPTokenizer, loadCLIPTokenizer } from "./bpe/clip";
import { CharTokenizer } from "./char";
import { UnsupportedTokenizerError } from "./errors";
import { SentencePieceTokenizer } from "./sentencepiece";
import { loadTekkenJson } from "./tekken";
import type { Tokenizer } from "./tokenizer";

export type TokenizerFormat =
  | "auto"
  | "tokenizer-json"
  | "bytelevel-vocab-merges"
  | "clip-vocab-merges"
  | "tekken-json"
  | "sentencepiece-model"
  | "char";

export type LoadTokenizerOptions = {
  format?: TokenizerFormat;
};

export type TokenizerFileSet = {
  directory?: string;
  tokenizerJsonPath?: string;
  tekkenJsonPath?: string;
  tokenizerModelPath?: string;
  tokenizerConfigPath?: string;
  specialTokensMapPath?: string;
  addedTokensPath?: string;
  tokenizerConfigData?: Record<string, unknown>;
  specialTokensMapData?: Record<string, unknown>;
  addedTokensData?: Record<string, unknown>;
  vocabJsonPath?: string;
  mergesTextPath?: string;
  vocabTextPath?: string;
};

function expectRecord(value: unknown, context: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new UnsupportedTokenizerError(`${context} must be an object`);
  }
  return Object.fromEntries(Object.entries(value));
}

function readJsonFile(path: string | undefined, context: string): Record<string, unknown> {
  if (path === undefined || !existsSync(path)) {
    return {};
  }
  const text = readFileSync(path, "utf8");
  const parsed: unknown = JSON.parse(text);
  return expectRecord(parsed, context);
}

function readMergedJsonFile(
  path: string | undefined,
  inlineData: Record<string, unknown> | undefined,
  context: string,
): Record<string, unknown> {
  return {
    ...readJsonFile(path, context),
    ...(inlineData ?? {}),
  };
}

function isDirectory(path: string): boolean {
  return existsSync(path) && statSync(path).isDirectory();
}

function resolveFileSet(source: string | TokenizerFileSet): TokenizerFileSet {
  if (typeof source !== "string") {
    return source;
  }

  if (isDirectory(source)) {
    const tokenizerModelPath = join(source, "tokenizer.model");
    const spieceModelPath = join(source, "spiece.model");
    return {
      directory: source,
      tokenizerJsonPath: join(source, "tokenizer.json"),
      tekkenJsonPath: join(source, "tekken.json"),
      tokenizerModelPath: existsSync(tokenizerModelPath) ? tokenizerModelPath : spieceModelPath,
      tokenizerConfigPath: join(source, "tokenizer_config.json"),
      specialTokensMapPath: join(source, "special_tokens_map.json"),
      addedTokensPath: join(source, "added_tokens.json"),
      vocabJsonPath: join(source, "vocab.json"),
      mergesTextPath: join(source, "merges.txt"),
    };
  }

  const name = basename(source);
  if (name === "vocab.json") {
    return {
      vocabJsonPath: source,
      mergesTextPath: join(dirname(source), "merges.txt"),
      tokenizerConfigPath: join(dirname(source), "tokenizer_config.json"),
      specialTokensMapPath: join(dirname(source), "special_tokens_map.json"),
      addedTokensPath: join(dirname(source), "added_tokens.json"),
    };
  }
  if (name === "merges.txt") {
    return {
      mergesTextPath: source,
      vocabJsonPath: join(dirname(source), "vocab.json"),
      tokenizerConfigPath: join(dirname(source), "tokenizer_config.json"),
      specialTokensMapPath: join(dirname(source), "special_tokens_map.json"),
      addedTokensPath: join(dirname(source), "added_tokens.json"),
    };
  }

  if (source.endsWith(".model")) {
    return { tokenizerModelPath: source };
  }
  if (source.endsWith(".json")) {
    return source.endsWith("tekken.json")
      ? { tekkenJsonPath: source }
      : { tokenizerJsonPath: source };
  }
  if (source.endsWith(".txt")) {
    return { vocabTextPath: source };
  }

  return {
    directory: source,
    tokenizerJsonPath: join(source, "tokenizer.json"),
    tekkenJsonPath: join(source, "tekken.json"),
    tokenizerModelPath: join(source, "tokenizer.model"),
    tokenizerConfigPath: join(source, "tokenizer_config.json"),
    specialTokensMapPath: join(source, "special_tokens_map.json"),
    addedTokensPath: join(source, "added_tokens.json"),
    vocabJsonPath: join(source, "vocab.json"),
    mergesTextPath: join(source, "merges.txt"),
  };
}

function existingPath(path: string | undefined): string | undefined {
  return path !== undefined && existsSync(path) ? path : undefined;
}

function hasCLIPTokenizerFiles(fileSet: TokenizerFileSet): boolean {
  return (
    existingPath(fileSet.vocabJsonPath) !== undefined &&
    existingPath(fileSet.mergesTextPath) !== undefined
  );
}

function isConfiguredCLIPTokenizer(fileSet: TokenizerFileSet): boolean {
  const tokenizerConfig = readMergedJsonFile(
    fileSet.tokenizerConfigPath,
    fileSet.tokenizerConfigData,
    "tokenizer_config.json",
  );
  const tokenizerClass = tokenizerConfig.tokenizer_class;
  return tokenizerClass === "CLIPTokenizer" || tokenizerClass === "CLIPTokenizerFast";
}

function isConfiguredByteLevelBPEVocabMergesTokenizer(fileSet: TokenizerFileSet): boolean {
  const tokenizerConfig = readMergedJsonFile(
    fileSet.tokenizerConfigPath,
    fileSet.tokenizerConfigData,
    "tokenizer_config.json",
  );
  const tokenizerClass = tokenizerConfig.tokenizer_class;
  return tokenizerClass === "Qwen2Tokenizer" || tokenizerClass === "Qwen2TokenizerFast";
}

function loadCharTokenizer(fileSet: TokenizerFileSet): CharTokenizer {
  const vocabPath = existingPath(fileSet.vocabTextPath) ?? existingPath(fileSet.tokenizerJsonPath);
  if (vocabPath === undefined) {
    throw new Error("loadTokenizer: char format requires a local text or vocab file");
  }
  return CharTokenizer.fromText(readFileSync(vocabPath, "utf8"));
}

function loadConfiguredVocabMergesTokenizer(fileSet: TokenizerFileSet): Tokenizer | null {
  if (!hasCLIPTokenizerFiles(fileSet)) {
    return null;
  }
  if (isConfiguredCLIPTokenizer(fileSet)) {
    return loadCLIP(fileSet);
  }
  if (isConfiguredByteLevelBPEVocabMergesTokenizer(fileSet)) {
    return loadByteLevelBPEVocabMerges(fileSet);
  }
  return null;
}

function loadAutoTokenizer(fileSet: TokenizerFileSet): Tokenizer {
  const configuredVocabMerges = loadConfiguredVocabMergesTokenizer(fileSet);
  if (configuredVocabMerges !== null) {
    return configuredVocabMerges;
  }

  const tokenizerJsonPath = existingPath(fileSet.tokenizerJsonPath);
  if (tokenizerJsonPath !== undefined) {
    try {
      return loadTokenizerJson({ ...fileSet, tokenizerJsonPath });
    } catch (error) {
      if (!(error instanceof UnsupportedTokenizerError)) {
        throw error;
      }
      if (
        existingPath(fileSet.tekkenJsonPath) === undefined &&
        existingPath(fileSet.tokenizerModelPath) === undefined &&
        !hasCLIPTokenizerFiles(fileSet)
      ) {
        throw error;
      }
    }
  }

  if (hasCLIPTokenizerFiles(fileSet)) {
    return loadCLIP(fileSet);
  }

  if (existingPath(fileSet.tekkenJsonPath) !== undefined) {
    return loadTekken(fileSet);
  }

  if (existingPath(fileSet.tokenizerModelPath) !== undefined) {
    return loadSentencePiece(fileSet);
  }

  throw new Error(
    "loadTokenizer: could not find a supported tokenizer.json, vocab.json + merges.txt, tekken.json, or tokenizer.model",
  );
}

/** Load a tokenizer from a local directory or explicit tokenizer files. */
export function loadTokenizer(
  source: string | TokenizerFileSet,
  options: LoadTokenizerOptions = {},
): Tokenizer {
  const fileSet = resolveFileSet(source);
  switch (options.format ?? "auto") {
    case "char":
      return loadCharTokenizer(fileSet);
    case "tokenizer-json":
      return loadTokenizerJson(fileSet);
    case "bytelevel-vocab-merges":
      return loadByteLevelBPEVocabMerges(fileSet);
    case "clip-vocab-merges":
      return loadCLIP(fileSet);
    case "tekken-json":
      return loadTekken(fileSet);
    case "sentencepiece-model":
      return loadSentencePiece(fileSet);
    case "auto":
      return loadAutoTokenizer(fileSet);
  }
}

/** Load a generic byte-level BPE tokenizer from `vocab.json` and `merges.txt`. */
export function loadByteLevelBPEVocabMerges(source: string | TokenizerFileSet): BPETokenizer {
  const fileSet = resolveFileSet(source);
  const vocabJsonPath = fileSet.vocabJsonPath;
  const mergesTextPath = fileSet.mergesTextPath;
  if (
    vocabJsonPath === undefined ||
    mergesTextPath === undefined ||
    !existsSync(vocabJsonPath) ||
    !existsSync(mergesTextPath)
  ) {
    throw new Error("loadByteLevelBPEVocabMerges: vocab.json and merges.txt were not found");
  }

  const tokenizerConfig = readMergedJsonFile(
    fileSet.tokenizerConfigPath,
    fileSet.tokenizerConfigData,
    "tokenizer_config.json",
  );
  const specialTokensMap = readMergedJsonFile(
    fileSet.specialTokensMapPath,
    fileSet.specialTokensMapData,
    "special_tokens_map.json",
  );
  const addedTokens = readMergedJsonFile(
    fileSet.addedTokensPath,
    fileSet.addedTokensData,
    "added_tokens.json",
  );
  return loadByteLevelBPEFromVocabMerges(
    readJsonFile(vocabJsonPath, "vocab.json"),
    readFileSync(mergesTextPath, "utf8"),
    {
      tokenizerConfig,
      specialTokensMap,
      addedTokens,
    },
  );
}

/** Load a CLIP vocab/merges tokenizer. */
export function loadCLIP(source: string | TokenizerFileSet): CLIPTokenizer {
  const fileSet = resolveFileSet(source);
  const vocabJsonPath = fileSet.vocabJsonPath;
  const mergesTextPath = fileSet.mergesTextPath;
  if (
    vocabJsonPath === undefined ||
    mergesTextPath === undefined ||
    !existsSync(vocabJsonPath) ||
    !existsSync(mergesTextPath)
  ) {
    throw new Error("loadCLIP: vocab.json and merges.txt were not found");
  }

  const tokenizerConfig = readMergedJsonFile(
    fileSet.tokenizerConfigPath,
    fileSet.tokenizerConfigData,
    "tokenizer_config.json",
  );
  const specialTokensMap = readMergedJsonFile(
    fileSet.specialTokensMapPath,
    fileSet.specialTokensMapData,
    "special_tokens_map.json",
  );
  return loadCLIPTokenizer(
    readJsonFile(vocabJsonPath, "vocab.json"),
    readFileSync(mergesTextPath, "utf8"),
    {
      tokenizerConfig,
      specialTokensMap,
    },
  );
}

/** Load a supported tokenizer.json tokenizer. */
export function loadTokenizerJson(source: string | TokenizerFileSet): BPETokenizer {
  const fileSet = resolveFileSet(source);
  const tokenizerJsonPath = fileSet.tokenizerJsonPath;
  if (tokenizerJsonPath === undefined || !existsSync(tokenizerJsonPath)) {
    throw new Error("loadTokenizerJson: tokenizer.json was not found");
  }

  const tokenizerJson = readJsonFile(tokenizerJsonPath, "tokenizer.json");
  const tokenizerConfig = readMergedJsonFile(
    fileSet.tokenizerConfigPath,
    fileSet.tokenizerConfigData,
    "tokenizer_config.json",
  );
  const specialTokensMap = readMergedJsonFile(
    fileSet.specialTokensMapPath,
    fileSet.specialTokensMapData,
    "special_tokens_map.json",
  );
  return loadBPEFromTokenizerJson(tokenizerJson, tokenizerConfig, specialTokensMap);
}

/** Load a Tekken tokenizer JSON file. */
export function loadTekken(source: string | TokenizerFileSet): BPETokenizer {
  const fileSet = resolveFileSet(source);
  const tekkenJsonPath = fileSet.tekkenJsonPath;
  if (tekkenJsonPath === undefined || !existsSync(tekkenJsonPath)) {
    throw new Error("loadTekken: tekken.json was not found");
  }

  const tekkenJson = readJsonFile(tekkenJsonPath, "tekken.json");
  const tokenizerConfig = readMergedJsonFile(
    fileSet.tokenizerConfigPath,
    fileSet.tokenizerConfigData,
    "tokenizer_config.json",
  );
  const specialTokensMap = readMergedJsonFile(
    fileSet.specialTokensMapPath,
    fileSet.specialTokensMapData,
    "special_tokens_map.json",
  );
  return loadTekkenJson(tekkenJson, tokenizerConfig, specialTokensMap);
}

/** Load a SentencePiece tokenizer model. */
export function loadSentencePiece(source: string | TokenizerFileSet): SentencePieceTokenizer {
  const fileSet = resolveFileSet(source);
  const tokenizerModelPath = fileSet.tokenizerModelPath;
  if (tokenizerModelPath === undefined || !existsSync(tokenizerModelPath)) {
    throw new Error("loadSentencePiece: tokenizer.model was not found");
  }

  return SentencePieceTokenizer.fromModelBytes(new Uint8Array(readFileSync(tokenizerModelPath)));
}
