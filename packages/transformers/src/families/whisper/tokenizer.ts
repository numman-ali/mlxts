/**
 * Whisper tokenizer prompt helpers.
 * @module
 */

import type { Tokenizer } from "@mlxts/tokenizers";

import type { WhisperConfig } from "./types";

export type WhisperTask = "transcribe" | "translate";

export type WhisperSpecialTokens = {
  startOfTranscript: number;
  endOfTextTokenIds: readonly number[];
  transcribeTokenId?: number;
  translateTokenId?: number;
  noTimestampsTokenId?: number;
  timestampBeginTokenId?: number;
  noSpeechTokenId?: number;
  startOfPreviousTokenId?: number;
  startOfLmTokenId?: number;
  languageTokenIds: ReadonlyMap<string, number>;
};

export type WhisperPromptOptions = {
  task?: WhisperTask;
  language?: string | null;
  withoutTimestamps?: boolean;
};

type WhisperSpecialTokenResolveOptions = {
  tokenizer?: Tokenizer;
  config?: WhisperConfig;
};

function expectRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? Object.fromEntries(Object.entries(value))
    : null;
}

function addedTokenContent(value: unknown): string | null {
  if (typeof value === "string") {
    return value;
  }
  const record = expectRecord(value);
  if (record === null) {
    return null;
  }
  return typeof record.content === "string" ? record.content : null;
}

function parseConfiguredTokens(tokenizerConfig: unknown): Map<string, number> {
  const root = expectRecord(tokenizerConfig);
  const configured = new Map<string, number>();
  if (root === null) {
    return configured;
  }

  const addedTokensDecoder = expectRecord(root.added_tokens_decoder);
  if (addedTokensDecoder === null) {
    return configured;
  }

  for (const [idText, entry] of Object.entries(addedTokensDecoder)) {
    const id = Number(idText);
    const content = addedTokenContent(entry);
    if (content !== null && Number.isSafeInteger(id) && id >= 0) {
      configured.set(content, id);
    }
  }
  return configured;
}

function tokenizerToken(tokenizer: Tokenizer | undefined, content: string): number | null {
  if (tokenizer === undefined) {
    return null;
  }
  const ids = tokenizer.encode(content, { addSpecialTokens: false });
  return ids.length === 1 ? (ids[0] ?? null) : null;
}

function resolveTokenId(
  configuredTokens: ReadonlyMap<string, number>,
  tokenizer: Tokenizer | undefined,
  content: string,
): number | null {
  return configuredTokens.get(content) ?? tokenizerToken(tokenizer, content);
}

function requiredToken(
  configuredTokens: ReadonlyMap<string, number>,
  tokenizer: Tokenizer | undefined,
  content: string,
  fallback: number | null | undefined,
): number {
  const token = resolveTokenId(configuredTokens, tokenizer, content) ?? fallback;
  if (token === undefined || token === null) {
    throw new Error(`resolveWhisperSpecialTokens: missing ${content}.`);
  }
  return token;
}

function optionalToken(
  configuredTokens: ReadonlyMap<string, number>,
  tokenizer: Tokenizer | undefined,
  content: string,
): number | undefined {
  return resolveTokenId(configuredTokens, tokenizer, content) ?? undefined;
}

function eosTokenIds(
  config: WhisperConfig | undefined,
  tokenizer: Tokenizer | undefined,
): number[] {
  const ids: number[] = [];
  const configEos = config?.eosTokenId;
  if (typeof configEos === "number") {
    ids.push(configEos);
  } else if (Array.isArray(configEos)) {
    ids.push(...configEos);
  }
  if (tokenizer !== undefined) {
    ids.push(...tokenizer.eosTokenIds);
  }
  return [...new Set(ids)].sort((left, right) => left - right);
}

function languageTokenIds(configuredTokens: ReadonlyMap<string, number>): Map<string, number> {
  const languages = new Map<string, number>();
  for (const [content, id] of configuredTokens) {
    if (!content.startsWith("<|") || !content.endsWith("|>")) {
      continue;
    }
    const code = content.slice(2, -2);
    if (/^[a-z]{2,3}$/.test(code)) {
      languages.set(code, id);
    }
  }
  return languages;
}

/** Resolve Whisper special-token ids from tokenizer metadata and tokenizer behavior. */
export function resolveWhisperSpecialTokens(
  tokenizerConfig: unknown,
  options: WhisperSpecialTokenResolveOptions = {},
): WhisperSpecialTokens {
  const configuredTokens = parseConfiguredTokens(tokenizerConfig);
  const tokenizer = options.tokenizer;
  const config = options.config;
  const endOfTextTokenIds = eosTokenIds(config, tokenizer);
  if (endOfTextTokenIds.length === 0) {
    throw new Error("resolveWhisperSpecialTokens: missing end-of-text token.");
  }

  const specialTokens: WhisperSpecialTokens = {
    startOfTranscript: requiredToken(
      configuredTokens,
      tokenizer,
      "<|startoftranscript|>",
      config?.decoderStartTokenId,
    ),
    endOfTextTokenIds,
    languageTokenIds: languageTokenIds(configuredTokens),
  };
  const transcribeTokenId = optionalToken(configuredTokens, tokenizer, "<|transcribe|>");
  const translateTokenId = optionalToken(configuredTokens, tokenizer, "<|translate|>");
  const noTimestampsTokenId = optionalToken(configuredTokens, tokenizer, "<|notimestamps|>");
  const timestampBeginTokenId = optionalToken(configuredTokens, tokenizer, "<|0.00|>");
  const noSpeechTokenId = optionalToken(configuredTokens, tokenizer, "<|nospeech|>");
  const startOfPreviousTokenId = optionalToken(configuredTokens, tokenizer, "<|startofprev|>");
  const startOfLmTokenId = optionalToken(configuredTokens, tokenizer, "<|startoflm|>");

  if (transcribeTokenId !== undefined) {
    specialTokens.transcribeTokenId = transcribeTokenId;
  }
  if (translateTokenId !== undefined) {
    specialTokens.translateTokenId = translateTokenId;
  }
  if (noTimestampsTokenId !== undefined) {
    specialTokens.noTimestampsTokenId = noTimestampsTokenId;
  }
  if (timestampBeginTokenId !== undefined) {
    specialTokens.timestampBeginTokenId = timestampBeginTokenId;
  }
  if (noSpeechTokenId !== undefined) {
    specialTokens.noSpeechTokenId = noSpeechTokenId;
  }
  if (startOfPreviousTokenId !== undefined) {
    specialTokens.startOfPreviousTokenId = startOfPreviousTokenId;
  }
  if (startOfLmTokenId !== undefined) {
    specialTokens.startOfLmTokenId = startOfLmTokenId;
  }
  return specialTokens;
}

function defaultLanguage(specialTokens: WhisperSpecialTokens, language: string | null | undefined) {
  if (language !== undefined) {
    return language;
  }
  return specialTokens.languageTokenIds.has("en") ? "en" : null;
}

function taskTokenId(specialTokens: WhisperSpecialTokens, task: WhisperTask): number {
  const tokenId =
    task === "transcribe" ? specialTokens.transcribeTokenId : specialTokens.translateTokenId;
  if (tokenId === undefined) {
    throw new Error(`createWhisperDecoderPromptTokenIds: missing ${task} token.`);
  }
  return tokenId;
}

/** Build the decoder prompt prefix used by Whisper greedy transcription. */
export function createWhisperDecoderPromptTokenIds(
  specialTokens: WhisperSpecialTokens,
  options: WhisperPromptOptions = {},
): number[] {
  const task = options.task ?? "transcribe";
  const language = defaultLanguage(specialTokens, options.language);
  const prompt = [specialTokens.startOfTranscript];
  if (language !== null) {
    const languageTokenId = specialTokens.languageTokenIds.get(language);
    if (languageTokenId === undefined) {
      throw new Error(`createWhisperDecoderPromptTokenIds: unknown language "${language}".`);
    }
    prompt.push(languageTokenId);
  }
  prompt.push(taskTokenId(specialTokens, task));
  if (options.withoutTimestamps ?? true) {
    if (specialTokens.noTimestampsTokenId === undefined) {
      throw new Error("createWhisperDecoderPromptTokenIds: missing no-timestamps token.");
    }
    prompt.push(specialTokens.noTimestampsTokenId);
  }
  return prompt;
}

/** Decode generated Whisper token ids into text, dropping EOT and timestamp tokens. */
export function decodeWhisperGeneratedTokenIds(
  tokenizer: Tokenizer,
  tokenIds: readonly number[],
  specialTokens: WhisperSpecialTokens,
): string {
  const endOfText = new Set(specialTokens.endOfTextTokenIds);
  const textTokenIds: number[] = [];
  for (const tokenId of tokenIds) {
    if (endOfText.has(tokenId)) {
      break;
    }
    if (
      specialTokens.timestampBeginTokenId !== undefined &&
      tokenId >= specialTokens.timestampBeginTokenId
    ) {
      continue;
    }
    textTokenIds.push(tokenId);
  }
  return tokenizer.decode(textTokenIds, { skipSpecialTokens: true }).trim();
}
