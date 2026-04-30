/**
 * T5-family tokenizer helpers.
 * @module
 */

import type { SentencePieceTokenizer } from "./sentencepiece";

export type T5TextInput = {
  inputIds: number[];
  attentionMask: number[];
  truncated: boolean;
};

export type EncodeT5TextInputOptions = {
  maxLength?: number;
};

/** Fixed-length T5 prompt encoding used by FLUX text-conditioning paths. */
export function encodeT5TextInput(
  tokenizer: SentencePieceTokenizer,
  text: string,
  options: EncodeT5TextInputOptions = {},
): T5TextInput {
  const maxLength = options.maxLength ?? 512;
  if (!Number.isInteger(maxLength) || maxLength < 1) {
    throw new Error("encodeT5TextInput: maxLength must be a positive integer");
  }

  const eosTokenId = tokenizer.eosTokenIds[0];
  const padTokenId = tokenizer.padTokenId;
  if (eosTokenId === undefined || padTokenId === undefined) {
    throw new Error("encodeT5TextInput: tokenizer must define EOS and PAD token IDs");
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
