import type { BenchmarkPrompt } from "./benchmark-serve-completions";
import type { ProtocolMode } from "./benchmark-serve-options";

function createPromptTokenIds(length: number, vocabSize: number): number[] {
  const tokenIds: number[] = [];
  let state = 0x12345678;
  const usableVocab = Math.max(2, vocabSize - 1);

  for (let index = 0; index < length; index += 1) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    tokenIds.push((state % usableVocab) + 1);
  }

  return tokenIds;
}

function createPromptText(length: number, tokenizer: { encode(text: string): number[] }): string {
  const seed =
    "This is a deterministic serving benchmark prompt about Apple Silicon ML throughput. ";
  let text = seed;
  while (tokenizer.encode(text).length < length) {
    text += seed;
  }
  return text;
}

export function createBenchmarkPrompt(
  length: number,
  vocabSize: number,
  tokenizer: { encode(text: string): number[] },
  protocolMode: ProtocolMode,
): BenchmarkPrompt {
  return {
    tokenIds: createPromptTokenIds(length, vocabSize),
    text: protocolMode === "completions" ? "" : createPromptText(length, tokenizer),
  };
}
