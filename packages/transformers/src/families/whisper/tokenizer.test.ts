import { describe, expect, test } from "bun:test";
import type {
  BatchEncoding,
  DecodeOptions,
  EncodeOptions,
  Encoding,
  Tokenizer,
} from "@mlxts/tokenizers";
import {
  createWhisperDecoderPromptTokenIds,
  decodeWhisperGeneratedTokenIds,
  resolveWhisperSpecialTokens,
} from "./tokenizer";
import type { WhisperConfig } from "./types";

class FakeTokenizer implements Tokenizer {
  readonly vocabSize = 128;
  readonly bosTokenId = undefined;
  readonly eosTokenIds = [5];
  readonly padTokenId = undefined;

  encode(text: string, _options: EncodeOptions = {}): number[] {
    if (text === "<|startoftranscript|>") {
      return [1];
    }
    return [10];
  }

  encodeWithOffsets(text: string, options: EncodeOptions = {}): Encoding {
    return { ids: this.encode(text, options) };
  }

  encodeBatch(texts: string[], options: EncodeOptions = {}): BatchEncoding {
    return texts.map((text) => this.encodeWithOffsets(text, options));
  }

  decode(tokenIds: number[], options: DecodeOptions = {}): string {
    const specialIds = new Set([1, 2, 3, 4, 5, 6, 7, 8, 100]);
    return tokenIds
      .filter((tokenId) => !(options.skipSpecialTokens === true && specialIds.has(tokenId)))
      .map((tokenId) => `token${tokenId}`)
      .join(" ");
  }

  decodeBatch(batch: number[][], options: DecodeOptions = {}): string[] {
    return batch.map((tokens) => this.decode(tokens, options));
  }
}

function whisperConfig(): WhisperConfig {
  return {
    modelType: "whisper",
    rawConfig: {},
    vocabSize: 128,
    numMelBins: 4,
    encoderLayers: 1,
    encoderAttentionHeads: 1,
    decoderLayers: 1,
    decoderAttentionHeads: 1,
    encoderFfnDim: 8,
    decoderFfnDim: 8,
    dModel: 4,
    encoderHeadDim: 4,
    decoderHeadDim: 4,
    activationFunction: "gelu",
    maxSourcePositions: 2,
    maxTargetPositions: 8,
    padTokenId: 5,
    bosTokenId: 5,
    eosTokenId: 5,
    decoderStartTokenId: 1,
    scaleEmbedding: false,
    useCache: true,
  };
}

function tokenizerConfig(): Record<string, unknown> {
  return {
    added_tokens_decoder: {
      "1": { content: "<|startoftranscript|>", special: true },
      "2": { content: "<|transcribe|>", special: true },
      "3": { content: "<|notimestamps|>", special: true },
      "4": { content: "<|en|>", special: true },
      "5": { content: "<|endoftext|>", special: true },
      "100": { content: "<|0.00|>", special: true },
      "7": { content: "<|translate|>", special: true },
      "8": { content: "<|nospeech|>", special: true },
    },
  };
}

describe("Whisper tokenizer helpers", () => {
  test("resolve special tokens from tokenizer metadata", () => {
    const specialTokens = resolveWhisperSpecialTokens(tokenizerConfig(), {
      tokenizer: new FakeTokenizer(),
      config: whisperConfig(),
    });

    expect(specialTokens.startOfTranscript).toBe(1);
    expect(specialTokens.endOfTextTokenIds).toEqual([5]);
    expect(specialTokens.languageTokenIds.get("en")).toBe(4);
    expect(specialTokens.noTimestampsTokenId).toBe(3);
    expect(specialTokens.noSpeechTokenId).toBe(8);
  });

  test("builds Whisper decoder prompt ids", () => {
    const specialTokens = resolveWhisperSpecialTokens(tokenizerConfig(), {
      tokenizer: new FakeTokenizer(),
      config: whisperConfig(),
    });

    expect(createWhisperDecoderPromptTokenIds(specialTokens)).toEqual([1, 4, 2, 3]);
    expect(
      createWhisperDecoderPromptTokenIds(specialTokens, {
        task: "translate",
        language: null,
        withoutTimestamps: false,
      }),
    ).toEqual([1, 7]);
  });

  test("decodes generated text before EOT and timestamp tokens", () => {
    const specialTokens = resolveWhisperSpecialTokens(tokenizerConfig(), {
      tokenizer: new FakeTokenizer(),
      config: whisperConfig(),
    });

    const text = decodeWhisperGeneratedTokenIds(
      new FakeTokenizer(),
      [10, 100, 11, 5, 12],
      specialTokens,
    );

    expect(text).toBe("token10 token11");
  });
});
