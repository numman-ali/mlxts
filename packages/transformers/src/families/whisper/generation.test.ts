import { describe, expect, test } from "bun:test";
import { zeros } from "@mlxts/core";
import type {
  BatchEncoding,
  DecodeOptions,
  EncodeOptions,
  Encoding,
  Tokenizer,
} from "@mlxts/tokenizers";

import { generateWhisperGreedyTranscription } from "./generation";
import { WhisperForConditionalGeneration } from "./model";
import { resolveWhisperSpecialTokens } from "./tokenizer";
import type { WhisperConfig } from "./types";

class AnyIdTokenizer implements Tokenizer {
  readonly vocabSize = 32;
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

  decode(tokenIds: number[], _options: DecodeOptions = {}): string {
    return tokenIds.map((tokenId) => `token${tokenId}`).join(" ");
  }

  decodeBatch(batch: number[][], options: DecodeOptions = {}): string[] {
    return batch.map((tokens) => this.decode(tokens, options));
  }
}

function whisperConfig(): WhisperConfig {
  return {
    modelType: "whisper",
    rawConfig: {},
    vocabSize: 32,
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

const tokenizerConfig = {
  added_tokens_decoder: {
    "1": { content: "<|startoftranscript|>", special: true },
    "2": { content: "<|transcribe|>", special: true },
    "3": { content: "<|notimestamps|>", special: true },
    "5": { content: "<|endoftext|>", special: true },
  },
};

describe("generateWhisperGreedyTranscription", () => {
  test("runs a finite greedy decode over prepared features", () => {
    const config = whisperConfig();
    using model = new WhisperForConditionalGeneration(config);
    const tokenizer = new AnyIdTokenizer();
    const specialTokens = resolveWhisperSpecialTokens(tokenizerConfig, { tokenizer, config });
    using inputFeatures = zeros([1, config.maxSourcePositions * 2, config.numMelBins]);

    const result = generateWhisperGreedyTranscription(
      model,
      inputFeatures,
      tokenizer,
      specialTokens,
      {
        language: null,
        maxNewTokens: 1,
      },
    );

    expect(result.promptTokenIds).toEqual([1, 2, 3]);
    expect(result.generatedTokens).toBe(1);
    expect(result.tokenIds).toHaveLength(4);
    expect(["eos", "max_tokens"]).toContain(result.stoppedReason);
  });
});
