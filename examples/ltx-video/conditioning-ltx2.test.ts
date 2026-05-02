import { describe, expect, test } from "bun:test";
import { MxArray, mxEval } from "@mlxts/core";
import type { Ltx2TextConnectorsConfig } from "@mlxts/diffusion";
import type {
  BatchEncoding,
  DecodeOptions,
  EncodeOptions,
  Encoding,
  Tokenizer,
} from "@mlxts/tokenizers";
import type { Gemma3TextModelOptions, Gemma3TextModelOutput } from "@mlxts/transformers";

import { createLtx2PromptConditioner } from "./conditioning-ltx2";

function tinyConnectorConfig(): Ltx2TextConnectorsConfig {
  return {
    captionChannels: 2,
    textProjInFactor: 3,
    textEncoderDim: 6,
    videoConnectorNumAttentionHeads: 1,
    videoConnectorAttentionHeadDim: 2,
    videoConnectorHiddenSize: 2,
    videoConnectorNumLayers: 1,
    videoConnectorNumLearnableRegisters: null,
    videoGatedAttn: false,
    audioConnectorNumAttentionHeads: 1,
    audioConnectorAttentionHeadDim: 2,
    audioConnectorHiddenSize: 2,
    audioConnectorNumLayers: 1,
    audioConnectorNumLearnableRegisters: null,
    audioGatedAttn: false,
    connectorRopeBaseSeqLen: 4,
    ropeTheta: 4,
    ropeDoublePrecision: true,
    causalTemporalPositioning: false,
    ropeType: "interleaved",
    perModalityProjections: false,
    videoHiddenDim: 2,
    audioHiddenDim: 2,
    projBias: false,
    rawConfig: {},
  };
}

class FakeTokenizer implements Tokenizer {
  readonly vocabSize = 128;
  readonly bosTokenId = 1;
  readonly eosTokenIds = [2];
  readonly padTokenId = undefined;

  encode(text: string, _options?: EncodeOptions): number[] {
    if (text === "long") {
      return [10, 11, 12, 13, 14];
    }
    if (text === "") {
      return [2];
    }
    return [3, 4];
  }

  encodeWithOffsets(text: string, options?: EncodeOptions): Encoding {
    return { ids: this.encode(text, options) };
  }

  encodeBatch(texts: string[], options?: EncodeOptions): BatchEncoding {
    return texts.map((text) => this.encodeWithOffsets(text, options));
  }

  decode(tokenIds: number[], _options?: DecodeOptions): string {
    return tokenIds.join(" ");
  }

  decodeBatch(batch: number[][], options?: DecodeOptions): string[] {
    return batch.map((tokenIds) => this.decode(tokenIds, options));
  }
}

class FakeGemmaTextModel {
  constructor(private readonly layerCount: number) {}

  runWithHiddenStates(inputIds: MxArray, _options?: Gemma3TextModelOptions): Gemma3TextModelOutput {
    const [batch, sequenceLength] = inputIds.shape;
    if (batch === undefined || sequenceLength === undefined) {
      throw new Error("FakeGemmaTextModel requires rank-2 input ids.");
    }
    const hiddenStates: MxArray[] = [];
    for (let layerIndex = 0; layerIndex < this.layerCount; layerIndex += 1) {
      hiddenStates.push(
        MxArray.fromData(
          Array.from({ length: batch * sequenceLength * 2 }, () => layerIndex + 1),
          [batch, sequenceLength, 2],
          "float32",
        ),
      );
    }
    return {
      lastHiddenState: MxArray.fromData(
        Array.from({ length: batch * sequenceLength * 2 }, () => 9),
        [batch, sequenceLength, 2],
        "float32",
      ),
      hiddenStates,
    };
  }
}

class FakeGemmaTextEncoder {
  readonly model: FakeGemmaTextModel;

  constructor(layerCount: number) {
    this.model = new FakeGemmaTextModel(layerCount);
  }

  [Symbol.dispose](): void {}
}

class FakeConnector {
  readonly config = tinyConnectorConfig();
  readonly hiddenShapes: number[][] = [];
  readonly masks: unknown[] = [];

  run(textEncoderHiddenStates: MxArray, attentionMask: MxArray) {
    this.hiddenShapes.push([...textEncoderHiddenStates.shape]);
    this.masks.push(attentionMask.toList());
    const [batch, sequenceLength] = attentionMask.shape;
    if (batch === undefined || sequenceLength === undefined) {
      throw new Error("FakeConnector requires rank-2 attention masks.");
    }
    return {
      videoPromptEmbeds: MxArray.fromData(
        Array.from({ length: batch * sequenceLength * 2 }, () => 1),
        [batch, sequenceLength, 2],
        "float32",
      ),
      audioPromptEmbeds: MxArray.fromData(
        Array.from({ length: batch * sequenceLength * 2 }, () => 2),
        [batch, sequenceLength, 2],
        "float32",
      ),
      attentionMask: MxArray.fromData(
        Array.from({ length: batch * sequenceLength }, () => 1),
        [batch, sequenceLength],
        "int32",
      ),
    };
  }

  [Symbol.dispose](): void {}
}

describe("LTX-2 prompt conditioning", () => {
  test("left-pads Gemma prompts and flattens all hidden states for connectors", () => {
    const connector = new FakeConnector();
    using conditioner = createLtx2PromptConditioner({
      tokenizer: new FakeTokenizer(),
      textEncoder: new FakeGemmaTextEncoder(3),
      connectors: connector,
    });
    using result = conditioner.encodePrompt({
      prompt: "hello",
      negativePrompt: "bad",
      includeNegativePrompt: true,
      maxSequenceLength: 4,
    });

    mxEval(
      result.conditioning.promptEmbeds,
      result.conditioning.audioPromptEmbeds,
      result.conditioning.promptAttentionMask,
    );
    expect(result.batchSize).toBe(1);
    expect(result.conditioning.promptEmbeds.shape).toEqual([1, 4, 2]);
    expect(result.conditioning.audioPromptEmbeds.shape).toEqual([1, 4, 2]);
    expect(result.conditioning.negativePromptEmbeds?.shape).toEqual([1, 4, 2]);
    expect(connector.hiddenShapes).toEqual([
      [1, 4, 6],
      [1, 4, 6],
    ]);
    expect(connector.masks).toEqual([[[0, 0, 1, 1]], [[0, 0, 1, 1]]]);
  });

  test("reports truncation and repeats prompt batches", () => {
    const connector = new FakeConnector();
    using conditioner = createLtx2PromptConditioner({
      tokenizer: new FakeTokenizer(),
      textEncoder: new FakeGemmaTextEncoder(3),
      connectors: connector,
    });
    using result = conditioner.encodePrompt({
      prompt: "long",
      numVideosPerPrompt: 2,
      maxSequenceLength: 4,
    });

    mxEval(result.conditioning.promptEmbeds, result.conditioning.promptAttentionMask);
    expect(result.batchSize).toBe(2);
    expect(result.promptTruncated).toBe(true);
    expect(result.conditioning.promptEmbeds.shape).toEqual([2, 4, 2]);
    expect(result.conditioning.promptAttentionMask.shape).toEqual([2, 4]);
  });

  test("rejects Gemma outputs without the connector layer count", () => {
    using conditioner = createLtx2PromptConditioner({
      tokenizer: new FakeTokenizer(),
      textEncoder: new FakeGemmaTextEncoder(2),
      connectors: new FakeConnector(),
    });

    expect(() =>
      conditioner.encodePrompt({
        prompt: "hello",
        maxSequenceLength: 4,
      }),
    ).toThrow("expected 3 Gemma hidden states");
  });
});
