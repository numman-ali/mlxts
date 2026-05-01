import { describe, expect, test } from "bun:test";
import { MxArray, mxEval } from "@mlxts/core";

import type { Ltx2TextConnectorsConfig } from "./config";
import { disposeLtx2TextConnectorOutput, Ltx2TextConnectors } from "./connectors-ltx2";
import {
  ltx2ConnectorBinaryMask,
  ltx2FlattenTextHiddenStates,
  ltx2PerLayerMaskedMeanNorm,
  ltx2PerTokenRmsNorm,
  retainLtx2HiddenStack,
} from "./connectors-ltx2-normalization";
import {
  applyLtx2ConnectorRotary,
  createLtx2ConnectorRotaryEmbeddings,
} from "./connectors-ltx2-rotary";
import {
  disposeLtx2ConnectorTransformerOutput,
  Ltx2ConnectorTransformer1d,
  replaceLtx2ConnectorPaddingWithRegisters,
} from "./connectors-ltx2-transformer";

function tinyConnectorConfig(
  overrides: Partial<Ltx2TextConnectorsConfig> = {},
): Ltx2TextConnectorsConfig {
  return {
    captionChannels: 2,
    textProjInFactor: 2,
    textEncoderDim: 4,
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
    ...overrides,
  };
}

function expectCloseList(actual: ArrayLike<number>, expected: readonly number[]): void {
  expect(actual.length).toBe(expected.length);
  for (let index = 0; index < expected.length; index += 1) {
    expect(Number(actual[index])).toBeCloseTo(expected[index] ?? Number.NaN, 5);
  }
}

describe("LTX-2 text connectors", () => {
  test("normalizes hidden-state stacks with masked mean/range semantics", () => {
    const config = tinyConnectorConfig();
    using hiddenStates = MxArray.fromData(
      [1, 3, 5, 7, 2, 4, 6, 8, 9, 9, 9, 9],
      [1, 3, 2, 2],
      "float32",
    );
    using mask = MxArray.fromData([1, 1, 0], [1, 3], "int32");
    using normalized = ltx2PerLayerMaskedMeanNorm(hiddenStates, mask, config);

    normalized.eval();
    expect(normalized.shape).toEqual([1, 3, 4]);
    expectCloseList(Array.from(normalized.toTypedArray()).slice(8, 12), [0, 0, 0, 0]);
  });

  test("normalizes each valid token before per-modality projection", () => {
    const config = tinyConnectorConfig({ perModalityProjections: true });
    using hiddenStates = MxArray.fromData(
      [3, 4, 0, 0, 5, 12, 0, 0, 7, 8, 9, 10],
      [1, 3, 2, 2],
      "float32",
    );
    using mask = MxArray.fromData([1, 1, 0], [1, 3], "int32");
    using normalized = ltx2PerTokenRmsNorm(hiddenStates, mask, config);

    normalized.eval();
    expect(normalized.shape).toEqual([1, 3, 4]);
    expectCloseList(Array.from(normalized.toTypedArray()).slice(8, 12), [0, 0, 0, 0]);
  });

  test("accepts flattened and stacked Gemma hidden-state tensors", () => {
    const config = tinyConnectorConfig();
    using flattenedInput = MxArray.fromData([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 4], "float32");
    using flattened = ltx2FlattenTextHiddenStates(flattenedInput, config);
    using retainedFromFlat = retainLtx2HiddenStack(flattenedInput, config);
    using stackedInput = MxArray.fromData([1, 2, 3, 4], [1, 1, 2, 2], "float32");
    using retainedFromStack = retainLtx2HiddenStack(stackedInput, config);

    mxEval(flattened, retainedFromFlat, retainedFromStack);
    expect(flattened.shape).toEqual([1, 2, 4]);
    expect(retainedFromFlat.shape).toEqual([1, 2, 2, 2]);
    expect(retainedFromStack.shape).toEqual([1, 1, 2, 2]);
  });

  test("rejects malformed connector hidden-state and mask shapes", () => {
    const config = tinyConnectorConfig();
    using badHiddenStates = MxArray.fromData([1, 2, 3, 4, 5, 6], [1, 1, 3, 2], "float32");
    using hiddenStates = MxArray.fromData([1, 2, 3, 4], [1, 1, 2, 2], "float32");
    using badMask = MxArray.fromData([1, 1], [1, 2], "int32");

    expect(() => ltx2PerLayerMaskedMeanNorm(badHiddenStates, badMask, config)).toThrow(
      /text hidden states/,
    );
    expect(() => ltx2PerLayerMaskedMeanNorm(hiddenStates, badMask, config)).toThrow(
      /attention mask/,
    );
  });

  test("retains binary connector masks", () => {
    using mask = MxArray.fromData([1, 0], [1, 2], "int32");
    using binaryMask = ltx2ConnectorBinaryMask(mask, 1, 2);

    binaryMask.eval();
    expect(binaryMask.shape).toEqual([1, 2]);
    expectCloseList(binaryMask.toTypedArray(), [1, 0]);
  });

  test("replaces padding tokens with learned connector registers", () => {
    using hiddenStates = MxArray.fromData([10, 10, 20, 20, 30, 30, 40, 40], [1, 4, 2], "float32");
    using mask = MxArray.fromData([0, 0, 1, 1], [1, 4], "int32");
    using registers = MxArray.fromData([1, 1, 2, 2], [2, 2], "float32");
    using replaced = replaceLtx2ConnectorPaddingWithRegisters(hiddenStates, mask, registers);

    replaced.eval();
    expect(replaced.shape).toEqual([1, 4, 2]);
    expectCloseList(replaced.toTypedArray(), [30, 30, 40, 40, 1, 1, 2, 2]);
  });

  test("validates connector RoPE geometry", () => {
    expect(() =>
      createLtx2ConnectorRotaryEmbeddings({
        batch: 1,
        length: 1,
        dim: 1,
        theta: 4,
        baseSequenceLength: 4,
        ropeType: "interleaved",
        heads: 1,
      }),
    ).toThrow(/dim must be at least 2/);
    expect(() =>
      createLtx2ConnectorRotaryEmbeddings({
        batch: 1,
        length: 1,
        dim: 6,
        theta: 4,
        baseSequenceLength: 4,
        ropeType: "split",
        heads: 2,
      }),
    ).toThrow(/split RoPE/);

    const oddInterleaved = createLtx2ConnectorRotaryEmbeddings({
      batch: 1,
      length: 1,
      dim: 3,
      theta: 4,
      baseSequenceLength: 4,
      ropeType: "interleaved",
      heads: 1,
    });
    try {
      oddInterleaved.cos.eval();
      expect(oddInterleaved.cos.shape).toEqual([1, 1, 3]);
      expectCloseList(oddInterleaved.cos.toTypedArray().slice(0, 1), [1]);
    } finally {
      oddInterleaved.cos.free();
      oddInterleaved.sin.free();
    }

    const interleaved = createLtx2ConnectorRotaryEmbeddings({
      batch: 1,
      length: 1,
      dim: 4,
      theta: 4,
      baseSequenceLength: 4,
      ropeType: "interleaved",
      heads: 1,
    });
    try {
      using oddHidden = MxArray.fromData([1, 2, 3], [1, 1, 3], "float32");
      using shortHidden = MxArray.fromData([1, 2], [1, 1, 2], "float32");
      expect(() => applyLtx2ConnectorRotary(oddHidden, interleaved, "interleaved", 1)).toThrow(
        /hidden size must be even/,
      );
      expect(() => applyLtx2ConnectorRotary(shortHidden, interleaved, "interleaved", 1)).toThrow(
        /RoPE shape mismatch/,
      );
    } finally {
      interleaved.cos.free();
      interleaved.sin.free();
    }

    const split = createLtx2ConnectorRotaryEmbeddings({
      batch: 1,
      length: 1,
      dim: 4,
      theta: 4,
      baseSequenceLength: 4,
      ropeType: "split",
      heads: 1,
    });
    try {
      using badHeadHidden = MxArray.fromData([1, 2, 3, 4, 5, 6], [1, 1, 6], "float32");
      using wrongLengthHidden = MxArray.fromData([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 4], "float32");
      expect(() => applyLtx2ConnectorRotary(badHeadHidden, split, "split", 2)).toThrow(
        /per-head hidden size/,
      );
      expect(() => applyLtx2ConnectorRotary(wrongLengthHidden, split, "split", 1)).toThrow(
        /RoPE shape mismatch/,
      );
    } finally {
      split.cos.free();
      split.sin.free();
    }
  });

  test("runs split-RoPE connector transformer blocks", () => {
    const connector = new Ltx2ConnectorTransformer1d({
      heads: 1,
      headDim: 2,
      numLayers: 1,
      numLearnableRegisters: null,
      ropeBaseSeqLen: 4,
      ropeTheta: 4,
      ropeType: "split",
      gatedAttention: false,
    });
    using hiddenStates = MxArray.fromData([1, 2, 3, 4], [1, 2, 2], "float32");
    using mask = MxArray.fromData([1, 1], [1, 2], "int32");
    const output = connector.run(hiddenStates, mask);
    try {
      mxEval(output.hiddenStates, output.attentionMask);
      expect(output.hiddenStates.shape).toEqual([1, 2, 2]);
      expect(output.attentionMask.shape).toEqual([1, 2]);
      expectCloseList(output.attentionMask.toTypedArray(), [1, 1]);
    } finally {
      disposeLtx2ConnectorTransformerOutput(output);
    }
  });

  test("emits video and audio prompt embeddings with shared projection", () => {
    const connectors = new Ltx2TextConnectors(tinyConnectorConfig());
    using hiddenStates = MxArray.fromData([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 2, 2], "float32");
    using mask = MxArray.fromData([1, 0], [1, 2], "int32");
    const output = connectors.run(hiddenStates, mask);
    try {
      mxEval(output.videoPromptEmbeds, output.audioPromptEmbeds, output.attentionMask);
      expect(output.videoPromptEmbeds.shape).toEqual([1, 2, 2]);
      expect(output.audioPromptEmbeds.shape).toEqual([1, 2, 2]);
      expect(output.attentionMask.shape).toEqual([1, 2]);
      expectCloseList(output.attentionMask.toTypedArray(), [1, 1]);
    } finally {
      disposeLtx2TextConnectorOutput(output);
    }
  });

  test("emits per-modality projection shapes", () => {
    const connectors = new Ltx2TextConnectors(
      tinyConnectorConfig({
        perModalityProjections: true,
        videoHiddenDim: 2,
        audioHiddenDim: 2,
      }),
    );
    using hiddenStates = MxArray.fromData([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 2, 2], "float32");
    using mask = MxArray.fromData([1, 1], [1, 2], "int32");
    const output = connectors.run(hiddenStates, mask);
    try {
      mxEval(output.videoPromptEmbeds, output.audioPromptEmbeds, output.attentionMask);
      expect(output.videoPromptEmbeds.shape).toEqual([1, 2, 2]);
      expect(output.audioPromptEmbeds.shape).toEqual([1, 2, 2]);
      expectCloseList(output.attentionMask.toTypedArray(), [1, 1]);
    } finally {
      disposeLtx2TextConnectorOutput(output);
    }
  });

  test("rejects per-modality projection dims that do not match connector hidden sizes", () => {
    expect(
      () =>
        new Ltx2TextConnectors(
          tinyConnectorConfig({
            perModalityProjections: true,
            videoHiddenDim: 4,
            audioHiddenDim: 2,
          }),
        ),
    ).toThrow(/per-modality projection dims/);
  });
});
