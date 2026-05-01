import { describe, expect, test } from "bun:test";
import { array, MxArray, mxEval, zeros } from "@mlxts/core";

import { ZImageSelfAttention } from "./attention";
import { ZImageFinalLayer, ZImageTransformerBlock } from "./blocks";
import {
  ZImageCaptionEmbedder,
  ZImageRopeEmbedder,
  ZImageTimestepEmbedder,
  zImageTimestepEmbedding,
} from "./embeddings";
import {
  assertAttention4d,
  assertFeature2d,
  assertIds2d,
  assertSequence3d,
  checkedModule,
  selectLastAxis,
  sliceAxis,
  sliceLastAxis,
} from "./tensor-utils";

describe("Z-Image embedding and block coverage", () => {
  test("runs timestep, caption, and RoPE embedders", () => {
    using timesteps = MxArray.fromData([0.25], [1]);
    using embedding = zImageTimestepEmbedding(timesteps, 8, { dtype: "float16" });
    using timestepEmbedder = new ZImageTimestepEmbedder(12, 16);
    using projectedTime = timestepEmbedder.forward(timesteps);
    using caption = MxArray.fromData(
      Array.from({ length: 12 }, (_, index) => index / 12),
      [3, 4],
    );
    using captionEmbedder = new ZImageCaptionEmbedder(4, 12, 1e-5);
    using projectedCaption = captionEmbedder.forward(caption);
    using ids = MxArray.fromData([0, 0, 0, 0, 0, 1], [2, 3], "int32");
    using ropeEmbedder = new ZImageRopeEmbedder(6, 256, [2, 2, 2], [8, 8, 8]);
    using rope = ropeEmbedder.forward(ids);

    mxEval(embedding, projectedTime, projectedCaption, rope);
    expect(embedding.shape).toEqual([1, 8]);
    expect(embedding.dtype).toBe("float16");
    expect(projectedTime.shape).toEqual([1, 12]);
    expect(timestepEmbedder.outputDims).toBe(12);
    expect(projectedCaption.shape).toEqual([3, 12]);
    expect(rope.shape).toEqual([1, 1, 2, 3, 2, 2]);
    expect(ropeEmbedder.headDim).toBe(6);
  });

  test("rejects invalid Z-Image embedding inputs", () => {
    using scalarTimesteps = array(0);
    using timesteps = MxArray.fromData([1], [1]);
    using badCaption = zeros([2, 3]);
    using captionEmbedder = new ZImageCaptionEmbedder(4, 12, 1e-5);
    using badIds = zeros([2, 2]);

    expect(() => zImageTimestepEmbedding(scalarTimesteps)).toThrow("rank-1");
    expect(() => zImageTimestepEmbedding(timesteps, 3)).toThrow("even integer");
    expect(() => captionEmbedder.forward(badCaption)).toThrow("expected last dimension");
    expect(() => new ZImageRopeEmbedder(6, 256, [2, 2, 4], [8, 8, 8])).toThrow("sum");
    expect(() => new ZImageRopeEmbedder(6, 256, [2, 3, 1], [8, 8, 8])).toThrow("even");
    expect(() => new ZImageRopeEmbedder(6, 256, [2, 2, 2], [8, 8, 8]).forward(badIds)).toThrow(
      "[length, 3]",
    );
  });

  test("runs modulated, unmodulated, attention, and final block paths", () => {
    using ids = MxArray.fromData([0, 0, 0, 0, 0, 1], [2, 3], "int32");
    using ropeEmbedder = new ZImageRopeEmbedder(6, 256, [2, 2, 2], [8, 8, 8]);
    using rope = ropeEmbedder.forward(ids);
    using hidden = MxArray.fromData(
      Array.from({ length: 24 }, (_, index) => index / 24),
      [1, 2, 12],
    );
    using mask = MxArray.fromData([1, 1], [1, 2], "bool");
    using adaln = zeros([1, 12]);
    using modulated = new ZImageTransformerBlock({
      hiddenSize: 12,
      numHeads: 2,
      normEps: 1e-5,
      qkNorm: true,
      modulation: true,
      adalnDims: 12,
    });
    using unmodulated = new ZImageTransformerBlock({
      hiddenSize: 12,
      numHeads: 2,
      normEps: 1e-5,
      qkNorm: false,
      modulation: false,
      adalnDims: 12,
    });
    using attention = new ZImageSelfAttention(12, 2, false, 1e-5);
    using finalLayer = new ZImageFinalLayer(12, 12, 4);

    using modulatedOutput = modulated.forward(hidden, rope, adaln, mask);
    using unmodulatedOutput = unmodulated.forward(hidden, rope);
    using attentionOutput = attention.forward(hidden, rope, mask);
    using finalOutput = finalLayer.forward(hidden, adaln);

    mxEval(modulatedOutput, unmodulatedOutput, attentionOutput, finalOutput);
    expect(modulatedOutput.shape).toEqual([1, 2, 12]);
    expect(unmodulatedOutput.shape).toEqual([1, 2, 12]);
    expect(attentionOutput.shape).toEqual([1, 2, 12]);
    expect(attention.hiddenSize).toBe(12);
    expect(modulated.hiddenSize).toBe(12);
    expect(unmodulated.hiddenSize).toBe(12);
    expect(finalOutput.shape).toEqual([1, 2, 4]);
    expect(finalLayer.hiddenSize).toBe(12);
    expect(() => modulated.forward(hidden, rope)).toThrow("adalnInput");
  });
});

describe("Z-Image tensor utility coverage", () => {
  test("validates helper shapes and slices axes", () => {
    using sequence = zeros([1, 2, 3]);
    using feature = zeros([2, 3]);
    using attention = zeros([1, 2, 3, 4]);
    using ids = zeros([2, 3]);
    using sliced = sliceAxis(sequence, 1, 0, 1);
    using lastSliced = sliceLastAxis(sequence, 0, 2);
    using selected = selectLastAxis(sequence, 1);

    expect(assertSequence3d(sequence, "test")).toEqual({ batch: 1, length: 2, channels: 3 });
    expect(assertFeature2d(feature, "test")).toEqual({ length: 2, channels: 3 });
    expect(assertAttention4d(attention, "test")).toEqual({
      batch: 1,
      heads: 2,
      length: 3,
      headDim: 4,
    });
    expect(assertIds2d(ids, "test")).toEqual({ length: 2 });
    expect(checkedModule(["a"], 0, "test")).toBe("a");
    expect(sliced.shape).toEqual([1, 1, 3]);
    expect(lastSliced.shape).toEqual([1, 2, 2]);
    expect(selected.shape).toEqual([1, 2]);
  });

  test("rejects invalid helper shapes and missing modules", () => {
    using rank1 = zeros([2]);
    using ids = zeros([2, 2]);

    expect(() => assertSequence3d(rank1, "sequence")).toThrow("rank-3");
    expect(() => assertFeature2d(rank1, "feature")).toThrow("rank-2");
    expect(() => assertAttention4d(rank1, "attention")).toThrow("rank-4");
    expect(() => assertIds2d(ids, "ids")).toThrow("[length, 3]");
    expect(() => checkedModule([], 0, "modules")).toThrow("missing module");
  });
});
