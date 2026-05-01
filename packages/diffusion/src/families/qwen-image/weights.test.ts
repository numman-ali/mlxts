import { describe, expect, test } from "bun:test";
import { MxArray, mxEval } from "@mlxts/core";

import { qwenImageAutoencoderWeightPath, transformQwenImageAutoencoderWeight } from "./weights";

function expectTensorValues(actual: ArrayLike<number>, expected: readonly number[]): void {
  expect(actual.length).toBe(expected.length);
  for (let index = 0; index < expected.length; index += 1) {
    expect(actual[index]).toBe(expected[index]);
  }
}

describe("Qwen-Image VAE weight mapping", () => {
  test("maps Diffusers VAE names onto the package parameter tree", () => {
    expect(qwenImageAutoencoderWeightPath("encoder.conv_in.weight")).toBe("encoder.convIn.weight");
    expect(qwenImageAutoencoderWeightPath("encoder.down_blocks.0.norm1.gamma")).toBe(
      "encoder.downBlocks.0.norm1.weight",
    );
    expect(
      qwenImageAutoencoderWeightPath("decoder.up_blocks.0.upsamplers.0.resample.1.weight"),
    ).toBe("decoder.upBlocks.0.upsampler.resample.weight");
    expect(
      qwenImageAutoencoderWeightPath("decoder.up_blocks.0.upsamplers.0.time_conv.weight"),
    ).toBe("decoder.upBlocks.0.upsampler.timeConv.weight");
    expect(qwenImageAutoencoderWeightPath("post_quant_conv.bias")).toBe("postQuantConv.bias");
    expect(qwenImageAutoencoderWeightPath("")).toBeNull();
    expect(qwenImageAutoencoderWeightPath("encoder.num_batches_tracked")).toBeNull();
  });

  test("transposes Diffusers Conv3d weights into MLX layout", () => {
    using checkpoint = MxArray.fromData([1, 2, 3, 4], [1, 2, 2, 1, 1]);
    using transformed = transformQwenImageAutoencoderWeight(
      "encoder.conv_in.weight",
      "encoder.convIn.weight",
      checkpoint,
    );

    mxEval(transformed);
    expect(transformed.shape).toEqual([1, 2, 1, 1, 2]);
    expectTensorValues(transformed.toTypedArray(), [1, 3, 2, 4]);
  });

  test("transposes Diffusers Conv2d weights into MLX layout", () => {
    using checkpoint = MxArray.fromData([1, 2, 3, 4], [1, 2, 1, 2]);
    using transformed = transformQwenImageAutoencoderWeight(
      "decoder.mid_block.attentions.0.to_qkv.weight",
      "decoder.midBlock.attentions.0.toQkv.weight",
      checkpoint,
    );

    mxEval(transformed);
    expect(transformed.shape).toEqual([1, 1, 2, 2]);
    expectTensorValues(transformed.toTypedArray(), [1, 3, 2, 4]);
  });

  test("squeezes RMS normalization gamma tensors", () => {
    using checkpoint = MxArray.fromData([5, 6], [2, 1, 1, 1]);
    using transformed = transformQwenImageAutoencoderWeight(
      "encoder.down_blocks.0.norm1.gamma",
      "encoder.downBlocks.0.norm1.weight",
      checkpoint,
    );

    mxEval(transformed);
    expect(transformed.shape).toEqual([2]);
    expectTensorValues(transformed.toTypedArray(), [5, 6]);
  });

  test("leaves bias tensors in their checkpoint layout", () => {
    using checkpoint = MxArray.fromData([1, 2], [2]);
    const transformed = transformQwenImageAutoencoderWeight(
      "encoder.conv_in.bias",
      "encoder.convIn.bias",
      checkpoint,
    );

    expect(transformed).toBe(checkpoint);
  });
});
