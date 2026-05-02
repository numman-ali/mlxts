import { describe, expect, test } from "bun:test";
import { MxArray, mxEval, treeFlatten } from "@mlxts/core";

import { QwenImageAutoencoderKL } from "./autoencoder";
import { QwenImageResample } from "./autoencoder-blocks";
import type { QwenImageAutoencoderConfig } from "./config";

function tinyConfig(): QwenImageAutoencoderConfig {
  return {
    baseDim: 2,
    latentChannels: 1,
    latentChannelsOut: 2,
    dimMultipliers: [1],
    numResBlocks: 0,
    attentionScales: [],
    temporalDownsample: [],
    temporalUpsample: [],
    dropout: 0,
    inputChannels: 3,
    latentsMean: [0],
    latentsStd: [1],
    spatialCompressionRatio: 1,
    rawConfig: {},
  };
}

describe("QwenImageAutoencoderKL", () => {
  test("constructs a Qwen-Image VAE parameter tree without inherited block parameters", () => {
    using autoencoder = new QwenImageAutoencoderKL(tinyConfig());
    const paths = treeFlatten(autoencoder.parameters()).map(([path]) => path.join("."));

    expect(paths).toContain("encoder.convIn.weight");
    expect(paths).toContain("encoder.midBlock.attentions.0.toQkv.weight");
    expect(paths).toContain("quantConv.weight");
    expect(paths).toContain("postQuantConv.weight");
    expect(paths).toContain("decoder.convOut.weight");
    expect(paths).not.toContain("norm1.weight");
    expect(autoencoder.latentChannels).toBe(1);
    expect(autoencoder.spatialCompressionRatio).toBe(1);
    expect(autoencoder.latentsMean).toEqual([0]);
    expect(autoencoder.latentsStd).toEqual([1]);
  });

  test("keeps encode/decode tensor boundaries in NCFHW layout", () => {
    using autoencoder = new QwenImageAutoencoderKL(tinyConfig());
    autoencoder.eval();
    using sample = MxArray.fromData(
      [-1, -0.5, 0.25, 0.5, 0.75, -0.25, 0.125, -0.125, 0.875, -0.875, 0.375, -0.375],
      [1, 3, 1, 2, 2],
    );
    using moments = autoencoder.encodeMoments(sample);
    using encoded = autoencoder.encodeRaw(sample);
    using latents = MxArray.fromData([0.1, -0.1, 0.2, -0.2], [1, 1, 1, 2, 2]);
    using decoded = autoencoder.decodeRaw(latents);
    using reconstructed = autoencoder.forward(sample);

    mxEval(moments, encoded, decoded, reconstructed);
    expect(moments.shape).toEqual([1, 2, 1, 2, 2]);
    expect(encoded.shape).toEqual([1, 1, 1, 2, 2]);
    expect(decoded.shape).toEqual([1, 3, 1, 2, 2]);
    expect(reconstructed.shape).toEqual([1, 3, 1, 2, 2]);
  });

  test("constructs temporal downsample and upsample blocks from config flags", () => {
    using autoencoder = new QwenImageAutoencoderKL({
      ...tinyConfig(),
      dimMultipliers: [1, 2],
      temporalDownsample: [true],
      temporalUpsample: [true],
    });
    const temporalDownsample = autoencoder.encoder.downBlocks.find(
      (block) => block instanceof QwenImageResample && block.mode === "downsample3d",
    );
    const firstUpBlock = autoencoder.decoder.upBlocks[0];

    expect(temporalDownsample).toBeInstanceOf(QwenImageResample);
    expect(firstUpBlock?.upsampler?.mode).toBe("upsample3d");
  });

  test("rejects invalid VAE topology configs", () => {
    expect(() => new QwenImageAutoencoderKL({ ...tinyConfig(), dimMultipliers: [] })).toThrow(
      "dimMultipliers",
    );
    expect(() => new QwenImageAutoencoderKL({ ...tinyConfig(), latentChannelsOut: 3 })).toThrow(
      "latentChannelsOut",
    );
    expect(
      () => new QwenImageAutoencoderKL({ ...tinyConfig(), temporalDownsample: [true] }),
    ).toThrow("temporalDownsample");
  });
});
