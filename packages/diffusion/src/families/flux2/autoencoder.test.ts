import { describe, expect, test } from "bun:test";
import { MxArray, zeros } from "@mlxts/core";

import { Flux2KleinAutoencoderKL } from "./autoencoder";
import type { Flux2KleinAutoencoderConfig } from "./config";

function tinyFlux2VaeConfig(
  overrides: Partial<Flux2KleinAutoencoderConfig> = {},
): Flux2KleinAutoencoderConfig {
  return {
    inChannels: 3,
    outChannels: 3,
    latentChannels: 2,
    latentChannelsOut: 4,
    packedLatentChannels: 8,
    useQuantConv: true,
    usePostQuantConv: true,
    blockOutChannels: [4, 8],
    decoderBlockOutChannels: null,
    layersPerBlock: 1,
    normNumGroups: 1,
    forceUpcast: false,
    midBlockAddAttention: true,
    batchNormEps: 0.0001,
    batchNormMomentum: 0.1,
    patchSize: [2, 2],
    sampleSize: 16,
    vaeScaleFactor: 2,
    downBlockTypes: ["DownEncoderBlock2D", "DownEncoderBlock2D"],
    upBlockTypes: ["UpDecoderBlock2D", "UpDecoderBlock2D"],
    rawConfig: {},
    ...overrides,
  };
}

describe("Flux2KleinAutoencoderKL", () => {
  test("exposes FLUX.2 latent metadata and NCHW decode boundary", () => {
    using vae = new Flux2KleinAutoencoderKL(tinyFlux2VaeConfig());

    expect(vae.latentChannels).toBe(2);
    expect(vae.packedLatentChannels).toBe(8);
    expect(vae.patchSize).toBe(2);
    expect(vae.vaeScaleFactor).toBe(2);
    expect(vae.batchNormMean).toEqual([0, 0, 0, 0, 0, 0, 0, 0]);
    expect(vae.batchNormVar).toEqual([1, 1, 1, 1, 1, 1, 1, 1]);

    using latents = zeros([1, 2, 2, 2]);
    using decoded = vae.decode(latents);

    expect(decoded.shape).toEqual([1, 3, 4, 4]);
  });

  test("encodes and splits NCHW posterior moments on the channel axis", () => {
    using vae = new Flux2KleinAutoencoderKL(tinyFlux2VaeConfig());

    using image = zeros([1, 3, 4, 4]);
    using moments = vae.encodeMoments(image);
    using posterior = vae.splitMoments(moments);

    expect(moments.shape).toEqual([1, 4, 2, 2]);
    expect(posterior.mean.shape).toEqual([1, 2, 2, 2]);
    expect(posterior.logVariance.shape).toEqual([1, 2, 2, 2]);
  });

  test("honors decoder_block_out_channels for small decoder snapshots", () => {
    using vae = new Flux2KleinAutoencoderKL(
      tinyFlux2VaeConfig({
        decoderBlockOutChannels: [6, 10],
      }),
    );

    expect(vae.encoder.convIn.weight.shape).toEqual([4, 3, 3, 3]);
    expect(vae.decoder.convIn.weight.shape).toEqual([10, 3, 3, 2]);
    expect(vae.decoder.convOut.weight.shape).toEqual([3, 3, 3, 6]);
  });

  test("loads copied batch-norm stats and rejects invalid stats", () => {
    using vae = new Flux2KleinAutoencoderKL(tinyFlux2VaeConfig());
    const mean = [0, 1, 2, 3, 4, 5, 6, 7];
    const variance = [1, 2, 3, 4, 5, 6, 7, 8];

    vae.setBatchNormStats(mean, variance);
    mean[0] = 100;

    expect(vae.batchNormMean).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
    expect(vae.batchNormVar).toEqual(variance);
    expect(() => vae.setBatchNormStats([1], variance)).toThrow("batchNormMean");
    expect(() =>
      vae.setBatchNormStats(
        Array.from({ length: 8 }, () => Number.NaN),
        variance,
      ),
    ).toThrow("finite");
  });

  test("rejects moment tensors with the wrong channel count", () => {
    using vae = new Flux2KleinAutoencoderKL(tinyFlux2VaeConfig());
    using moments = MxArray.fromData([1, 2, 3], [1, 3, 1, 1]);

    expect(() => vae.splitMoments(moments)).toThrow("channel dimension");
  });
});
