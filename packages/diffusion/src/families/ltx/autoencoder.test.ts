import { describe, expect, test } from "bun:test";
import { MxArray, mxEval } from "@mlxts/core";

import { LtxVideoAutoencoderKL, unpatchLtxVideoDecoderOutput } from "./autoencoder";
import { LtxVideoResnetBlock3d, LtxVideoUpsampler3d } from "./autoencoder-blocks";
import type { LtxVideoAutoencoderConfig } from "./config";

function config(overrides: Partial<LtxVideoAutoencoderConfig> = {}): LtxVideoAutoencoderConfig {
  return {
    inChannels: 3,
    outChannels: 1,
    latentChannels: 1,
    latentChannelsOut: 2,
    blockOutChannels: [1],
    downBlockTypes: ["LTXVideoDownBlock3D"],
    decoderBlockOutChannels: [1],
    layersPerBlock: [0, 0],
    decoderLayersPerBlock: [0, 0],
    spatioTemporalScaling: [false],
    decoderSpatioTemporalScaling: [false],
    decoderInjectNoise: [false, false],
    downsampleTypes: ["conv"],
    upsampleResidual: [false],
    upsampleFactors: [1],
    timestepConditioning: false,
    patchSize: 2,
    patchSizeT: 1,
    resnetNormEps: 1e-6,
    scalingFactor: 1,
    encoderCausal: true,
    decoderCausal: false,
    spatialCompressionRatio: null,
    temporalCompressionRatio: null,
    rawConfig: {},
    ...overrides,
  };
}

function expectCloseList(actual: ArrayLike<number>, expected: readonly number[]): void {
  expect(actual.length).toBe(expected.length);
  for (let index = 0; index < expected.length; index += 1) {
    expect(actual[index]).toBeCloseTo(expected[index] ?? Number.NaN, 6);
  }
}

describe("LtxVideoAutoencoderKL", () => {
  test("tracks decoder-only VAE metadata and latent stats", () => {
    using vae = new LtxVideoAutoencoderKL(config());

    expect(vae.latentChannels).toBe(1);
    expect(vae.scalingFactor).toBe(1);
    expect(vae.spatialCompressionRatio).toBe(2);
    expect(vae.temporalCompressionRatio).toBe(1);

    vae.setLatentStats([2], [3]);
    expect(vae.latentsMean).toEqual([2]);
    expect(vae.latentsStd).toEqual([3]);
  });

  test("unpatches decoder channels in Diffusers spatial order", () => {
    using packed = MxArray.fromData([1, 2, 3, 4], [1, 1, 1, 1, 4]);
    using unpacked = unpatchLtxVideoDecoderOutput(packed, 1, 2, 1);

    mxEval(unpacked);
    expect(unpacked.shape).toEqual([1, 1, 2, 2, 1]);
    expectCloseList(unpacked.toTypedArray(), [1, 3, 2, 4]);
  });

  test("runs a tiny decoder from BCFHW latents to BCFHW samples", () => {
    using vae = new LtxVideoAutoencoderKL(config());
    using latents = MxArray.fromData([0], [1, 1, 1, 1, 1]);
    using decoded = vae.decodeRaw(latents);
    using forwarded = vae.forward(latents);

    mxEval(decoded, forwarded);
    expect(decoded.shape).toEqual([1, 1, 1, 2, 2]);
    expect(forwarded.shape).toEqual([1, 1, 1, 2, 2]);
    expect(Array.from(decoded.toTypedArray()).every(Number.isFinite)).toBe(true);
  });

  test("runs residual and spatiotemporal upsampling decoder blocks", () => {
    using input = MxArray.fromData([0], [1, 1, 1, 1, 1]);
    using sameBlock = new LtxVideoResnetBlock3d(1, 1);
    using projectedBlock = new LtxVideoResnetBlock3d(1, 2);
    using upsampler = new LtxVideoUpsampler3d(1, [2, 2, 2]);
    using same = sameBlock.forward(input);
    using projected = projectedBlock.forward(input);
    using upsampled = upsampler.forward(input);

    mxEval(same, projected, upsampled);
    expect(same.shape).toEqual([1, 1, 1, 1, 1]);
    expect(projected.shape).toEqual([1, 1, 1, 1, 2]);
    expect(upsampled.shape).toEqual([1, 1, 2, 2, 1]);
  });

  test("rejects unsupported timestep-conditioned decoder variants", () => {
    expect(() => new LtxVideoAutoencoderKL(config({ timestepConditioning: true }))).toThrow(
      "timestep-conditioned",
    );
  });

  test("rejects unsupported residual and factorized decoder upsampling variants", () => {
    expect(() => new LtxVideoAutoencoderKL(config({ upsampleFactors: [2] }))).toThrow(
      "upsample factors",
    );
    expect(() => new LtxVideoAutoencoderKL(config({ upsampleResidual: [true] }))).toThrow(
      "residual",
    );
  });
});
