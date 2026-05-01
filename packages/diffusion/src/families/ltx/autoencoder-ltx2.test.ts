import { describe, expect, test } from "bun:test";
import { MxArray, mxEval } from "@mlxts/core";

import { Ltx2VideoAutoencoderKL, unpatchLtx2VideoDecoderOutput } from "./autoencoder-ltx2";
import {
  Ltx2VideoCausalConv3d,
  Ltx2VideoMidBlock3d,
  Ltx2VideoResnetBlock3d,
  Ltx2VideoUpBlock3d,
  Ltx2VideoUpsampler3d,
} from "./autoencoder-ltx2-blocks";
import type { Ltx2VideoAutoencoderConfig } from "./config";

function config(overrides: Partial<Ltx2VideoAutoencoderConfig> = {}): Ltx2VideoAutoencoderConfig {
  return {
    inChannels: 3,
    outChannels: 1,
    latentChannels: 1,
    latentChannelsOut: 2,
    blockOutChannels: [8],
    downBlockTypes: ["LTX2VideoDownBlock3D"],
    decoderBlockOutChannels: [8],
    layersPerBlock: [0, 0],
    decoderLayersPerBlock: [0, 0],
    spatioTemporalScaling: [true],
    decoderSpatioTemporalScaling: [true],
    decoderInjectNoise: [false, false],
    downsampleTypes: ["spatiotemporal"],
    upsampleTypes: ["spatiotemporal"],
    upsampleResidual: [true],
    upsampleFactors: [2],
    timestepConditioning: false,
    patchSize: 1,
    patchSizeT: 1,
    resnetNormEps: 1e-6,
    scalingFactor: 1,
    encoderCausal: true,
    decoderCausal: true,
    encoderSpatialPaddingMode: "zeros",
    decoderSpatialPaddingMode: "reflect",
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

describe("Ltx2VideoAutoencoderKL", () => {
  test("tracks decoder-only VAE metadata and latent stats", () => {
    using vae = new Ltx2VideoAutoencoderKL(config());

    expect(vae.latentChannels).toBe(1);
    expect(vae.scalingFactor).toBe(1);
    expect(vae.spatialCompressionRatio).toBe(2);
    expect(vae.temporalCompressionRatio).toBe(2);

    vae.setLatentStats([2], [3]);
    expect(vae.latentsMean).toEqual([2]);
    expect(vae.latentsStd).toEqual([3]);
  });

  test("unpatches decoder channels in Diffusers spatial order", () => {
    using packed = MxArray.fromData([1, 2, 3, 4], [1, 1, 1, 1, 4]);
    using unpacked = unpatchLtx2VideoDecoderOutput(packed, 1, 2, 1);

    mxEval(unpacked);
    expect(unpacked.shape).toEqual([1, 1, 2, 2, 1]);
    expectCloseList(unpacked.toTypedArray(), [1, 3, 2, 4]);
  });

  test("runs reflect-padded causal conv and residual upsampler", () => {
    using input = MxArray.fromData(
      Array.from({ length: 32 }, (_, index) => index / 32),
      [1, 1, 2, 2, 8],
    );
    using conv = new Ltx2VideoCausalConv3d(8, 8, 3, 1, 1, true, "reflect");
    using upsampler = new Ltx2VideoUpsampler3d(8, [2, 2, 2], true, 2, "reflect");
    using convolved = conv.forward(input);
    using upsampled = upsampler.forward(input);

    mxEval(convolved, upsampled);
    expect(convolved.shape).toEqual([1, 1, 2, 2, 8]);
    expect(upsampled.shape).toEqual([1, 1, 4, 4, 4]);
    expect(Array.from(upsampled.toTypedArray()).every(Number.isFinite)).toBe(true);
  });

  test("runs residual, mid, and up decoder blocks through their Module forward surfaces", () => {
    using blockInput = MxArray.fromData(
      Array.from({ length: 16 }, (_, index) => index / 16),
      [1, 1, 2, 2, 4],
    );
    using upBlockInput = MxArray.fromData(
      Array.from({ length: 32 }, (_, index) => index / 32),
      [1, 1, 2, 2, 8],
    );
    using resnet = new Ltx2VideoResnetBlock3d(4, 8, "reflect");
    using midBlock = new Ltx2VideoMidBlock3d(4, 1, "reflect");
    using upBlock = new Ltx2VideoUpBlock3d(4, 4, 1, true, true, 2, "reflect");
    using resnetOutput = resnet.forward(blockInput);
    using midOutput = midBlock.forward(blockInput);
    using upBlockOutput = upBlock.forward(upBlockInput);

    mxEval(resnetOutput, midOutput, upBlockOutput);
    expect(resnetOutput.shape).toEqual([1, 1, 2, 2, 8]);
    expect(midOutput.shape).toEqual([1, 1, 2, 2, 4]);
    expect(upBlockOutput.shape).toEqual([1, 1, 4, 4, 4]);
    expect(Array.from(upBlockOutput.toTypedArray()).every(Number.isFinite)).toBe(true);
  });

  test("runs a tiny decoder from BCFHW latents to BCFHW samples", () => {
    using vae = new Ltx2VideoAutoencoderKL(config());
    using latents = MxArray.fromData([0, 0, 0, 0], [1, 1, 1, 2, 2]);
    using decoded = vae.decodeRaw(latents);
    using forwarded = vae.forward(latents);

    mxEval(decoded, forwarded);
    expect(decoded.shape).toEqual([1, 1, 1, 4, 4]);
    expect(forwarded.shape).toEqual([1, 1, 1, 4, 4]);
    expect(Array.from(decoded.toTypedArray()).every(Number.isFinite)).toBe(true);
  });

  test("rejects unsupported timestep, noise, and temporal patch variants", () => {
    expect(() => new Ltx2VideoAutoencoderKL(config({ timestepConditioning: true }))).toThrow(
      "timestep-conditioned",
    );
    expect(() => new Ltx2VideoAutoencoderKL(config({ decoderInjectNoise: [false, true] }))).toThrow(
      "noise injection",
    );
    expect(() => new Ltx2VideoAutoencoderKL(config({ patchSizeT: 2 }))).toThrow(
      "temporal VAE patch sizes",
    );
  });
});
