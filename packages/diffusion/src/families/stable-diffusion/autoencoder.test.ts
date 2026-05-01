import { describe, expect, test } from "bun:test";

import { MxArray, mxEval, random, treeFlatten } from "@mlxts/core";
import {
  StableDiffusionAutoencoderKL,
  StableDiffusionVaeDecoder,
  StableDiffusionVaeEncoder,
} from "./autoencoder";
import {
  StableDiffusionVaeAttentionBlock2d,
  StableDiffusionVaeResnetBlock2d,
} from "./autoencoder-blocks";
import type { StableDiffusionAutoencoderConfig } from "./config";

const tinyConfig: StableDiffusionAutoencoderConfig = {
  inChannels: 3,
  outChannels: 3,
  latentChannels: 2,
  latentChannelsOut: 4,
  useQuantConv: true,
  usePostQuantConv: true,
  blockOutChannels: [4, 8],
  layersPerBlock: 1,
  normNumGroups: 2,
  scalingFactor: 0.18215,
  downBlockTypes: ["DownEncoderBlock2D", "DownEncoderBlock2D"],
  upBlockTypes: ["UpDecoderBlock2D", "UpDecoderBlock2D"],
  forceUpcast: true,
  rawConfig: {},
};

describe("StableDiffusionAutoencoderKL", () => {
  test("constructs encoder and decoder modules with expected NHWC shapes", () => {
    using autoencoder = new StableDiffusionAutoencoderKL(tinyConfig);
    using image = random.normal([1, 8, 8, 3]);
    using moments = autoencoder.encodeMoments(image);
    mxEval(moments);

    expect(moments.shape).toEqual([1, 4, 4, 4]);

    using latent = random.normal([1, 4, 4, 2]);
    using decoded = autoencoder.decode(latent);
    mxEval(decoded);

    expect(decoded.shape).toEqual([1, 8, 8, 3]);
  });

  test("forward reconstructs through posterior mode without applying latent scaling", () => {
    using autoencoder = new StableDiffusionAutoencoderKL(tinyConfig);
    using image = random.normal([1, 8, 8, 3]);
    using reconstructed = autoencoder.forward(image);
    mxEval(reconstructed);

    expect(reconstructed.shape).toEqual([1, 8, 8, 3]);
    expect(autoencoder.scalingFactor).toBe(tinyConfig.scalingFactor);
  });

  test("splits posterior moments on the channel-last axis", () => {
    using autoencoder = new StableDiffusionAutoencoderKL(tinyConfig);
    using moments = MxArray.fromData([1, 2, 3, 4], [1, 1, 1, 4]);
    using posterior = autoencoder.splitMoments(moments);
    using mode = posterior.mode();
    using key = random.key(7);
    using sample = posterior.sample(key);
    mxEval(mode, sample);

    expect(mode.shape).toEqual([1, 1, 1, 2]);
    expect(sample.shape).toEqual([1, 1, 1, 2]);
    expect(mode.toList()).toEqual([[[[1, 2]]]]);
  });

  test("parameter tree exposes module structure and hides config state", () => {
    using autoencoder = new StableDiffusionAutoencoderKL(tinyConfig);
    const paths = treeFlatten(autoencoder.parameters()).map(([path]) => path.join("."));

    expect(paths).toContain("encoder.convIn.weight");
    expect(paths).toContain("encoder.downBlocks.0.resnets.0.conv1.weight");
    expect(paths).toContain("encoder.midBlock.attention.queryProjection.weight");
    expect(paths).toContain("decoder.upBlocks.0.resnets.1.conv2.weight");
    expect(paths).toContain("quantConv.weight");
    expect(paths).toContain("postQuantConv.weight");
    expect(paths.some((path) => path.includes("scalingFactor"))).toBe(false);
    expect(paths.some((path) => path.includes("latentChannels"))).toBe(false);
    expect(paths.some((path) => path.includes("vaeScaleFactor"))).toBe(false);
  });

  test("omits quantization projections when the checkpoint config disables them", () => {
    const noQuantConfig = {
      ...tinyConfig,
      useQuantConv: false,
      usePostQuantConv: false,
    } satisfies StableDiffusionAutoencoderConfig;
    using autoencoder = new StableDiffusionAutoencoderKL(noQuantConfig);
    const paths = treeFlatten(autoencoder.parameters()).map(([path]) => path.join("."));

    expect(paths).not.toContain("quantConv.weight");
    expect(paths).not.toContain("postQuantConv.weight");

    using image = random.normal([1, 8, 8, 3]);
    using moments = autoencoder.encodeMoments(image);
    mxEval(moments);
    expect(moments.shape).toEqual([1, 4, 4, 4]);

    using latent = random.normal([1, 4, 4, 2]);
    using decoded = autoencoder.decode(latent);
    mxEval(decoded);
    expect(decoded.shape).toEqual([1, 8, 8, 3]);
  });

  test("exposes the VAE scale factor derived from downsampling depth", () => {
    using autoencoder = new StableDiffusionAutoencoderKL(tinyConfig);

    expect(autoencoder.vaeScaleFactor).toBe(2);
  });

  test("decoder up blocks use one more resnet layer than encoder blocks", () => {
    using encoder = new StableDiffusionVaeEncoder(tinyConfig);
    using decoder = new StableDiffusionVaeDecoder(tinyConfig);

    expect(encoder.downBlocks[0]?.resnets).toHaveLength(tinyConfig.layersPerBlock);
    expect(decoder.upBlocks[0]?.resnets).toHaveLength(tinyConfig.layersPerBlock + 1);
  });

  test("resnet shortcut parameters exist only when channels change", () => {
    using sameChannels = new StableDiffusionVaeResnetBlock2d(4, 4, 2);
    using changedChannels = new StableDiffusionVaeResnetBlock2d(4, 8, 2);
    const samePaths = treeFlatten(sameChannels.parameters()).map(([path]) => path.join("."));
    const changedPaths = treeFlatten(changedChannels.parameters()).map(([path]) => path.join("."));

    expect(samePaths.some((path) => path.includes("convShortcut"))).toBe(false);
    expect(changedPaths).toContain("convShortcut.weight");
  });

  test("attention block preserves NHWC shape and rejects non-images", () => {
    using attention = new StableDiffusionVaeAttentionBlock2d(4, 2);
    using image = random.normal([1, 4, 4, 4]);
    using output = attention.forward(image);
    mxEval(output);

    expect(output.shape).toEqual([1, 4, 4, 4]);

    using tokens = random.normal([1, 4, 4]);
    expect(() => attention.forward(tokens)).toThrow("rank-4 NHWC");
  });

  test("invalid autoencoder configs fail before module construction drifts", () => {
    const invalidBlockTypes = {
      ...tinyConfig,
      downBlockTypes: ["DownEncoderBlock2D"],
    } satisfies StableDiffusionAutoencoderConfig;
    expect(() => new StableDiffusionAutoencoderKL(invalidBlockTypes)).toThrow("downBlockTypes");

    const invalidLatents = {
      ...tinyConfig,
      latentChannelsOut: 5,
    } satisfies StableDiffusionAutoencoderConfig;
    expect(() => new StableDiffusionAutoencoderKL(invalidLatents)).toThrow("latentChannelsOut");
  });
});
