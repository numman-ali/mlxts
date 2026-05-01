import { describe, expect, test } from "bun:test";
import { MxArray, mxEval, retainArray, zeros } from "@mlxts/core";
import { Ltx2AudioAutoencoderKL } from "./autoencoder-ltx2-audio";
import {
  Ltx2AudioAttnBlock,
  Ltx2AudioCausalConv2d,
  Ltx2AudioPixelNorm,
  Ltx2AudioUpsample,
} from "./autoencoder-ltx2-audio-blocks";
import type { Ltx2AudioAutoencoderConfig } from "./config";
import { decodeLtx2AudioLatents } from "./decoding";

function expectCloseList(actual: ArrayLike<number>, expected: readonly number[]): void {
  expect(actual.length).toBe(expected.length);
  for (let index = 0; index < expected.length; index += 1) {
    expect(actual[index]).toBeCloseTo(expected[index] ?? Number.NaN, 6);
  }
}

function tinyConfig(
  overrides: Partial<Ltx2AudioAutoencoderConfig> = {},
): Ltx2AudioAutoencoderConfig {
  return {
    baseChannels: 1,
    outputChannels: 1,
    chMult: [1],
    numResBlocks: 0,
    attnResolutions: null,
    inChannels: 1,
    resolution: 4,
    latentChannels: 1,
    normType: "pixel",
    causalityAxis: "height",
    dropout: 0,
    midBlockAddAttention: false,
    sampleRate: 16000,
    melHopLength: 160,
    isCausal: true,
    melBins: 4,
    melCompressionRatio: 4,
    temporalCompressionRatio: 4,
    packedFeatureSize: 1,
    doubleZ: true,
    rawConfig: {},
    ...overrides,
  };
}

class EchoAudioDecoder {
  readonly latentStatSize = 4;
  readonly latentsMean = [10, 20, 30, 40];
  readonly latentsStd = [2, 2, 2, 2];

  decodeRaw(latents: MxArray): MxArray {
    return retainArray(latents);
  }
}

describe("LTX-2 audio VAE", () => {
  test("causal Conv2d pads only prior time positions on the height axis", () => {
    using conv = new Ltx2AudioCausalConv2d(1, 1, [3, 1], 1, 1, 1, false, "height");
    conv.conv.weight.free();
    conv.conv.weight = MxArray.fromData([1, 1, 1], [1, 3, 1, 1]);
    using input = MxArray.fromData([1, 2], [1, 2, 1, 1]);
    using output = conv.forward(input);

    mxEval(output);
    expect(output.shape).toEqual([1, 2, 1, 1]);
    expectCloseList(output.toTypedArray(), [1, 3]);
  });

  test("pixel norm normalizes across channel-last features", () => {
    using norm = new Ltx2AudioPixelNorm(0);
    using input = MxArray.fromData([3, 4], [1, 1, 1, 2]);
    using output = norm.forward(input);

    mxEval(output);
    const scale = Math.sqrt((9 + 16) / 2);
    expectCloseList(output.toTypedArray(), [3 / scale, 4 / scale]);
  });

  test("attention and upsample blocks preserve Diffusers audio spectrogram geometry", () => {
    using attn = new Ltx2AudioAttnBlock(2);
    using attnInput = zeros([1, 2, 2, 2]);
    using attnOutput = attn.forward(attnInput);
    mxEval(attnOutput);
    expect(attnOutput.shape).toEqual([1, 2, 2, 2]);

    using badChannels = zeros([1, 1, 1, 1]);
    expect(() => attn.forward(badChannels)).toThrow("expected 2 channels");

    using heightUpsample = new Ltx2AudioUpsample(1, "height");
    using widthUpsample = new Ltx2AudioUpsample(1, "width");
    using symmetricUpsample = new Ltx2AudioUpsample(1, null);
    using latent = zeros([1, 2, 2, 1]);
    using heightOutput = heightUpsample.forward(latent);
    using widthOutput = widthUpsample.forward(latent);
    using symmetricOutput = symmetricUpsample.forward(latent);

    mxEval(heightOutput, widthOutput, symmetricOutput);
    expect(heightOutput.shape).toEqual([1, 3, 4, 1]);
    expect(widthOutput.shape).toEqual([1, 4, 3, 1]);
    expect(symmetricOutput.shape).toEqual([1, 4, 4, 1]);
  });

  test("decodes BCLM latent spectrograms to causal mel spectrogram shape", () => {
    using vae = new Ltx2AudioAutoencoderKL(
      tinyConfig({ baseChannels: 2, melBins: 8, packedFeatureSize: 2 }),
    );
    using latents = zeros([1, 1, 2, 2]);
    using decoded = vae.decodeRaw(latents);

    mxEval(decoded);
    expect(decoded.shape).toEqual([1, 1, 5, 8]);
    expect(vae.latentStatSize).toBe(2);
    expect(vae.temporalCompressionRatio).toBe(4);
    expect(vae.melCompressionRatio).toBe(4);
  });

  test("uses decoder causality axis, not is_causal, for causal output cropping", () => {
    using vae = new Ltx2AudioAutoencoderKL(tinyConfig({ isCausal: false }));
    using latents = zeros([1, 1, 2, 1]);
    using decoded = vae.decodeRaw(latents);

    mxEval(decoded);
    expect(decoded.shape).toEqual([1, 1, 5, 4]);
  });

  test("denormalizes packed audio tokens before unpacking and decoding", () => {
    using packed = MxArray.fromData([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 4]);
    using decoded = decodeLtx2AudioLatents(new EchoAudioDecoder(), packed, 2, 2);

    mxEval(decoded);
    expect(decoded.shape).toEqual([1, 2, 2, 2]);
    expectCloseList(decoded.toTypedArray(), [12, 24, 20, 32, 36, 48, 44, 56]);
  });

  test("validates latent channel and packed stats sizes", () => {
    using vae = new Ltx2AudioAutoencoderKL(tinyConfig());
    expect(() => vae.setLatentStats([0, 1], [1, 1])).toThrow("latentsMean length");

    using badLatents = zeros([1, 2, 1, 1]);
    expect(() => vae.decodeRaw(badLatents)).toThrow("expected 1 latent channels");

    using badPacked = zeros([1, 1, 2]);
    expect(() => decodeLtx2AudioLatents(vae, badPacked, 1, 1)).toThrow("packed features");
  });
});
