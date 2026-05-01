import { describe, expect, test } from "bun:test";
import { MxArray, mxEval, random, retainArray, zeros } from "@mlxts/core";

import { FlowMatchEulerScheduler } from "../../schedulers/flow-match-euler";
import {
  decodeZImageLatents,
  denoiseZImageLatents,
  generateZImage,
  type ZImageDenoiser,
} from "./pipeline";

class ZeroZImageDenoiser implements ZImageDenoiser {
  forward(input: Parameters<ZImageDenoiser["forward"]>[0]): MxArray {
    const latent = input.latents[0];
    if (latent === undefined) {
      throw new Error("missing latent");
    }
    const [channels, frames, height, width] = latent.shape;
    return zeros([1, channels ?? 0, frames ?? 0, height ?? 0, width ?? 0], latent.dtype);
  }
}

class ThrowingZImageDenoiser implements ZImageDenoiser {
  forward(): MxArray {
    throw new Error("denoiser failed");
  }
}

describe("Z-Image pipeline", () => {
  test("denoises NCHW latents with a prepared-caption denoiser", () => {
    const scheduler = new FlowMatchEulerScheduler({ shift: 1 });
    const denoiser = new ZeroZImageDenoiser();
    using latents = zeros([1, 4, 2, 2]);
    using caption = MxArray.fromData(
      Array.from({ length: 16 }, (_, index) => index / 16),
      [2, 8],
    );

    using output = denoiseZImageLatents({
      denoiser,
      scheduler,
      initialLatents: latents,
      conditioning: { captionFeatures: [caption] },
      numInferenceSteps: 1,
    });

    mxEval(output);
    expect(output.shape).toEqual([1, 4, 2, 2]);
    expect(output.toList()).toEqual([
      [
        [
          [0, 0],
          [0, 0],
        ],
        [
          [0, 0],
          [0, 0],
        ],
        [
          [0, 0],
          [0, 0],
        ],
        [
          [0, 0],
          [0, 0],
        ],
      ],
    ]);
  });

  test("decodes NCHW latents through an NHWC VAE surface", () => {
    using latents = zeros([1, 2, 2, 2]);
    const vae = {
      scalingFactor: 1,
      shiftFactor: 0,
      latentChannels: 2,
      decode(input: MxArray): MxArray {
        expect(input.shape).toEqual([1, 2, 2, 2]);
        return retainArray(input);
      },
    };

    using decoded = decodeZImageLatents(vae, latents);

    mxEval(decoded);
    expect(decoded.shape).toEqual([1, 2, 2, 2]);
  });

  test("generates an image from prepared caption features", () => {
    const scheduler = new FlowMatchEulerScheduler({ shift: 1 });
    const denoiser = new ZeroZImageDenoiser();
    using caption = MxArray.fromData(
      Array.from({ length: 16 }, (_, index) => index / 16),
      [2, 8],
    );
    using rngKey = random.key(7);
    let stepCount = 0;
    const vae = {
      scalingFactor: 1,
      shiftFactor: 0,
      latentChannels: 4,
      vaeScaleFactor: 8,
      decode(input: MxArray): MxArray {
        expect(input.shape).toEqual([1, 2, 2, 4]);
        return retainArray(input);
      },
    };

    using image = generateZImage({
      denoiser,
      scheduler,
      vae,
      batchSize: 1,
      height: 16,
      width: 16,
      dtype: "float32",
      rngKey,
      conditioning: { captionFeatures: [caption] },
      numInferenceSteps: 1,
      evaluateEachStep: false,
      onStep: () => {
        stepCount += 1;
      },
    });

    mxEval(image);
    expect(image.shape).toEqual([1, 2, 2, 4]);
    expect(stepCount).toBe(1);
  });

  test("rejects invalid Z-Image sampling requests", () => {
    const scheduler = new FlowMatchEulerScheduler({ shift: 1 });
    const denoiser = new ZeroZImageDenoiser();
    using caption = zeros([2, 8]);
    using rank3Latents = zeros([1, 2, 2]);
    using singleLatents = zeros([1, 4, 2, 2]);
    using batchedLatents = zeros([2, 4, 2, 2]);

    expect(() =>
      denoiseZImageLatents({
        denoiser,
        scheduler,
        initialLatents: singleLatents,
        conditioning: { captionFeatures: [caption] },
        numInferenceSteps: 0,
      }),
    ).toThrow("numInferenceSteps");
    expect(() =>
      denoiseZImageLatents({
        denoiser,
        scheduler,
        initialLatents: rank3Latents,
        conditioning: { captionFeatures: [caption] },
        numInferenceSteps: 1,
      }),
    ).toThrow("NCHW");
    expect(() =>
      denoiseZImageLatents({
        denoiser,
        scheduler,
        initialLatents: singleLatents,
        conditioning: { captionFeatures: [] },
        numInferenceSteps: 1,
      }),
    ).toThrow("conditioning.captionFeatures");
    expect(() =>
      denoiseZImageLatents({
        denoiser,
        scheduler,
        initialLatents: batchedLatents,
        conditioning: { captionFeatures: [caption, caption] },
        numInferenceSteps: 1,
      }),
    ).toThrow("batch size 1");
    expect(() =>
      generateZImage({
        denoiser,
        scheduler,
        vae: {
          scalingFactor: 1,
          shiftFactor: 0,
          latentChannels: 4,
          decode(input: MxArray): MxArray {
            return retainArray(input);
          },
        },
        batchSize: 1,
        height: 15,
        width: 16,
        conditioning: { captionFeatures: [caption] },
        numInferenceSteps: 1,
      }),
    ).toThrow("height");
  });

  test("frees retained latents when denoising fails", () => {
    const scheduler = new FlowMatchEulerScheduler({ shift: 1 });
    using caption = zeros([2, 8]);
    using latents = zeros([1, 4, 2, 2]);

    expect(() =>
      denoiseZImageLatents({
        denoiser: new ThrowingZImageDenoiser(),
        scheduler,
        initialLatents: latents,
        conditioning: { captionFeatures: [caption] },
        numInferenceSteps: 1,
      }),
    ).toThrow("denoiser failed");
  });
});
