import { describe, expect, test } from "bun:test";
import { MxArray, mxEval, retainArray } from "@mlxts/core";

import type { Ltx2VocoderConfig } from "./config";
import { decodeLtx2AudioLatents } from "./decoding";
import { Ltx2Vocoder } from "./vocoder-ltx2";

function tinyConfig(overrides: Partial<Ltx2VocoderConfig> = {}): Ltx2VocoderConfig {
  return {
    inChannels: 2,
    hiddenChannels: 4,
    outChannels: 1,
    upsampleKernelSizes: [4],
    upsampleFactors: [2],
    totalUpsampleFactor: 2,
    resnetKernelSizes: [3],
    resnetDilations: [[1]],
    actFn: "leaky_relu",
    leakyReluNegativeSlope: 0.1,
    antialias: false,
    antialiasRatio: 2,
    antialiasKernelSize: 12,
    finalActFn: "tanh",
    finalBias: true,
    outputSamplingRate: 20,
    rawConfig: {},
    ...overrides,
  };
}

describe("LTX-2 vocoder", () => {
  test("maps BCLM mel spectrograms to BCS waveforms", () => {
    using vocoder = new Ltx2Vocoder(tinyConfig());
    const mel = MxArray.fromData([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [1, 1, 3, 2]);
    const waveform = vocoder.forward(mel);
    mxEval(waveform);

    expect(waveform.shape).toEqual([1, 1, 6]);

    mel.free();
    waveform.free();
  });

  test("rejects mel spectrograms whose flattened channel width does not match config", () => {
    using vocoder = new Ltx2Vocoder(tinyConfig());
    const mel = MxArray.fromData([1, 2, 3], [1, 1, 3, 1]);

    expect(() => vocoder.forward(mel)).toThrow("expected 2 channels");

    mel.free();
  });

  test("rejects unsupported activation branches before silent approximation", () => {
    expect(() => new Ltx2Vocoder(tinyConfig({ actFn: "snakebeta" }))).toThrow("leaky_relu");
    expect(() => new Ltx2Vocoder(tinyConfig({ antialias: true }))).toThrow("antialiasing");
  });

  test("decoded audio latents feed the vocoder BCLM boundary", () => {
    const decoder = {
      latentStatSize: 1,
      latentsMean: [0],
      latentsStd: [1],
      decodeRaw(latents: MxArray): MxArray {
        return retainArray(latents);
      },
    };
    using packed = MxArray.fromData([1, 2], [1, 2, 1]);
    using mel = decodeLtx2AudioLatents(decoder, packed, 2, 1);
    using vocoder = new Ltx2Vocoder(tinyConfig({ inChannels: 1 }));
    const waveform = vocoder.forward(mel);
    mxEval(waveform);

    expect(mel.shape).toEqual([1, 1, 2, 1]);
    expect(waveform.shape).toEqual([1, 1, 4]);

    waveform.free();
  });
});
