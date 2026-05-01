import { describe, expect, test } from "bun:test";
import { add, MxArray, mxEval, zeros } from "@mlxts/core";
import {
  type Ltx2VideoLatentNormalizer,
  upsampleLtx2VideoLatents,
  upsamplePackedLtx2VideoLatents,
} from "./latent-upsample-ltx2";
import {
  type Ltx2LatentUpsamplerConfig,
  Ltx2LatentUpsamplerModel,
  parseLtx2LatentUpsamplerConfig,
  pixelShuffleLtx2Latents2d,
} from "./latent-upsampler-ltx2";
import { packLtxVideoLatents } from "./latents";

function expectCloseList(actual: ArrayLike<number>, expected: readonly number[]): void {
  expect(actual.length).toBe(expected.length);
  for (let index = 0; index < expected.length; index += 1) {
    expect(actual[index]).toBeCloseTo(expected[index] ?? Number.NaN, 6);
  }
}

function config(overrides: Partial<Ltx2LatentUpsamplerConfig> = {}): Ltx2LatentUpsamplerConfig {
  return {
    inChannels: 2,
    midChannels: 32,
    numBlocksPerStage: 0,
    dims: 3,
    spatialUpsample: true,
    temporalUpsample: false,
    rationalSpatialScale: 2,
    useRationalResampler: true,
    rawConfig: {},
    ...overrides,
  };
}

class RecordingLtx2Upsampler extends Ltx2LatentUpsamplerModel {
  readonly inputs: number[][] = [];
  #output: MxArray | null;

  constructor(output: MxArray | null = null) {
    super(config());
    this.#output = output;
  }

  override forward(latents: MxArray): MxArray {
    mxEval(latents);
    this.inputs.push(Array.from(latents.toTypedArray()));
    if (this.#output !== null) {
      return add(this.#output, 0);
    }
    return add(latents, 1);
  }
}

const normalizer: Ltx2VideoLatentNormalizer = {
  latentChannels: 2,
  latentsMean: [10, 20],
  latentsStd: [2, 4],
  scalingFactor: 4,
};

describe("LTX-2 latent upsampler", () => {
  test("parses the Diffusers sidecar config and validates LTX-2 fields", () => {
    const parsed = parseLtx2LatentUpsamplerConfig({
      _class_name: "LTX2LatentUpsamplerModel",
      _diffusers_version: "0.37.0.dev0",
      dims: 3,
      in_channels: 128,
      mid_channels: 1024,
      num_blocks_per_stage: 4,
      rational_spatial_scale: 1.5,
      spatial_upsample: true,
      temporal_upsample: false,
      use_rational_resampler: true,
    });

    expect(parsed).toMatchObject({
      dims: 3,
      inChannels: 128,
      midChannels: 1024,
      numBlocksPerStage: 4,
      rationalSpatialScale: 1.5,
      useRationalResampler: true,
    });
    expect(() =>
      parseLtx2LatentUpsamplerConfig({
        _class_name: "LTXLatentUpsamplerModel",
        dims: 3,
      }),
    ).toThrow("class_name");
    expect(() =>
      parseLtx2LatentUpsamplerConfig({
        _class_name: "LTX2LatentUpsamplerModel",
        rational_spatial_scale: 1.25,
      }),
    ).toThrow("rational_spatial_scale");
    expect(() =>
      parseLtx2LatentUpsamplerConfig({
        _class_name: "LTX2LatentUpsamplerModel",
        rational_spatial_scale: 1.5,
        use_rational_resampler: false,
      }),
    ).toThrow("scale 2");
  });

  test("matches Diffusers PixelShuffleND spatial order for configurable factors", () => {
    using spatial = MxArray.fromData([1, 2, 3, 4, 5, 6, 7, 8], [1, 1, 1, 8]);
    using output = pixelShuffleLtx2Latents2d(spatial, 2);

    mxEval(output);
    expect(output.shape).toEqual([1, 2, 2, 2]);
    expectCloseList(output.toTypedArray(), [1, 5, 2, 6, 3, 7, 4, 8]);
  });

  test("runs rational spatial, integer spatial, temporal-spatial, and 2D forward shapes", () => {
    using rational3d = new Ltx2LatentUpsamplerModel(config({ rationalSpatialScale: 1.5 }));
    using integer3d = new Ltx2LatentUpsamplerModel(config({ useRationalResampler: false }));
    using temporal3d = new Ltx2LatentUpsamplerModel(
      config({ temporalUpsample: true, spatialUpsample: true, useRationalResampler: false }),
    );
    using spatial2d = new Ltx2LatentUpsamplerModel(
      config({ dims: 2, useRationalResampler: false }),
    );
    using volume = zeros([1, 2, 2, 4, 4]);
    using temporalVolume = zeros([1, 2, 2, 1, 1]);

    using rationalOutput = rational3d.forward(volume);
    using integerOutput = integer3d.forward(volume);
    using temporalOutput = temporal3d.forward(temporalVolume);
    using spatial2dOutput = spatial2d.forward(volume);

    mxEval(rationalOutput, integerOutput, temporalOutput, spatial2dOutput);
    expect(rationalOutput.shape).toEqual([1, 2, 2, 6, 6]);
    expect(integerOutput.shape).toEqual([1, 2, 2, 8, 8]);
    expect(temporalOutput.shape).toEqual([1, 2, 3, 2, 2]);
    expect(spatial2dOutput.shape).toEqual([1, 2, 2, 8, 8]);
  });

  test("denormalizes supplied normalized latents before upsampling", () => {
    using upsampler = new RecordingLtx2Upsampler();
    using latents = MxArray.fromData([1, 2], [1, 2, 1, 1, 1]);
    using output = upsampleLtx2VideoLatents(upsampler, latents, {
      latentsNormalized: true,
      normalizer,
    });

    mxEval(output);
    expectCloseList(upsampler.inputs[0] ?? [], [10.5, 22]);
    expectCloseList(output.toTypedArray(), [11.5, 23]);
  });

  test("unpacks and repacks packed latent tokens around the LTX-2 upsampler", () => {
    using upsampledVolume = zeros([1, 2, 1, 2, 2]);
    using upsampler = new RecordingLtx2Upsampler(upsampledVolume);
    using latents = MxArray.fromData([1, 2], [1, 2, 1, 1, 1]);
    using packed = packLtxVideoLatents(latents);
    using output = upsamplePackedLtx2VideoLatents(upsampler, packed, 1, 1, 1);

    mxEval(output);
    expect(output.shape).toEqual([1, 4, 2]);
    expectCloseList(upsampler.inputs[0] ?? [], [1, 2]);
    expectCloseList(output.toTypedArray(), [0, 0, 0, 0, 0, 0, 0, 0]);
  });
});
