import { describe, expect, test } from "bun:test";
import { add, MxArray, mxEval, zeros } from "@mlxts/core";
import {
  type LtxVideoLatentNormalizer,
  upsampleLtxVideoLatents,
  upsamplePackedLtxVideoLatents,
} from "./latent-upsample";
import {
  type LtxVideoLatentUpsamplerConfig,
  LtxVideoLatentUpsamplerModel,
  parseLtxVideoLatentUpsamplerConfig,
  pixelShuffleLtxLatents1d,
  pixelShuffleLtxLatents2d,
  pixelShuffleLtxLatents3d,
} from "./latent-upsampler";
import { packLtxVideoLatents } from "./latents";

function expectCloseList(actual: ArrayLike<number>, expected: readonly number[]): void {
  expect(actual.length).toBe(expected.length);
  for (let index = 0; index < expected.length; index += 1) {
    expect(actual[index]).toBeCloseTo(expected[index] ?? Number.NaN, 6);
  }
}

function config(
  overrides: Partial<LtxVideoLatentUpsamplerConfig> = {},
): LtxVideoLatentUpsamplerConfig {
  return {
    inChannels: 2,
    midChannels: 32,
    numBlocksPerStage: 0,
    dims: 3,
    spatialUpsample: true,
    temporalUpsample: false,
    rawConfig: {},
    ...overrides,
  };
}

class RecordingUpsampler extends LtxVideoLatentUpsamplerModel {
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

const normalizer: LtxVideoLatentNormalizer = {
  latentChannels: 2,
  latentsMean: [10, 20],
  latentsStd: [2, 4],
  scalingFactor: 4,
};

describe("LTX video latent upsampler", () => {
  test("parses the Diffusers sidecar config and rejects LTX-2-specific configs", () => {
    const parsed = parseLtxVideoLatentUpsamplerConfig({
      _class_name: "LTXLatentUpsamplerModel",
      _diffusers_version: "0.35.0.dev0",
      dims: 3,
      in_channels: 128,
      mid_channels: 512,
      num_blocks_per_stage: 4,
      spatial_upsample: true,
      temporal_upsample: false,
    });

    expect(parsed).toMatchObject({
      dims: 3,
      inChannels: 128,
      midChannels: 512,
      numBlocksPerStage: 4,
      spatialUpsample: true,
      temporalUpsample: false,
    });
    expect(() =>
      parseLtxVideoLatentUpsamplerConfig({
        _class_name: "LTX2LatentUpsamplerModel",
        dims: 3,
      }),
    ).toThrow("class_name");
    expect(() =>
      parseLtxVideoLatentUpsamplerConfig({
        _class_name: "LTXLatentUpsamplerModel",
        spatial_upsample: false,
        temporal_upsample: false,
      }),
    ).toThrow("one upsample axis");
    expect(() =>
      parseLtxVideoLatentUpsamplerConfig({
        _class_name: "LTXLatentUpsamplerModel",
        dims: 2,
        temporal_upsample: true,
      }),
    ).toThrow("dims=3");
  });

  test("matches Diffusers PixelShuffleND channel order for 1D, 2D, and 3D paths", () => {
    using spatial = MxArray.fromData([1, 2, 3, 4], [1, 1, 1, 4]);
    using spatialOutput = pixelShuffleLtxLatents2d(spatial);
    using temporal = MxArray.fromData([1, 2], [1, 1, 1, 1, 2]);
    using temporalOutput = pixelShuffleLtxLatents1d(temporal);
    using spatiotemporal = MxArray.fromData([1, 2, 3, 4, 5, 6, 7, 8], [1, 1, 1, 1, 8]);
    using spatiotemporalOutput = pixelShuffleLtxLatents3d(spatiotemporal);

    mxEval(spatialOutput, temporalOutput, spatiotemporalOutput);
    expect(spatialOutput.shape).toEqual([1, 2, 2, 1]);
    expectCloseList(spatialOutput.toTypedArray(), [1, 2, 3, 4]);
    expect(temporalOutput.shape).toEqual([1, 2, 1, 1, 1]);
    expectCloseList(temporalOutput.toTypedArray(), [1, 2]);
    expect(spatiotemporalOutput.shape).toEqual([1, 2, 2, 2, 1]);
    expectCloseList(spatiotemporalOutput.toTypedArray(), [1, 2, 3, 4, 5, 6, 7, 8]);
  });

  test("runs spatial-only 3D, temporal-spatial 3D, and 2D forward shapes", () => {
    using spatial3d = new LtxVideoLatentUpsamplerModel(config());
    using temporal3d = new LtxVideoLatentUpsamplerModel(
      config({ temporalUpsample: true, spatialUpsample: true }),
    );
    using spatial2d = new LtxVideoLatentUpsamplerModel(config({ dims: 2 }));
    using volume = zeros([1, 2, 2, 2, 2]);
    using temporalVolume = zeros([1, 2, 2, 1, 1]);

    using spatial3dOutput = spatial3d.forward(volume);
    using temporal3dOutput = temporal3d.forward(temporalVolume);
    using spatial2dOutput = spatial2d.forward(volume);

    mxEval(spatial3dOutput, temporal3dOutput, spatial2dOutput);
    expect(spatial3dOutput.shape).toEqual([1, 2, 2, 4, 4]);
    expect(temporal3dOutput.shape).toEqual([1, 2, 3, 2, 2]);
    expect(spatial2dOutput.shape).toEqual([1, 2, 2, 4, 4]);
  });

  test("runs configured residual stages on 2D and 3D paths", () => {
    using spatial3d = new LtxVideoLatentUpsamplerModel(config({ numBlocksPerStage: 1 }));
    using spatial2d = new LtxVideoLatentUpsamplerModel(config({ dims: 2, numBlocksPerStage: 1 }));
    using volume = zeros([1, 2, 1, 2, 2]);

    using spatial3dOutput = spatial3d.forward(volume);
    using spatial2dOutput = spatial2d.forward(volume);

    mxEval(spatial3dOutput, spatial2dOutput);
    expect(spatial3dOutput.shape).toEqual([1, 2, 1, 4, 4]);
    expect(spatial2dOutput.shape).toEqual([1, 2, 1, 4, 4]);
  });

  test("denormalizes before upsampling and normalizes the returned latent", () => {
    using upsampler = new RecordingUpsampler();
    using latents = MxArray.fromData([1, 2], [1, 2, 1, 1, 1]);
    using output = upsampleLtxVideoLatents(upsampler, normalizer, latents);

    mxEval(output);
    expectCloseList(upsampler.inputs[0] ?? [], [10.5, 22]);
    expectCloseList(output.toTypedArray(), [3, 3]);
  });

  test("unpacks and repacks normalized latent tokens around the upsampler", () => {
    using upsampledVolume = zeros([1, 2, 1, 2, 2]);
    using upsampler = new RecordingUpsampler(upsampledVolume);
    using latents = MxArray.fromData([1, 2], [1, 2, 1, 1, 1]);
    using packed = packLtxVideoLatents(latents);
    using output = upsamplePackedLtxVideoLatents(upsampler, normalizer, packed, 1, 1, 1);

    mxEval(output);
    expect(output.shape).toEqual([1, 4, 2]);
    expectCloseList(upsampler.inputs[0] ?? [], [10.5, 22]);
    expectCloseList(output.toTypedArray(), [-20, -20, -20, -20, -20, -20, -20, -20]);
  });
});
