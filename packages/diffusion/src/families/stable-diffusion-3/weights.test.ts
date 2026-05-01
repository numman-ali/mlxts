import { describe, expect, test } from "bun:test";
import {
  MxArray,
  mxEval,
  type ParameterTree,
  saveSafetensors,
  treeFlatten,
  zeros,
} from "@mlxts/core";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

import { DiffusionMissingWeightsError, DiffusionWeightMismatchError } from "../../errors";
import type { DiffusionSnapshotComponent } from "../../pretrained/snapshot-manifest";
import { loadDiffusionSnapshotManifest } from "../../pretrained/snapshot-manifest";
import { StableDiffusion3AutoencoderKL } from "./autoencoder";
import type {
  StableDiffusion3AutoencoderConfig,
  StableDiffusion3TransformerConfig,
} from "./config";
import { StableDiffusion3Transformer2DModel } from "./transformer";
import { isIgnoredStableDiffusion3TransformerWeight } from "./weight-mapping";
import {
  loadStableDiffusion3AutoencoderFromSnapshot,
  loadStableDiffusion3AutoencoderWeights,
  loadStableDiffusion3TransformerFromSnapshot,
  loadStableDiffusion3TransformerWeights,
  stableDiffusion3AutoencoderWeightPath,
  stableDiffusion3TransformerWeightPath,
  transformStableDiffusion3AutoencoderWeight,
  transformStableDiffusion3TransformerWeight,
} from "./weights";

async function withTempDirectory<T>(
  prefix: string,
  run: (directory: string) => Promise<T>,
): Promise<T> {
  const directory = mkdtempSync(join(tmpdir(), prefix));
  try {
    return await run(directory);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
}

function writeJson(path: string, payload: unknown): void {
  writeFileSync(path, JSON.stringify(payload));
}

function writeComponentFiles(
  snapshot: string,
  component: string,
  metadataFiles: readonly string[],
  weightFiles: readonly string[] = [],
): void {
  const directory = join(snapshot, component);
  mkdirSync(directory, { recursive: true });
  for (const file of metadataFiles) {
    writeJson(join(directory, file), {});
  }
  for (const file of weightFiles) {
    writeFileSync(join(directory, file), "");
  }
}

function tinyTransformerConfig(
  options: {
    dualAttentionLayers?: readonly number[];
    qkNorm?: StableDiffusion3TransformerConfig["qkNorm"];
  } = {},
): StableDiffusion3TransformerConfig {
  return {
    sampleSize: 4,
    patchSize: 2,
    inChannels: 4,
    outChannels: 4,
    numLayers: 2,
    attentionHeadDim: 4,
    numAttentionHeads: 2,
    hiddenSize: 8,
    jointAttentionDim: 6,
    captionProjectionDim: 8,
    pooledProjectionDim: 5,
    posEmbedMaxSize: 4,
    dualAttentionLayers: options.dualAttentionLayers ?? [0],
    qkNorm: "qkNorm" in options ? options.qkNorm : "rms_norm",
    rawConfig: {},
  };
}

function tinyBaseTransformerConfig(): StableDiffusion3TransformerConfig {
  return tinyTransformerConfig({ dualAttentionLayers: [], qkNorm: null });
}

function diffusersTinyTransformerConfig(
  config: StableDiffusion3TransformerConfig = tinyTransformerConfig(),
): Record<string, unknown> {
  return {
    _class_name: "SD3Transformer2DModel",
    attention_head_dim: config.attentionHeadDim,
    caption_projection_dim: config.captionProjectionDim,
    dual_attention_layers: [...config.dualAttentionLayers],
    in_channels: config.inChannels,
    joint_attention_dim: config.jointAttentionDim,
    num_attention_heads: config.numAttentionHeads,
    num_layers: config.numLayers,
    out_channels: config.outChannels,
    patch_size: config.patchSize,
    pooled_projection_dim: config.pooledProjectionDim,
    pos_embed_max_size: config.posEmbedMaxSize,
    qk_norm: config.qkNorm,
    sample_size: config.sampleSize,
  };
}

function tinyAutoencoderConfig(): StableDiffusion3AutoencoderConfig {
  return {
    inChannels: 3,
    outChannels: 3,
    latentChannels: 4,
    latentChannelsOut: 8,
    useQuantConv: true,
    usePostQuantConv: true,
    blockOutChannels: [4],
    layersPerBlock: 1,
    normNumGroups: 1,
    scalingFactor: 1.5035,
    shiftFactor: 0.0609,
    sampleSize: 32,
    vaeScaleFactor: 1,
    downBlockTypes: ["DownEncoderBlock2D"],
    upBlockTypes: ["UpDecoderBlock2D"],
    forceUpcast: true,
    rawConfig: {},
  };
}

function diffusersTinyAutoencoderConfig(): Record<string, unknown> {
  return {
    _class_name: "AutoencoderKL",
    act_fn: "silu",
    block_out_channels: [4],
    down_block_types: ["DownEncoderBlock2D"],
    force_upcast: true,
    in_channels: 3,
    latent_channels: 4,
    latents_mean: null,
    latents_std: null,
    layers_per_block: 1,
    norm_num_groups: 1,
    out_channels: 3,
    sample_size: 32,
    scaling_factor: 1.5035,
    shift_factor: 0.0609,
    up_block_types: ["UpDecoderBlock2D"],
    use_post_quant_conv: true,
    use_quant_conv: true,
  };
}

function writeStableDiffusion3Snapshot(snapshot: string): void {
  writeJson(join(snapshot, "model_index.json"), {
    _class_name: "StableDiffusion3Pipeline",
    scheduler: ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    text_encoder: ["transformers", "CLIPTextModelWithProjection"],
    text_encoder_2: ["transformers", "CLIPTextModelWithProjection"],
    text_encoder_3: ["transformers", "T5EncoderModel"],
    tokenizer: ["transformers", "CLIPTokenizer"],
    tokenizer_2: ["transformers", "CLIPTokenizer"],
    tokenizer_3: ["transformers", "T5TokenizerFast"],
    transformer: ["diffusers", "SD3Transformer2DModel"],
    vae: ["diffusers", "AutoencoderKL"],
  });
  writeComponentFiles(
    snapshot,
    "transformer",
    ["config.json"],
    ["diffusion_pytorch_model.safetensors"],
  );
  writeJson(join(snapshot, "transformer", "config.json"), diffusersTinyTransformerConfig());
  writeComponentFiles(snapshot, "vae", ["config.json"], ["diffusion_pytorch_model.safetensors"]);
  writeJson(join(snapshot, "vae", "config.json"), diffusersTinyAutoencoderConfig());
  writeComponentFiles(snapshot, "scheduler", ["scheduler_config.json"]);
  writeJson(join(snapshot, "scheduler", "scheduler_config.json"), {
    _class_name: "FlowMatchEulerDiscreteScheduler",
  });
  writeComponentFiles(snapshot, "text_encoder", ["config.json"], ["model.safetensors"]);
  writeComponentFiles(snapshot, "text_encoder_2", ["config.json"], ["model.safetensors"]);
  writeComponentFiles(snapshot, "text_encoder_3", ["config.json"], ["model.safetensors"]);
  writeComponentFiles(snapshot, "tokenizer", ["vocab.json", "merges.txt"]);
  writeComponentFiles(snapshot, "tokenizer_2", ["vocab.json", "merges.txt"]);
  writeComponentFiles(snapshot, "tokenizer_3", ["spiece.model"]);
}

function vaeComponent(weightPaths: readonly string[]): DiffusionSnapshotComponent {
  return {
    name: "vae",
    role: "vae",
    library: "diffusers",
    className: "AutoencoderKL",
    enabled: true,
    optional: false,
    subfolder: "vae",
    metadataPaths: [],
    weightPaths,
  };
}

function transformerComponent(weightPaths: readonly string[]): DiffusionSnapshotComponent {
  return {
    name: "transformer",
    role: "backbone",
    library: "diffusers",
    className: "SD3Transformer2DModel",
    enabled: true,
    optional: false,
    subfolder: "transformer",
    metadataPaths: [],
    weightPaths,
  };
}

function checkpointNameForAutoencoderPath(path: string): string {
  return path
    .replaceAll("postQuantConv", "post_quant_conv")
    .replaceAll("quantConv", "quant_conv")
    .replaceAll("downBlocks", "down_blocks")
    .replaceAll("upBlocks", "up_blocks")
    .replaceAll("convNormOut", "conv_norm_out")
    .replaceAll("convIn", "conv_in")
    .replaceAll("convOut", "conv_out")
    .replaceAll("convShortcut", "conv_shortcut")
    .replaceAll("downsample.conv", "downsamplers.0.conv")
    .replaceAll("upsample.conv", "upsamplers.0.conv")
    .replaceAll("midBlock.resnetIn", "mid_block.resnets.0")
    .replaceAll("midBlock.attention", "mid_block.attentions.0")
    .replaceAll("midBlock.resnetOut", "mid_block.resnets.1")
    .replaceAll("groupNorm", "group_norm")
    .replaceAll("queryProjection", "to_q")
    .replaceAll("keyProjection", "to_k")
    .replaceAll("valueProjection", "to_v")
    .replaceAll("outputProjection", "to_out.0");
}

function checkpointNameForTransformerPath(path: string): string {
  return path
    .replaceAll("posEmbed.projection", "pos_embed.proj")
    .replaceAll("timeTextEmbed.timestepLinear1", "time_text_embed.timestep_embedder.linear_1")
    .replaceAll("timeTextEmbed.timestepLinear2", "time_text_embed.timestep_embedder.linear_2")
    .replaceAll("timeTextEmbed.textEmbedder.linear1", "time_text_embed.text_embedder.linear_1")
    .replaceAll("timeTextEmbed.textEmbedder.linear2", "time_text_embed.text_embedder.linear_2")
    .replaceAll("contextEmbedder", "context_embedder")
    .replaceAll("transformerBlocks", "transformer_blocks")
    .replaceAll("norm1Context", "norm1_context")
    .replaceAll("attention2.", "attn2.")
    .replaceAll("attention.", "attn.")
    .replaceAll("addQProj", "add_q_proj")
    .replaceAll("addKProj", "add_k_proj")
    .replaceAll("addVProj", "add_v_proj")
    .replaceAll("toAddOut", "to_add_out")
    .replaceAll("toOut", "to_out.0")
    .replaceAll("toQ", "to_q")
    .replaceAll("toK", "to_k")
    .replaceAll("toV", "to_v")
    .replaceAll("normAddedQ", "norm_added_q")
    .replaceAll("normAddedK", "norm_added_k")
    .replaceAll("normQ", "norm_q")
    .replaceAll("normK", "norm_k")
    .replaceAll("ffContext.linear1", "ff_context.net.0.proj")
    .replaceAll("ffContext.linear2", "ff_context.net.2")
    .replaceAll("ff.linear1", "ff.net.0.proj")
    .replaceAll("ff.linear2", "ff.net.2")
    .replaceAll("normOut.linear", "norm_out.linear")
    .replaceAll("projOut", "proj_out");
}

function checkpointShapeForConv2dInternal(tensor: MxArray): number[] {
  const [outputChannels, kernelHeight, kernelWidth, inputChannels] = tensor.shape;
  if (
    outputChannels === undefined ||
    kernelHeight === undefined ||
    kernelWidth === undefined ||
    inputChannels === undefined
  ) {
    throw new Error("checkpointShapeForConv2dInternal: expected a rank-4 tensor.");
  }
  return [outputChannels, inputChannels, kernelHeight, kernelWidth];
}

function checkpointShapeForAutoencoderPath(path: string, tensor: MxArray): number[] {
  if (path.endsWith(".weight") && tensor.shape.length === 4) {
    return checkpointShapeForConv2dInternal(tensor);
  }
  return [...tensor.shape];
}

function checkpointShapeForTransformerPath(path: string, tensor: MxArray): number[] {
  if (path === "posEmbed.projection.weight") {
    return checkpointShapeForConv2dInternal(tensor);
  }
  return [...tensor.shape];
}

function checkpointTensorsForParameters(
  parameters: ParameterTree,
  nameForPath: (path: string) => string,
  shapeForPath: (path: string, tensor: MxArray) => number[],
  options: { omitPath?: string; extra?: Record<string, MxArray>; mismatchPath?: string } = {},
): Record<string, MxArray> {
  const tensors: Record<string, MxArray> = {};
  for (const [path, parameter] of treeFlatten(parameters)) {
    const internalPath = path.join(".");
    if (internalPath === options.omitPath) {
      continue;
    }
    const shape =
      internalPath === options.mismatchPath ? [1] : shapeForPath(internalPath, parameter);
    tensors[nameForPath(internalPath)] = zeros(shape);
  }
  return { ...tensors, ...(options.extra ?? {}) };
}

function freeTensors(tensors: Record<string, MxArray>): void {
  for (const tensor of Object.values(tensors)) {
    tensor.free();
  }
}

async function writeCheckpoint(path: string, tensors: Record<string, MxArray>): Promise<void> {
  try {
    await saveSafetensors(tensors, path);
  } finally {
    freeTensors(tensors);
  }
}

function expectTensorValues(actual: ArrayLike<number>, expected: readonly number[]): void {
  expect(actual.length).toBe(expected.length);
  for (let index = 0; index < expected.length; index += 1) {
    expect(actual[index]).toBe(expected[index]);
  }
}

describe("Stable Diffusion 3 VAE weight mapping", () => {
  test("maps Diffusers VAE names onto the package parameter tree", () => {
    expect(stableDiffusion3AutoencoderWeightPath("encoder.conv_in.weight")).toBe(
      "encoder.convIn.weight",
    );
    expect(stableDiffusion3AutoencoderWeightPath("encoder.mid_block.attentions.0.to_q.bias")).toBe(
      "encoder.midBlock.attention.queryProjection.bias",
    );
    expect(
      stableDiffusion3AutoencoderWeightPath("decoder.up_blocks.0.upsamplers.0.conv.weight"),
    ).toBe("decoder.upBlocks.0.upsample.conv.weight");
    expect(stableDiffusion3AutoencoderWeightPath("post_quant_conv.bias")).toBe(
      "postQuantConv.bias",
    );
    expect(stableDiffusion3AutoencoderWeightPath("")).toBeNull();
  });

  test("transposes Diffusers VAE Conv2d weights into MLX layout", () => {
    using checkpoint = MxArray.fromData([1, 2, 3, 4], [1, 2, 1, 2]);
    using transformed = transformStableDiffusion3AutoencoderWeight(
      "encoder.conv_in.weight",
      "encoder.convIn.weight",
      checkpoint,
    );

    mxEval(transformed);
    expect(transformed.shape).toEqual([1, 1, 2, 2]);
    expectTensorValues(transformed.toTypedArray(), [1, 3, 2, 4]);
  });

  test("loads generated VAE weights and preserves SD3 shift metadata", async () => {
    await withTempDirectory("mlxts-sd3-vae-weights-", async (directory) => {
      using source = new StableDiffusion3AutoencoderKL(tinyAutoencoderConfig());
      const checkpointPath = join(directory, "diffusion_pytorch_model.safetensors");
      await writeCheckpoint(
        checkpointPath,
        checkpointTensorsForParameters(
          source.parameters(),
          checkpointNameForAutoencoderPath,
          checkpointShapeForAutoencoderPath,
        ),
      );
      using target = new StableDiffusion3AutoencoderKL(tinyAutoencoderConfig());

      const result = await loadStableDiffusion3AutoencoderWeights(
        target,
        vaeComponent([checkpointPath]),
      );

      expect(result.shardCount).toBe(1);
      expect(result.unexpectedWeights).toEqual([]);
      expect(target.shiftFactor).toBe(0.0609);
      expect(result.assignedPaths).toEqual(
        treeFlatten(target.parameters())
          .map(([path]) => path.join("."))
          .toSorted((left, right) => left.localeCompare(right)),
      );
    });
  });

  test("loads a VAE from an inspected SD3 snapshot manifest", async () => {
    await withTempDirectory("mlxts-sd3-vae-manifest-", async (directory) => {
      writeStableDiffusion3Snapshot(directory);
      using source = new StableDiffusion3AutoencoderKL(tinyAutoencoderConfig());
      await writeCheckpoint(
        join(directory, "vae", "diffusion_pytorch_model.safetensors"),
        checkpointTensorsForParameters(
          source.parameters(),
          checkpointNameForAutoencoderPath,
          checkpointShapeForAutoencoderPath,
        ),
      );

      const manifest = await loadDiffusionSnapshotManifest(directory);
      using model = await loadStableDiffusion3AutoencoderFromSnapshot(manifest);

      expect(model.latentChannels).toBe(4);
      expect(model.shiftFactor).toBe(0.0609);
    });
  });

  test("rejects missing, mismatched, and strict unexpected VAE weights", async () => {
    await withTempDirectory("mlxts-sd3-vae-invalid-", async (directory) => {
      using source = new StableDiffusion3AutoencoderKL(tinyAutoencoderConfig());
      const missingPath = join(directory, "missing.safetensors");
      await writeCheckpoint(
        missingPath,
        checkpointTensorsForParameters(
          source.parameters(),
          checkpointNameForAutoencoderPath,
          checkpointShapeForAutoencoderPath,
          { omitPath: "decoder.convOut.bias" },
        ),
      );
      using missingTarget = new StableDiffusion3AutoencoderKL(tinyAutoencoderConfig());
      await expect(
        loadStableDiffusion3AutoencoderWeights(missingTarget, vaeComponent([missingPath])),
      ).rejects.toThrow(DiffusionMissingWeightsError);

      const mismatchPath = join(directory, "mismatch.safetensors");
      await writeCheckpoint(
        mismatchPath,
        checkpointTensorsForParameters(
          source.parameters(),
          checkpointNameForAutoencoderPath,
          checkpointShapeForAutoencoderPath,
          { mismatchPath: "encoder.convIn.weight" },
        ),
      );
      using mismatchTarget = new StableDiffusion3AutoencoderKL(tinyAutoencoderConfig());
      await expect(
        loadStableDiffusion3AutoencoderWeights(mismatchTarget, vaeComponent([mismatchPath])),
      ).rejects.toThrow(DiffusionWeightMismatchError);

      const unexpectedPath = join(directory, "unexpected.safetensors");
      await writeCheckpoint(
        unexpectedPath,
        checkpointTensorsForParameters(
          source.parameters(),
          checkpointNameForAutoencoderPath,
          checkpointShapeForAutoencoderPath,
          { extra: { "decoder.extra.weight": zeros([1]) } },
        ),
      );
      using unexpectedTarget = new StableDiffusion3AutoencoderKL(tinyAutoencoderConfig());
      await expect(
        loadStableDiffusion3AutoencoderWeights(unexpectedTarget, vaeComponent([unexpectedPath]), {
          strictUnexpectedWeights: true,
        }),
      ).rejects.toThrow("unexpected unmapped weights");
    });
  });
});

describe("Stable Diffusion 3 transformer weight mapping", () => {
  test("maps Diffusers transformer names onto the package parameter tree", () => {
    expect(stableDiffusion3TransformerWeightPath("pos_embed.proj.weight")).toBe(
      "posEmbed.projection.weight",
    );
    expect(
      stableDiffusion3TransformerWeightPath("time_text_embed.timestep_embedder.linear_1.bias"),
    ).toBe("timeTextEmbed.timestepLinear1.bias");
    expect(
      stableDiffusion3TransformerWeightPath("time_text_embed.text_embedder.linear_2.weight"),
    ).toBe("timeTextEmbed.textEmbedder.linear2.weight");
    expect(
      stableDiffusion3TransformerWeightPath("transformer_blocks.0.norm1_context.linear.bias"),
    ).toBe("transformerBlocks.0.norm1Context.linear.bias");
    expect(
      stableDiffusion3TransformerWeightPath("transformer_blocks.0.attn.norm_added_k.weight"),
    ).toBe("transformerBlocks.0.attention.normAddedK.weight");
    expect(stableDiffusion3TransformerWeightPath("transformer_blocks.0.attn2.norm_q.weight")).toBe(
      "transformerBlocks.0.attention2.normQ.weight",
    );
    expect(
      stableDiffusion3TransformerWeightPath("transformer_blocks.0.ff_context.net.2.bias"),
    ).toBe("transformerBlocks.0.ffContext.linear2.bias");
    expect(stableDiffusion3TransformerWeightPath("norm_out.linear.weight")).toBe(
      "normOut.linear.weight",
    );
    expect(stableDiffusion3TransformerWeightPath("proj_out.bias")).toBe("projOut.bias");
    expect(stableDiffusion3TransformerWeightPath("pos_embed.pos_embed")).toBeNull();
    expect(isIgnoredStableDiffusion3TransformerWeight("pos_embed.pos_embed")).toBe(true);
  });

  test("transposes patch projection weights and leaves linear weights untouched", () => {
    using checkpoint = MxArray.fromData([1, 2, 3, 4], [1, 2, 1, 2]);
    using transformed = transformStableDiffusion3TransformerWeight(
      "pos_embed.proj.weight",
      "posEmbed.projection.weight",
      checkpoint,
    );
    using linear = zeros([2, 2]);

    const retained = transformStableDiffusion3TransformerWeight(
      "proj_out.weight",
      "projOut.weight",
      linear,
    );

    mxEval(transformed);
    expect(transformed.shape).toEqual([1, 1, 2, 2]);
    expectTensorValues(transformed.toTypedArray(), [1, 3, 2, 4]);
    expect(retained).toBe(linear);
  });

  test("loads generated SD3.5 transformer weights and ignores fixed positional buffers", async () => {
    await withTempDirectory("mlxts-sd3-transformer-weights-", async (directory) => {
      using source = new StableDiffusion3Transformer2DModel(tinyTransformerConfig());
      const checkpointPath = join(directory, "diffusion_pytorch_model.safetensors");
      await writeCheckpoint(
        checkpointPath,
        checkpointTensorsForParameters(
          source.parameters(),
          checkpointNameForTransformerPath,
          checkpointShapeForTransformerPath,
          { extra: { "pos_embed.pos_embed": zeros([1, 4, 8]) } },
        ),
      );
      using target = new StableDiffusion3Transformer2DModel(tinyTransformerConfig());

      const result = await loadStableDiffusion3TransformerWeights(
        target,
        transformerComponent([checkpointPath]),
        { strictUnexpectedWeights: true },
      );

      expect(result.shardCount).toBe(1);
      expect(result.unexpectedWeights).toEqual([]);
      expect(result.assignedPaths).toEqual(
        treeFlatten(target.parameters())
          .map(([path]) => path.join("."))
          .toSorted((left, right) => left.localeCompare(right)),
      );
      expect(target.transformerBlocks[0]?.attention2?.normQ?.weight.shape).toEqual([4]);
    });
  });

  test("loads base SD3 transformer weights without qk norms or dual attention", async () => {
    await withTempDirectory("mlxts-sd3-base-transformer-weights-", async (directory) => {
      const config = tinyBaseTransformerConfig();
      using source = new StableDiffusion3Transformer2DModel(config);
      const checkpointPath = join(directory, "diffusion_pytorch_model.safetensors");
      await writeCheckpoint(
        checkpointPath,
        checkpointTensorsForParameters(
          source.parameters(),
          checkpointNameForTransformerPath,
          checkpointShapeForTransformerPath,
          { extra: { "pos_embed.pos_embed": zeros([1, 4, 8]) } },
        ),
      );
      using target = new StableDiffusion3Transformer2DModel(config);

      const result = await loadStableDiffusion3TransformerWeights(
        target,
        transformerComponent([checkpointPath]),
        { strictUnexpectedWeights: true },
      );

      expect(result.unexpectedWeights).toEqual([]);
      expect(result.assignedPaths).not.toContain("transformerBlocks.0.attention.normQ.weight");
      expect(result.assignedPaths).not.toContain("transformerBlocks.0.attention2.toQ.weight");
      expect(target.transformerBlocks[0]?.attention.normQ).toBeNull();
      expect(target.transformerBlocks[0]?.attention2).toBeNull();
    });
  });

  test("loads a transformer from an inspected SD3 snapshot manifest", async () => {
    await withTempDirectory("mlxts-sd3-transformer-manifest-", async (directory) => {
      writeStableDiffusion3Snapshot(directory);
      using source = new StableDiffusion3Transformer2DModel(tinyTransformerConfig());
      await writeCheckpoint(
        join(directory, "transformer", "diffusion_pytorch_model.safetensors"),
        checkpointTensorsForParameters(
          source.parameters(),
          checkpointNameForTransformerPath,
          checkpointShapeForTransformerPath,
        ),
      );

      const manifest = await loadDiffusionSnapshotManifest(directory);
      using model = await loadStableDiffusion3TransformerFromSnapshot(manifest);

      expect(model.config.hiddenSize).toBe(8);
      expect(model.transformerBlocks[1]?.attention.toAddOut).toBeNull();
    });
  });

  test("rejects missing, mismatched, and strict unexpected transformer weights", async () => {
    await withTempDirectory("mlxts-sd3-transformer-invalid-", async (directory) => {
      using source = new StableDiffusion3Transformer2DModel(tinyTransformerConfig());
      const missingPath = join(directory, "missing.safetensors");
      await writeCheckpoint(
        missingPath,
        checkpointTensorsForParameters(
          source.parameters(),
          checkpointNameForTransformerPath,
          checkpointShapeForTransformerPath,
          { omitPath: "projOut.bias" },
        ),
      );
      using missingTarget = new StableDiffusion3Transformer2DModel(tinyTransformerConfig());
      await expect(
        loadStableDiffusion3TransformerWeights(missingTarget, transformerComponent([missingPath])),
      ).rejects.toThrow(DiffusionMissingWeightsError);

      const mismatchPath = join(directory, "mismatch.safetensors");
      await writeCheckpoint(
        mismatchPath,
        checkpointTensorsForParameters(
          source.parameters(),
          checkpointNameForTransformerPath,
          checkpointShapeForTransformerPath,
          { mismatchPath: "posEmbed.projection.weight" },
        ),
      );
      using mismatchTarget = new StableDiffusion3Transformer2DModel(tinyTransformerConfig());
      await expect(
        loadStableDiffusion3TransformerWeights(
          mismatchTarget,
          transformerComponent([mismatchPath]),
        ),
      ).rejects.toThrow(DiffusionWeightMismatchError);

      const unexpectedPath = join(directory, "unexpected.safetensors");
      await writeCheckpoint(
        unexpectedPath,
        checkpointTensorsForParameters(
          source.parameters(),
          checkpointNameForTransformerPath,
          checkpointShapeForTransformerPath,
          { extra: { "transformer.extra.weight": zeros([1]) } },
        ),
      );
      using unexpectedTarget = new StableDiffusion3Transformer2DModel(tinyTransformerConfig());
      await expect(
        loadStableDiffusion3TransformerWeights(
          unexpectedTarget,
          transformerComponent([unexpectedPath]),
          { strictUnexpectedWeights: true },
        ),
      ).rejects.toThrow("unexpected unmapped weights");
    });
  });
});
