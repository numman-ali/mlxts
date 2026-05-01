import { describe, expect, test } from "bun:test";
import { type MxArray, type ParameterTree, saveSafetensors, treeFlatten, zeros } from "@mlxts/core";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

import { DiffusionMissingWeightsError, DiffusionWeightMismatchError } from "../../errors";
import {
  type DiffusionSnapshotComponent,
  loadDiffusionSnapshotManifest,
} from "../../pretrained/snapshot-manifest";
import type { Ltx2TextConnectorsConfig } from "./config";
import { Ltx2TextConnectors } from "./connectors-ltx2";
import {
  loadLtx2TextConnectorsFromSnapshot,
  loadLtx2TextConnectorWeights,
  ltx2TextConnectorWeightPath,
} from "./connectors-ltx2-weights";

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

function tinyConfig(overrides: Partial<Ltx2TextConnectorsConfig> = {}): Ltx2TextConnectorsConfig {
  return {
    captionChannels: 4,
    textProjInFactor: 2,
    textEncoderDim: 8,
    videoConnectorNumAttentionHeads: 1,
    videoConnectorAttentionHeadDim: 4,
    videoConnectorHiddenSize: 4,
    videoConnectorNumLayers: 1,
    videoConnectorNumLearnableRegisters: 2,
    videoGatedAttn: false,
    audioConnectorNumAttentionHeads: 1,
    audioConnectorAttentionHeadDim: 4,
    audioConnectorHiddenSize: 4,
    audioConnectorNumLayers: 1,
    audioConnectorNumLearnableRegisters: 2,
    audioGatedAttn: false,
    connectorRopeBaseSeqLen: 4,
    ropeTheta: 4,
    ropeDoublePrecision: true,
    causalTemporalPositioning: false,
    ropeType: "interleaved",
    perModalityProjections: false,
    videoHiddenDim: 4,
    audioHiddenDim: 4,
    projBias: false,
    rawConfig: {},
    ...overrides,
  };
}

function connectorsComponent(weightPaths: readonly string[]): DiffusionSnapshotComponent {
  return {
    name: "connectors",
    role: "connector",
    library: "ltx2",
    className: "LTX2TextConnectors",
    enabled: true,
    optional: false,
    subfolder: "connectors",
    metadataPaths: [],
    weightPaths,
  };
}

const TOP_LEVEL_CHECKPOINT_PATHS = new Map<string, string>([
  ["textProjIn.weight", "text_proj_in.weight"],
  ["textProjIn.bias", "text_proj_in.bias"],
  ["videoTextProjIn.weight", "video_text_proj_in.weight"],
  ["videoTextProjIn.bias", "video_text_proj_in.bias"],
  ["audioTextProjIn.weight", "audio_text_proj_in.weight"],
  ["audioTextProjIn.bias", "audio_text_proj_in.bias"],
]);

const CONNECTOR_CHECKPOINT_PREFIXES = new Map<string, string>([
  ["videoConnector", "video_connector"],
  ["audioConnector", "audio_connector"],
]);

const ATTENTION_CHECKPOINT_PATHS = new Map<string, string>([
  ["normQ", "norm_q"],
  ["normK", "norm_k"],
  ["toQ", "to_q"],
  ["toK", "to_k"],
  ["toV", "to_v"],
  ["toOut", "to_out.0"],
  ["toGateLogits", "to_gate_logits"],
]);

function connectorCheckpointName(path: string): string | null {
  const register = path.match(/^(videoConnector|audioConnector)\.learnableRegisters$/);
  if (register !== null) {
    const [, connector] = register;
    if (connector === undefined) {
      throw new Error("connectorCheckpointName: malformed register path.");
    }
    return `${CONNECTOR_CHECKPOINT_PREFIXES.get(connector)}.learnable_registers`;
  }
  const attention = path.match(
    /^(videoConnector|audioConnector)\.transformerBlocks\.(\d+)\.attn1\.([^.]+)\.(.+)$/,
  );
  if (attention !== null) {
    const [, connector, index, local, leaf] = attention;
    if (
      connector === undefined ||
      index === undefined ||
      local === undefined ||
      leaf === undefined
    ) {
      throw new Error("connectorCheckpointName: malformed attention path.");
    }
    return `${CONNECTOR_CHECKPOINT_PREFIXES.get(
      connector,
    )}.transformer_blocks.${index}.attn1.${ATTENTION_CHECKPOINT_PATHS.get(local)}.${leaf}`;
  }
  const feedForward = path.match(
    /^(videoConnector|audioConnector)\.transformerBlocks\.(\d+)\.ff\.(linear[12])\.(.+)$/,
  );
  if (feedForward !== null) {
    const [, connector, index, layer, leaf] = feedForward;
    if (
      connector === undefined ||
      index === undefined ||
      layer === undefined ||
      leaf === undefined
    ) {
      throw new Error("connectorCheckpointName: malformed feed-forward path.");
    }
    const checkpointLayer = layer === "linear1" ? "net.0.proj" : "net.2";
    return `${CONNECTOR_CHECKPOINT_PREFIXES.get(
      connector,
    )}.transformer_blocks.${index}.ff.${checkpointLayer}.${leaf}`;
  }
  return null;
}

function checkpointNameForLtx2ConnectorPath(path: string): string {
  const mapped = TOP_LEVEL_CHECKPOINT_PATHS.get(path) ?? connectorCheckpointName(path);
  if (mapped !== undefined && mapped !== null) {
    return mapped;
  }
  throw new Error(`checkpointNameForLtx2ConnectorPath: unmapped path ${path}.`);
}

function checkpointTensorsForLtx2Connectors(
  parameters: ParameterTree,
  options: { omitPath?: string; mismatchPath?: string; extra?: Record<string, MxArray> } = {},
): Record<string, MxArray> {
  const tensors: Record<string, MxArray> = {};
  for (const [path, parameter] of treeFlatten(parameters)) {
    const internalPath = path.join(".");
    if (internalPath === options.omitPath) {
      continue;
    }
    const checkpointName = checkpointNameForLtx2ConnectorPath(internalPath);
    const shape = internalPath === options.mismatchPath ? [1] : [...parameter.shape];
    tensors[checkpointName] = zeros(shape);
  }
  return { ...tensors, ...(options.extra ?? {}) };
}

function splitIndexedShards(entries: Record<string, MxArray>): {
  firstShard: Record<string, MxArray>;
  secondShard: Record<string, MxArray>;
  weightMap: Record<string, string>;
} {
  const names = Object.keys(entries).toSorted((left, right) => left.localeCompare(right));
  const midpoint = Math.ceil(names.length / 2);
  const firstShard: Record<string, MxArray> = {};
  const secondShard: Record<string, MxArray> = {};
  const weightMap: Record<string, string> = {};

  for (const [index, name] of names.entries()) {
    const tensor = entries[name];
    if (tensor === undefined) {
      throw new Error(`splitIndexedShards: missing tensor for ${name}.`);
    }
    const shardName =
      index < midpoint
        ? "diffusion_pytorch_model-00001-of-00002.safetensors"
        : "diffusion_pytorch_model-00002-of-00002.safetensors";
    const shard = index < midpoint ? firstShard : secondShard;
    shard[name] = tensor;
    weightMap[name] = shardName;
  }

  return { firstShard, secondShard, weightMap };
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

function writeJson(path: string, payload: unknown): void {
  writeFileSync(path, JSON.stringify(payload));
}

function writeComponent(snapshot: string, name: string, config: Record<string, unknown>): void {
  const directory = join(snapshot, name);
  mkdirSync(directory, { recursive: true });
  writeJson(join(directory, "config.json"), config);
  writeFileSync(join(directory, "diffusion_pytorch_model.safetensors"), "");
}

function ltx2ConnectorsRawConfig(config = tinyConfig()): Record<string, unknown> {
  return {
    _class_name: "LTX2TextConnectors",
    audio_connector_attention_head_dim: config.audioConnectorAttentionHeadDim,
    audio_connector_num_attention_heads: config.audioConnectorNumAttentionHeads,
    audio_connector_num_layers: config.audioConnectorNumLayers,
    audio_connector_num_learnable_registers: config.audioConnectorNumLearnableRegisters,
    audio_gated_attn: config.audioGatedAttn,
    audio_hidden_dim: config.audioHiddenDim,
    caption_channels: config.captionChannels,
    connector_rope_base_seq_len: config.connectorRopeBaseSeqLen,
    per_modality_projections: config.perModalityProjections,
    proj_bias: config.projBias,
    rope_theta: config.ropeTheta,
    rope_type: config.ropeType,
    text_proj_in_factor: config.textProjInFactor,
    video_connector_attention_head_dim: config.videoConnectorAttentionHeadDim,
    video_connector_num_attention_heads: config.videoConnectorNumAttentionHeads,
    video_connector_num_layers: config.videoConnectorNumLayers,
    video_connector_num_learnable_registers: config.videoConnectorNumLearnableRegisters,
    video_gated_attn: config.videoGatedAttn,
    video_hidden_dim: config.videoHiddenDim,
  };
}

function writeLtx2SnapshotMetadata(directory: string): void {
  const config = tinyConfig();
  writeJson(join(directory, "model_index.json"), {
    _class_name: "LTX2Pipeline",
    audio_vae: ["diffusers", "AutoencoderKLLTX2Audio"],
    connectors: ["ltx2", "LTX2TextConnectors"],
    scheduler: ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    text_encoder: ["transformers", "Gemma3ForConditionalGeneration"],
    tokenizer: ["transformers", "GemmaTokenizerFast"],
    transformer: ["diffusers", "LTX2VideoTransformer3DModel"],
    vae: ["diffusers", "AutoencoderKLLTX2Video"],
    vocoder: ["ltx2", "LTX2Vocoder"],
  });
  writeComponent(directory, "transformer", {
    _class_name: "LTX2VideoTransformer3DModel",
    audio_in_channels: 4,
    audio_out_channels: 4,
    caption_channels: config.captionChannels,
    in_channels: 4,
  });
  writeComponent(directory, "vae", {
    _class_name: "AutoencoderKLLTX2Video",
    latent_channels: 4,
  });
  writeComponent(directory, "audio_vae", {
    _class_name: "AutoencoderKLLTX2Audio",
    latent_channels: 1,
    mel_bins: 16,
  });
  writeComponent(directory, "connectors", ltx2ConnectorsRawConfig(config));
  writeComponent(directory, "vocoder", {
    _class_name: "LTX2Vocoder",
    in_channels: 4,
  });
  writeComponent(directory, "text_encoder", {});
  const scheduler = join(directory, "scheduler");
  mkdirSync(scheduler, { recursive: true });
  writeJson(join(scheduler, "scheduler_config.json"), {
    _class_name: "FlowMatchEulerDiscreteScheduler",
  });
  const tokenizer = join(directory, "tokenizer");
  mkdirSync(tokenizer, { recursive: true });
  writeFileSync(join(tokenizer, "tokenizer.json"), "{}");
}

describe("LTX-2 text connector weight mapping", () => {
  test("maps top-level, connector, attention, gate, and feed-forward parameters", () => {
    expect(ltx2TextConnectorWeightPath("text_proj_in.weight")).toBe("textProjIn.weight");
    expect(ltx2TextConnectorWeightPath("video_text_proj_in.bias")).toBe("videoTextProjIn.bias");
    expect(ltx2TextConnectorWeightPath("video_connector.learnable_registers")).toBe(
      "videoConnector.learnableRegisters",
    );
    expect(
      ltx2TextConnectorWeightPath("audio_connector.transformer_blocks.3.attn1.norm_q.weight"),
    ).toBe("audioConnector.transformerBlocks.3.attn1.normQ.weight");
    expect(
      ltx2TextConnectorWeightPath("video_connector.transformer_blocks.3.attn1.to_out.0.bias"),
    ).toBe("videoConnector.transformerBlocks.3.attn1.toOut.bias");
    expect(
      ltx2TextConnectorWeightPath(
        "audio_connector.transformer_blocks.3.attn1.to_gate_logits.weight",
      ),
    ).toBe("audioConnector.transformerBlocks.3.attn1.toGateLogits.weight");
    expect(
      ltx2TextConnectorWeightPath("video_connector.transformer_blocks.3.ff.net.0.proj.weight"),
    ).toBe("videoConnector.transformerBlocks.3.ff.linear1.weight");
    expect(ltx2TextConnectorWeightPath("audio_connector.transformer_blocks.3.ff.net.2.bias")).toBe(
      "audioConnector.transformerBlocks.3.ff.linear2.bias",
    );
  });

  test("does not map dropout, affine block norms, RoPE, and unknown paths", () => {
    expect(
      ltx2TextConnectorWeightPath("video_connector.transformer_blocks.0.norm1.weight"),
    ).toBeNull();
    expect(ltx2TextConnectorWeightPath("audio_connector.norm_out.weight")).toBeNull();
    expect(ltx2TextConnectorWeightPath("video_connector.rope.freqs")).toBeNull();
    expect(
      ltx2TextConnectorWeightPath("video_connector.transformer_blocks.0.attn1.to_out.1.weight"),
    ).toBeNull();
    expect(ltx2TextConnectorWeightPath("unknown.weight")).toBeNull();
  });

  test("loads a complete generated connector safetensors shard", async () => {
    await withTempDirectory("mlxts-ltx2-connectors-weights-", async (directory) => {
      using target = new Ltx2TextConnectors(tinyConfig());
      using source = new Ltx2TextConnectors(tinyConfig());
      const checkpointPath = join(directory, "diffusion_pytorch_model.safetensors");
      await writeCheckpoint(
        checkpointPath,
        checkpointTensorsForLtx2Connectors(source.parameters()),
      );

      const result = await loadLtx2TextConnectorWeights(
        target,
        connectorsComponent([checkpointPath]),
      );

      expect(result.shardCount).toBe(1);
      expect(result.unexpectedWeights).toEqual([]);
      expect(result.assignedPaths).toContain("textProjIn.weight");
      expect(result.assignedPaths).toContain("videoConnector.learnableRegisters");
      expect(result.assignedPaths).toContain(
        "audioConnector.transformerBlocks.0.attn1.normK.weight",
      );
      expect(result.assignedPaths).toContain("videoConnector.transformerBlocks.0.ff.linear2.bias");
    });
  });

  test("loads connector shards from a Diffusers safetensors index", async () => {
    await withTempDirectory("mlxts-ltx2-connectors-index-", async (directory) => {
      using target = new Ltx2TextConnectors(tinyConfig());
      using source = new Ltx2TextConnectors(tinyConfig());
      const shards = splitIndexedShards(checkpointTensorsForLtx2Connectors(source.parameters()));
      const firstPath = join(directory, "diffusion_pytorch_model-00001-of-00002.safetensors");
      const secondPath = join(directory, "diffusion_pytorch_model-00002-of-00002.safetensors");
      await writeCheckpoint(firstPath, shards.firstShard);
      await writeCheckpoint(secondPath, shards.secondShard);
      const indexPath = join(directory, "diffusion_pytorch_model.safetensors.index.json");
      writeJson(indexPath, { metadata: {}, weight_map: shards.weightMap });

      const result = await loadLtx2TextConnectorWeights(target, connectorsComponent([indexPath]));

      expect(result.shardCount).toBe(2);
      expect(result.unexpectedWeights).toEqual([]);
    });
  });

  test("loads connectors directly from an inspected LTX-2 snapshot manifest", async () => {
    await withTempDirectory("mlxts-ltx2-connectors-snapshot-", async (directory) => {
      writeLtx2SnapshotMetadata(directory);
      using source = new Ltx2TextConnectors(tinyConfig());
      await writeCheckpoint(
        join(directory, "connectors", "diffusion_pytorch_model.safetensors"),
        checkpointTensorsForLtx2Connectors(source.parameters()),
      );
      const manifest = await loadDiffusionSnapshotManifest(directory);

      using target = await loadLtx2TextConnectorsFromSnapshot(manifest);

      expect(target.textProjIn?.weight.shape).toEqual([4, 8]);
      expect(target.videoConnector.learnableRegisters?.shape).toEqual([2, 4]);
      expect(target.audioConnector.transformerBlocks[0]?.attn1.toV.weight.shape).toEqual([4, 4]);
    });
  });

  test("rejects missing, mismatched, and strict unexpected connector weights", async () => {
    await withTempDirectory("mlxts-ltx2-connectors-invalid-", async (directory) => {
      using source = new Ltx2TextConnectors(tinyConfig());

      const missingPath = join(directory, "missing.safetensors");
      await writeCheckpoint(
        missingPath,
        checkpointTensorsForLtx2Connectors(source.parameters(), {
          omitPath: "videoConnector.transformerBlocks.0.attn1.toQ.weight",
        }),
      );
      using missingTarget = new Ltx2TextConnectors(tinyConfig());
      await expect(
        loadLtx2TextConnectorWeights(missingTarget, connectorsComponent([missingPath])),
      ).rejects.toThrow(DiffusionMissingWeightsError);

      const mismatchPath = join(directory, "mismatch.safetensors");
      await writeCheckpoint(
        mismatchPath,
        checkpointTensorsForLtx2Connectors(source.parameters(), {
          mismatchPath: "audioConnector.transformerBlocks.0.attn1.toK.weight",
        }),
      );
      using mismatchTarget = new Ltx2TextConnectors(tinyConfig());
      await expect(
        loadLtx2TextConnectorWeights(mismatchTarget, connectorsComponent([mismatchPath])),
      ).rejects.toThrow(DiffusionWeightMismatchError);

      const unexpectedPath = join(directory, "unexpected.safetensors");
      await writeCheckpoint(
        unexpectedPath,
        checkpointTensorsForLtx2Connectors(source.parameters(), {
          extra: { "video_connector.transformer_blocks.0.attn1.to_out.1.weight": zeros([1]) },
        }),
      );
      using unexpectedTarget = new Ltx2TextConnectors(tinyConfig());
      await expect(
        loadLtx2TextConnectorWeights(unexpectedTarget, connectorsComponent([unexpectedPath]), {
          strictUnexpectedWeights: true,
        }),
      ).rejects.toThrow("unexpected unmapped weights");
    });
  });
});
