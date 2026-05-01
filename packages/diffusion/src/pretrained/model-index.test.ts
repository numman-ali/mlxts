import { describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

import { DiffusionConfigError } from "../errors";

import { parseDiffusionModelIndex } from "./model-index";
import { loadDiffusionSnapshotManifest } from "./snapshot-manifest";

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

function writeStableDiffusionSnapshot(snapshot: string): void {
  writeJson(join(snapshot, "model_index.json"), {
    _class_name: "StableDiffusionPipeline",
    _diffusers_version: "0.35.2",
    vae: ["diffusers", "AutoencoderKL"],
    text_encoder: ["transformers", "CLIPTextModel"],
    tokenizer: ["transformers", "CLIPTokenizer"],
    unet: ["diffusers", "UNet2DConditionModel"],
    scheduler: ["diffusers", "DDIMScheduler"],
    safety_checker: [null, null],
    requires_safety_checker: false,
  });
  writeComponentFiles(snapshot, "vae", ["config.json"], ["diffusion_pytorch_model.safetensors"]);
  writeComponentFiles(snapshot, "text_encoder", ["config.json"], ["model.safetensors"]);
  writeComponentFiles(snapshot, "tokenizer", ["vocab.json", "merges.txt"]);
  writeComponentFiles(snapshot, "unet", ["config.json"], ["diffusion_pytorch_model.safetensors"]);
  writeComponentFiles(snapshot, "scheduler", ["scheduler_config.json"]);
  writeJson(join(snapshot, "scheduler", "scheduler_config.json"), {
    _class_name: "DDIMScheduler",
    beta_schedule: "linear",
    beta_start: 0.1,
    beta_end: 0.2,
    num_train_timesteps: 4,
  });
}

function writeStableDiffusion3Snapshot(snapshot: string): void {
  writeJson(join(snapshot, "model_index.json"), {
    _class_name: "StableDiffusion3Pipeline",
    _diffusers_version: "0.30.0.dev0",
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
  writeComponentFiles(snapshot, "vae", ["config.json"], ["diffusion_pytorch_model.safetensors"]);
  writeComponentFiles(snapshot, "scheduler", ["scheduler_config.json"]);
  writeJson(join(snapshot, "scheduler", "scheduler_config.json"), {
    _class_name: "FlowMatchEulerDiscreteScheduler",
    num_train_timesteps: 1000,
    shift: 1,
  });
  writeComponentFiles(snapshot, "text_encoder", ["config.json"], ["model.safetensors"]);
  writeComponentFiles(snapshot, "text_encoder_2", ["config.json"], ["model.safetensors"]);
  writeComponentFiles(snapshot, "text_encoder_3", ["config.json"], ["model.safetensors"]);
  writeComponentFiles(snapshot, "tokenizer", ["vocab.json", "merges.txt"]);
  writeComponentFiles(snapshot, "tokenizer_2", ["vocab.json", "merges.txt"]);
  writeComponentFiles(snapshot, "tokenizer_3", ["spiece.model"]);
}

function writeLtxVideoSnapshot(snapshot: string): void {
  writeJson(join(snapshot, "model_index.json"), {
    _class_name: "LTXPipeline",
    _diffusers_version: "0.32.0.dev0",
    scheduler: ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    text_encoder: ["transformers", "T5EncoderModel"],
    tokenizer: ["transformers", "T5Tokenizer"],
    transformer: ["diffusers", "LTXVideoTransformer3DModel"],
    vae: ["diffusers", "AutoencoderKLLTXVideo"],
  });
  writeComponentFiles(
    snapshot,
    "transformer",
    ["config.json"],
    ["diffusion_pytorch_model.safetensors"],
  );
  writeComponentFiles(snapshot, "vae", ["config.json"], ["diffusion_pytorch_model.safetensors"]);
  writeComponentFiles(snapshot, "scheduler", ["scheduler_config.json"]);
  writeJson(join(snapshot, "scheduler", "scheduler_config.json"), {
    _class_name: "FlowMatchEulerDiscreteScheduler",
    use_dynamic_shifting: true,
  });
  writeComponentFiles(snapshot, "text_encoder", ["config.json"], ["model.safetensors"]);
  writeComponentFiles(snapshot, "tokenizer", ["spiece.model"]);
}

function writeLtx2Snapshot(snapshot: string): void {
  writeJson(join(snapshot, "model_index.json"), {
    _class_name: "LTX2Pipeline",
    _diffusers_version: "0.37.0.dev0",
    audio_vae: ["diffusers", "AutoencoderKLLTX2Audio"],
    connectors: ["ltx2", "LTX2TextConnectors"],
    scheduler: ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    text_encoder: ["transformers", "Gemma3ForConditionalGeneration"],
    tokenizer: ["transformers", "GemmaTokenizerFast"],
    transformer: ["diffusers", "LTX2VideoTransformer3DModel"],
    vae: ["diffusers", "AutoencoderKLLTX2Video"],
    vocoder: ["ltx2", "LTX2Vocoder"],
  });
  writeComponentFiles(
    snapshot,
    "transformer",
    ["config.json"],
    ["diffusion_pytorch_model.safetensors"],
  );
  writeComponentFiles(snapshot, "vae", ["config.json"], ["diffusion_pytorch_model.safetensors"]);
  writeComponentFiles(
    snapshot,
    "audio_vae",
    ["config.json"],
    ["diffusion_pytorch_model.safetensors"],
  );
  writeComponentFiles(snapshot, "scheduler", ["scheduler_config.json"]);
  writeJson(join(snapshot, "scheduler", "scheduler_config.json"), {
    _class_name: "FlowMatchEulerDiscreteScheduler",
    shift: 3,
  });
  writeComponentFiles(snapshot, "text_encoder", ["config.json"], ["model.safetensors"]);
  writeComponentFiles(snapshot, "tokenizer", ["tokenizer.json"]);
  writeComponentFiles(
    snapshot,
    "connectors",
    ["config.json"],
    ["diffusion_pytorch_model.safetensors"],
  );
  writeComponentFiles(
    snapshot,
    "vocoder",
    ["config.json"],
    ["diffusion_pytorch_model.safetensors"],
  );
}

describe("diffusion model index loading", () => {
  test("parses Stable Diffusion model_index components", () => {
    const parsed = parseDiffusionModelIndex({
      _class_name: "StableDiffusionPipeline",
      _diffusers_version: "0.35.2",
      vae: ["diffusers", "AutoencoderKL"],
      text_encoder: ["transformers", "CLIPTextModel"],
      tokenizer: ["transformers", "CLIPTokenizerFast"],
      unet: ["diffusers", "UNet2DConditionModel"],
      scheduler: ["diffusers", "EulerDiscreteScheduler"],
      safety_checker: [null, null],
      requires_safety_checker: false,
    });

    expect(parsed.kind).toBe("stable-diffusion");
    expect(parsed.diffusersVersion).toBe("0.35.2");
    expect(parsed.pipelineConfig).toEqual({ requires_safety_checker: false });
    expect(parsed.components.map((component) => component.name)).toEqual([
      "vae",
      "text_encoder",
      "tokenizer",
      "unet",
      "scheduler",
      "safety_checker",
    ]);
    expect(
      parsed.components.find((component) => component.name === "safety_checker"),
    ).toMatchObject({
      enabled: false,
      optional: true,
    });
  });

  test("parses Stable Diffusion XL model_index components", () => {
    const parsed = parseDiffusionModelIndex({
      _class_name: "StableDiffusionXLPipeline",
      vae: ["diffusers", "AutoencoderKL"],
      text_encoder: ["transformers", "CLIPTextModel"],
      text_encoder_2: ["transformers", "CLIPTextModelWithProjection"],
      tokenizer: ["transformers", "CLIPTokenizer"],
      tokenizer_2: ["transformers", "CLIPTokenizerFast"],
      unet: ["diffusers", "UNet2DConditionModel"],
      scheduler: ["diffusers", "EulerDiscreteScheduler"],
      force_zeros_for_empty_prompt: true,
      add_watermarker: null,
    });

    expect(parsed.kind).toBe("stable-diffusion-xl");
    expect(parsed.pipelineConfig).toEqual({
      add_watermarker: null,
      force_zeros_for_empty_prompt: true,
    });
    expect(parsed.components.map((component) => component.name)).toEqual([
      "vae",
      "text_encoder",
      "text_encoder_2",
      "tokenizer",
      "tokenizer_2",
      "unet",
      "scheduler",
    ]);
  });

  test("parses Stable Diffusion 3 model_index components", () => {
    const parsed = parseDiffusionModelIndex({
      _class_name: "StableDiffusion3Pipeline",
      _diffusers_version: "0.30.0.dev0",
      scheduler: ["diffusers", "FlowMatchEulerDiscreteScheduler"],
      text_encoder: ["transformers", "CLIPTextModelWithProjection"],
      text_encoder_2: ["transformers", "CLIPTextModelWithProjection"],
      text_encoder_3: ["transformers", "T5EncoderModel"],
      tokenizer: ["transformers", "CLIPTokenizer"],
      tokenizer_2: ["transformers", "CLIPTokenizerFast"],
      tokenizer_3: ["transformers", "T5TokenizerFast"],
      transformer: ["diffusers", "SD3Transformer2DModel"],
      vae: ["diffusers", "AutoencoderKL"],
      image_encoder: [null, null],
      feature_extractor: [null, null],
    });

    expect(parsed.kind).toBe("stable-diffusion-3");
    expect(parsed.diffusersVersion).toBe("0.30.0.dev0");
    expect(parsed.components.map((component) => component.name)).toEqual([
      "transformer",
      "scheduler",
      "vae",
      "text_encoder",
      "tokenizer",
      "text_encoder_2",
      "tokenizer_2",
      "text_encoder_3",
      "tokenizer_3",
      "image_encoder",
      "feature_extractor",
    ]);
    expect(
      parsed.components.find((component) => component.name === "text_encoder_3"),
    ).toMatchObject({
      role: "text-encoder",
      className: "T5EncoderModel",
    });
    expect(parsed.components.find((component) => component.name === "image_encoder")).toMatchObject(
      {
        enabled: false,
        optional: true,
      },
    );
  });

  test("parses Flux model_index without claiming scheduler support", () => {
    const parsed = parseDiffusionModelIndex({
      _class_name: "FluxPipeline",
      transformer: ["diffusers", "FluxTransformer2DModel"],
      vae: ["diffusers", "AutoencoderKL"],
      scheduler: ["diffusers", "FlowMatchEulerDiscreteScheduler"],
      text_encoder: ["transformers", "CLIPTextModel"],
      tokenizer: ["transformers", "CLIPTokenizer"],
      text_encoder_2: ["transformers", "T5EncoderModel"],
      tokenizer_2: ["transformers", "T5TokenizerFast"],
    });

    expect(parsed.kind).toBe("flux");
    expect(parsed.components.map((component) => component.name)).toEqual([
      "transformer",
      "vae",
      "scheduler",
      "text_encoder",
      "tokenizer",
      "text_encoder_2",
      "tokenizer_2",
    ]);
  });

  test("parses FLUX.2 Klein model_index components", () => {
    const parsed = parseDiffusionModelIndex({
      _class_name: "Flux2KleinPipeline",
      _diffusers_version: "0.37.0.dev0",
      is_distilled: true,
      scheduler: ["diffusers", "FlowMatchEulerDiscreteScheduler"],
      text_encoder: ["transformers", "Qwen3ForCausalLM"],
      tokenizer: ["transformers", "Qwen2TokenizerFast"],
      transformer: ["diffusers", "Flux2Transformer2DModel"],
      vae: ["diffusers", "AutoencoderKLFlux2"],
    });

    expect(parsed.kind).toBe("flux2-klein");
    expect(parsed.diffusersVersion).toBe("0.37.0.dev0");
    expect(parsed.pipelineConfig).toEqual({ is_distilled: true });
    expect(parsed.components.map((component) => component.name)).toEqual([
      "transformer",
      "vae",
      "scheduler",
      "text_encoder",
      "tokenizer",
    ]);
    expect(parsed.components.find((component) => component.name === "text_encoder")).toMatchObject({
      role: "text-encoder",
      className: "Qwen3ForCausalLM",
    });
  });

  test("parses Qwen-Image model_index components", () => {
    const parsed = parseDiffusionModelIndex({
      _class_name: "QwenImagePipeline",
      _diffusers_version: "0.36.0.dev0",
      transformer: ["diffusers", "QwenImageTransformer2DModel"],
      vae: ["diffusers", "AutoencoderKLQwenImage"],
      scheduler: ["diffusers", "FlowMatchEulerDiscreteScheduler"],
      text_encoder: ["transformers", "Qwen2_5_VLForConditionalGeneration"],
      tokenizer: ["transformers", "Qwen2Tokenizer"],
    });

    expect(parsed.kind).toBe("qwen-image");
    expect(parsed.diffusersVersion).toBe("0.36.0.dev0");
    expect(parsed.components.map((component) => component.name)).toEqual([
      "transformer",
      "vae",
      "scheduler",
      "text_encoder",
      "tokenizer",
    ]);
    expect(parsed.components.find((component) => component.name === "vae")).toMatchObject({
      role: "vae",
      className: "AutoencoderKLQwenImage",
    });
  });

  test("parses Z-Image model_index components", () => {
    const parsed = parseDiffusionModelIndex({
      _class_name: "ZImagePipeline",
      _diffusers_version: "0.36.0.dev0",
      transformer: ["diffusers", "ZImageTransformer2DModel"],
      vae: ["diffusers", "AutoencoderKL"],
      scheduler: ["diffusers", "FlowMatchEulerDiscreteScheduler"],
      text_encoder: ["transformers", "Qwen3Model"],
      tokenizer: ["transformers", "Qwen2Tokenizer"],
    });

    expect(parsed.kind).toBe("z-image");
    expect(parsed.diffusersVersion).toBe("0.36.0.dev0");
    expect(parsed.components.map((component) => component.name)).toEqual([
      "transformer",
      "vae",
      "scheduler",
      "text_encoder",
      "tokenizer",
    ]);
    expect(parsed.components.find((component) => component.name === "text_encoder")).toMatchObject({
      role: "text-encoder",
      className: "Qwen3Model",
    });
  });

  test("parses LTX-Video model_index components", () => {
    const modelIndex = {
      _class_name: "LTXPipeline",
      _diffusers_version: "0.32.0.dev0",
      scheduler: ["diffusers", "FlowMatchEulerDiscreteScheduler"],
      text_encoder: ["transformers", "T5EncoderModel"],
      tokenizer: ["transformers", "T5Tokenizer"],
      transformer: ["diffusers", "LTXVideoTransformer3DModel"],
      vae: ["diffusers", "AutoencoderKLLTXVideo"],
    };
    const parsed = parseDiffusionModelIndex(modelIndex);

    expect(parsed.kind).toBe("ltx-video");
    expect(parsed.diffusersVersion).toBe("0.32.0.dev0");
    expect(parsed.components.map((component) => component.name)).toEqual([
      "transformer",
      "vae",
      "scheduler",
      "text_encoder",
      "tokenizer",
    ]);
    expect(parsed.components.find((component) => component.name === "transformer")).toMatchObject({
      role: "backbone",
      className: "LTXVideoTransformer3DModel",
    });
    expect(
      parseDiffusionModelIndex({
        ...modelIndex,
        _class_name: "LTXConditionPipeline",
        _diffusers_version: "0.34.0.dev0",
      }).kind,
    ).toBe("ltx-video");
  });

  test("parses LTX-2 audio-video model_index components", () => {
    const parsed = parseDiffusionModelIndex({
      _class_name: "LTX2Pipeline",
      _diffusers_version: "0.37.0.dev0",
      audio_vae: ["diffusers", "AutoencoderKLLTX2Audio"],
      connectors: ["ltx2", "LTX2TextConnectors"],
      scheduler: ["diffusers", "FlowMatchEulerDiscreteScheduler"],
      text_encoder: ["transformers", "Gemma3ForConditionalGeneration"],
      tokenizer: ["transformers", "GemmaTokenizerFast"],
      transformer: ["diffusers", "LTX2VideoTransformer3DModel"],
      vae: ["diffusers", "AutoencoderKLLTX2Video"],
      vocoder: ["ltx2", "LTX2Vocoder"],
    });

    expect(parsed.kind).toBe("ltx2");
    expect(parsed.diffusersVersion).toBe("0.37.0.dev0");
    expect(parsed.components.map((component) => component.name)).toEqual([
      "transformer",
      "vae",
      "audio_vae",
      "scheduler",
      "text_encoder",
      "tokenizer",
      "connectors",
      "vocoder",
    ]);
    expect(parsed.components.find((component) => component.name === "audio_vae")).toMatchObject({
      role: "audio-vae",
      className: "AutoencoderKLLTX2Audio",
    });
    expect(parsed.components.find((component) => component.name === "vocoder")).toMatchObject({
      role: "vocoder",
      library: "ltx2",
    });
  });

  test("loads a local snapshot manifest without constructing models", async () => {
    await withTempDirectory("mlxts-diffusion-model-index-", async (directory) => {
      writeStableDiffusionSnapshot(directory);

      const manifest = await loadDiffusionSnapshotManifest(directory);
      const unet = manifest.components.find((component) => component.name === "unet");
      const tokenizer = manifest.components.find((component) => component.name === "tokenizer");

      expect(manifest.modelIndex.className).toBe("StableDiffusionPipeline");
      expect(manifest.schedulerConfig.kind).toBe("ddim");
      expect(unet?.weightPaths).toEqual([
        join(directory, "unet", "diffusion_pytorch_model.safetensors"),
      ]);
      expect(tokenizer?.metadataPaths).toContain(join(directory, "tokenizer", "vocab.json"));
    });
  });

  test("loads a Stable Diffusion 3 local snapshot manifest", async () => {
    await withTempDirectory("mlxts-diffusion-sd3-model-index-", async (directory) => {
      writeStableDiffusion3Snapshot(directory);

      const manifest = await loadDiffusionSnapshotManifest(directory);
      const transformer = manifest.components.find((component) => component.name === "transformer");
      const t5Tokenizer = manifest.components.find((component) => component.name === "tokenizer_3");

      expect(manifest.modelIndex.className).toBe("StableDiffusion3Pipeline");
      expect(manifest.modelIndex.kind).toBe("stable-diffusion-3");
      expect(manifest.schedulerConfig.kind).toBe("flow-match-euler");
      expect(transformer?.weightPaths).toEqual([
        join(directory, "transformer", "diffusion_pytorch_model.safetensors"),
      ]);
      expect(t5Tokenizer?.metadataPaths).toEqual([join(directory, "tokenizer_3", "spiece.model")]);
    });
  });

  test("loads LTX-Video and LTX-2 local snapshot manifests", async () => {
    await withTempDirectory("mlxts-diffusion-ltx-video-model-index-", async (directory) => {
      writeLtxVideoSnapshot(directory);

      const manifest = await loadDiffusionSnapshotManifest(directory);

      expect(manifest.modelIndex.className).toBe("LTXPipeline");
      expect(manifest.modelIndex.kind).toBe("ltx-video");
      expect(manifest.schedulerConfig.kind).toBe("flow-match-euler");
      expect(
        manifest.components.find((component) => component.name === "tokenizer")?.metadataPaths,
      ).toEqual([join(directory, "tokenizer", "spiece.model")]);
    });

    await withTempDirectory("mlxts-diffusion-ltx2-model-index-", async (directory) => {
      writeLtx2Snapshot(directory);

      const manifest = await loadDiffusionSnapshotManifest(directory);
      const audioVae = manifest.components.find((component) => component.name === "audio_vae");
      const connector = manifest.components.find((component) => component.name === "connectors");

      expect(manifest.modelIndex.className).toBe("LTX2Pipeline");
      expect(manifest.modelIndex.kind).toBe("ltx2");
      expect(manifest.schedulerConfig.kind).toBe("flow-match-euler");
      expect(audioVae?.weightPaths).toEqual([
        join(directory, "audio_vae", "diffusion_pytorch_model.safetensors"),
      ]);
      expect(connector?.metadataPaths).toEqual([join(directory, "connectors", "config.json")]);
    });
  });

  test("loads symlinked Hub-cache component weights", async () => {
    await withTempDirectory("mlxts-diffusion-model-index-symlinks-", async (directory) => {
      writeStableDiffusionSnapshot(directory);
      const blobDirectory = join(directory, "blobs");
      const blobPath = join(blobDirectory, "vae-fp16");
      const indexBlobPath = join(blobDirectory, "unet-index");
      const vaeWeightPath = join(directory, "vae", "diffusion_pytorch_model.fp16.safetensors");
      const unetIndexPath = join(
        directory,
        "unet",
        "diffusion_pytorch_model.fp16.safetensors.index.json",
      );
      mkdirSync(blobDirectory);
      writeFileSync(blobPath, "weights");
      writeFileSync(indexBlobPath, "{}");
      rmSync(join(directory, "vae", "diffusion_pytorch_model.safetensors"), { force: true });
      rmSync(join(directory, "unet", "diffusion_pytorch_model.safetensors"), { force: true });
      symlinkSync(blobPath, vaeWeightPath);
      symlinkSync(indexBlobPath, unetIndexPath);

      const manifest = await loadDiffusionSnapshotManifest(directory);
      const vae = manifest.components.find((component) => component.name === "vae");
      const unet = manifest.components.find((component) => component.name === "unet");

      expect(vae?.weightPaths).toEqual([vaeWeightPath]);
      expect(unet?.weightPaths).toEqual([unetIndexPath]);
    });
  });

  test("rejects unsupported model_index shapes and component semantics", () => {
    expect(() => parseDiffusionModelIndex(null)).toThrow("JSON object");
    expect(() =>
      parseDiffusionModelIndex({
        _class_name: ["custom_pipeline", "Pipeline"],
      }),
    ).toThrow("_class_name");
    expect(() =>
      parseDiffusionModelIndex({ _class_name: "StableDiffusionImg2ImgPipeline" }),
    ).toThrow("not supported");
    expect(() =>
      parseDiffusionModelIndex({ _class_name: "StableDiffusion3Img2ImgPipeline" }),
    ).toThrow("not supported");
    expect(() => parseDiffusionModelIndex({ _class_name: "QwenImageEditPipeline" })).toThrow(
      "not supported",
    );
    expect(() => parseDiffusionModelIndex({ _class_name: "Flux2Pipeline" })).toThrow(
      "not supported",
    );
    expect(() => parseDiffusionModelIndex({ _class_name: "Flux2KleinInpaintPipeline" })).toThrow(
      "not supported",
    );
    expect(() => parseDiffusionModelIndex({ _class_name: "Flux2KleinKVPipeline" })).toThrow(
      "not supported",
    );
    expect(() => parseDiffusionModelIndex({ _class_name: "ZImageOmniPipeline" })).toThrow(
      "not supported",
    );
    expect(() => parseDiffusionModelIndex({ _class_name: "LTXImageToVideoPipeline" })).toThrow(
      "not supported",
    );
    expect(() => parseDiffusionModelIndex({ _class_name: "LTXLatentUpsamplePipeline" })).toThrow(
      "not supported",
    );
    expect(() => parseDiffusionModelIndex({ _class_name: "LTX2ImageToVideoPipeline" })).toThrow(
      "not supported",
    );
    expect(() => parseDiffusionModelIndex({ _class_name: "LTX2ConditionPipeline" })).toThrow(
      "not supported",
    );
    expect(() => parseDiffusionModelIndex({ _class_name: "LTX2LatentUpsamplePipeline" })).toThrow(
      "not supported",
    );
    expect(() =>
      parseDiffusionModelIndex({
        _class_name: "StableDiffusionPipeline",
        vae: ["diffusers", "AutoencoderKL"],
      }),
    ).toThrow("text_encoder");
    expect(() =>
      parseDiffusionModelIndex({
        _class_name: "StableDiffusionPipeline",
        vae: [null, null],
        text_encoder: ["transformers", "CLIPTextModel"],
        tokenizer: ["transformers", "CLIPTokenizer"],
        unet: ["diffusers", "UNet2DConditionModel"],
        scheduler: ["diffusers", "DDIMScheduler"],
      }),
    ).toThrow("vae");
    expect(() =>
      parseDiffusionModelIndex({
        _class_name: "StableDiffusionPipeline",
        vae: ["diffusers", "AutoencoderKL"],
        text_encoder: ["transformers", "CLIPTextModel"],
        tokenizer: ["transformers", "CLIPTokenizer"],
        unet: ["diffusers", "UNet2DConditionModel"],
        scheduler: ["diffusers", "PNDMScheduler"],
      }),
    ).toThrow("scheduler");
    expect(() =>
      parseDiffusionModelIndex({
        _class_name: "StableDiffusionPipeline",
        vae: ["diffusers", "AutoencoderKL"],
        text_encoder: ["transformers", "CLIPTextModel"],
        tokenizer: ["transformers", "CLIPTokenizer"],
        unet: ["diffusers", "UNet2DConditionModel"],
        scheduler: ["diffusers", "DDIMScheduler"],
        controlnet: ["diffusers", "ControlNetModel"],
      }),
    ).toThrow("controlnet");
    expect(() =>
      parseDiffusionModelIndex({
        _class_name: "StableDiffusionPipeline",
        vae: ["diffusers", "AutoencoderKL"],
        text_encoder: ["transformers", "CLIPTextModel"],
        tokenizer: ["transformers", "CLIPTokenizer"],
        unet: ["diffusers", "UNet2DConditionModel"],
        scheduler: ["diffusers", "DDIMScheduler"],
        requires_safety_checker: { enabled: false },
      }),
    ).toThrow("requires_safety_checker");
  });

  test("rejects local manifests with missing files or unsupported scheduler config", async () => {
    await withTempDirectory("mlxts-diffusion-missing-model-index-", async (directory) => {
      await expect(loadDiffusionSnapshotManifest(directory)).rejects.toThrow("model_index.json");
    });

    await withTempDirectory("mlxts-diffusion-invalid-model-index-", async (directory) => {
      writeFileSync(join(directory, "model_index.json"), "{");
      await expect(loadDiffusionSnapshotManifest(directory)).rejects.toThrow("valid JSON");
    });

    await withTempDirectory("mlxts-diffusion-missing-component-", async (directory) => {
      writeStableDiffusionSnapshot(directory);
      rmSync(join(directory, "unet"), { recursive: true, force: true });
      await expect(loadDiffusionSnapshotManifest(directory)).rejects.toThrow("unet");
    });

    await withTempDirectory("mlxts-diffusion-missing-weights-", async (directory) => {
      writeStableDiffusionSnapshot(directory);
      rmSync(join(directory, "vae", "diffusion_pytorch_model.safetensors"), { force: true });
      await expect(loadDiffusionSnapshotManifest(directory)).rejects.toThrow("safetensors");
    });

    await withTempDirectory("mlxts-diffusion-missing-tokenizer-", async (directory) => {
      writeStableDiffusionSnapshot(directory);
      rmSync(join(directory, "tokenizer", "vocab.json"), { force: true });
      rmSync(join(directory, "tokenizer", "merges.txt"), { force: true });
      await expect(loadDiffusionSnapshotManifest(directory)).rejects.toThrow("tokenizer");
    });

    await withTempDirectory("mlxts-diffusion-unsupported-scheduler-", async (directory) => {
      writeStableDiffusionSnapshot(directory);
      writeJson(join(directory, "scheduler", "scheduler_config.json"), {
        _class_name: "DDIMScheduler",
        prediction_type: "v_prediction",
      });
      await expect(loadDiffusionSnapshotManifest(directory)).rejects.toThrow(DiffusionConfigError);
    });

    await withTempDirectory("mlxts-diffusion-flux-scheduler-", async (directory) => {
      writeJson(join(directory, "model_index.json"), {
        _class_name: "FluxPipeline",
        transformer: ["diffusers", "FluxTransformer2DModel"],
        vae: ["diffusers", "AutoencoderKL"],
        scheduler: ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        text_encoder: ["transformers", "CLIPTextModel"],
        tokenizer: ["transformers", "CLIPTokenizer"],
        text_encoder_2: ["transformers", "T5EncoderModel"],
        tokenizer_2: ["transformers", "T5TokenizerFast"],
      });
      writeComponentFiles(
        directory,
        "transformer",
        ["config.json"],
        ["diffusion_pytorch_model.safetensors"],
      );
      writeComponentFiles(
        directory,
        "vae",
        ["config.json"],
        ["diffusion_pytorch_model.safetensors"],
      );
      writeComponentFiles(directory, "scheduler", ["scheduler_config.json"]);
      writeJson(join(directory, "scheduler", "scheduler_config.json"), {
        _class_name: "FlowMatchEulerDiscreteScheduler",
      });
      writeComponentFiles(directory, "text_encoder", ["config.json"], ["model.safetensors"]);
      writeComponentFiles(directory, "tokenizer", ["vocab.json", "merges.txt"]);
      writeComponentFiles(directory, "text_encoder_2", ["config.json"], ["model.safetensors"]);
      writeComponentFiles(directory, "tokenizer_2", ["spiece.model"]);

      const manifest = await loadDiffusionSnapshotManifest(directory);

      expect(manifest.modelIndex.kind).toBe("flux");
      expect(manifest.schedulerConfig.kind).toBe("flow-match-euler");
    });

    await withTempDirectory("mlxts-diffusion-qwen-image-manifest-", async (directory) => {
      writeJson(join(directory, "model_index.json"), {
        _class_name: "QwenImagePipeline",
        transformer: ["diffusers", "QwenImageTransformer2DModel"],
        vae: ["diffusers", "AutoencoderKLQwenImage"],
        scheduler: ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        text_encoder: ["transformers", "Qwen2_5_VLForConditionalGeneration"],
        tokenizer: ["transformers", "Qwen2Tokenizer"],
      });
      writeComponentFiles(
        directory,
        "transformer",
        ["config.json"],
        ["diffusion_pytorch_model.safetensors"],
      );
      writeComponentFiles(
        directory,
        "vae",
        ["config.json"],
        ["diffusion_pytorch_model.safetensors"],
      );
      writeComponentFiles(directory, "scheduler", ["scheduler_config.json"]);
      writeJson(join(directory, "scheduler", "scheduler_config.json"), {
        _class_name: "FlowMatchEulerDiscreteScheduler",
        shift_terminal: 0.02,
      });
      writeComponentFiles(directory, "text_encoder", ["config.json"], ["model.safetensors"]);
      writeComponentFiles(directory, "tokenizer", ["tokenizer.json"]);

      const manifest = await loadDiffusionSnapshotManifest(directory);

      expect(manifest.modelIndex.kind).toBe("qwen-image");
      expect(manifest.schedulerConfig.kind).toBe("flow-match-euler");
      expect(manifest.schedulerConfig.config).toMatchObject({ shiftTerminal: 0.02 });
      expect(
        manifest.components.find((component) => component.name === "tokenizer")?.metadataPaths,
      ).toEqual([join(directory, "tokenizer", "tokenizer.json")]);
    });

    await withTempDirectory("mlxts-diffusion-flux2-klein-manifest-", async (directory) => {
      writeJson(join(directory, "model_index.json"), {
        _class_name: "Flux2KleinPipeline",
        is_distilled: true,
        scheduler: ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        text_encoder: ["transformers", "Qwen3ForCausalLM"],
        tokenizer: ["transformers", "Qwen2TokenizerFast"],
        transformer: ["diffusers", "Flux2Transformer2DModel"],
        vae: ["diffusers", "AutoencoderKLFlux2"],
      });
      writeComponentFiles(
        directory,
        "transformer",
        ["config.json"],
        ["diffusion_pytorch_model.safetensors"],
      );
      writeComponentFiles(
        directory,
        "vae",
        ["config.json"],
        ["diffusion_pytorch_model.safetensors"],
      );
      writeComponentFiles(directory, "scheduler", ["scheduler_config.json"]);
      writeJson(join(directory, "scheduler", "scheduler_config.json"), {
        _class_name: "FlowMatchEulerDiscreteScheduler",
        use_dynamic_shifting: true,
        time_shift_type: "exponential",
      });
      writeComponentFiles(directory, "text_encoder", ["config.json"], ["model.safetensors"]);
      writeComponentFiles(directory, "tokenizer", ["tokenizer.json"]);

      const manifest = await loadDiffusionSnapshotManifest(directory);

      expect(manifest.modelIndex.kind).toBe("flux2-klein");
      expect(manifest.schedulerConfig.kind).toBe("flow-match-euler");
      expect(manifest.schedulerConfig.config).toMatchObject({
        timeShiftType: "exponential",
        useDynamicShifting: true,
      });
    });

    await withTempDirectory("mlxts-diffusion-z-image-manifest-", async (directory) => {
      writeJson(join(directory, "model_index.json"), {
        _class_name: "ZImagePipeline",
        transformer: ["diffusers", "ZImageTransformer2DModel"],
        vae: ["diffusers", "AutoencoderKL"],
        scheduler: ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        text_encoder: ["transformers", "Qwen3Model"],
        tokenizer: ["transformers", "Qwen2Tokenizer"],
      });
      writeComponentFiles(
        directory,
        "transformer",
        ["config.json"],
        ["diffusion_pytorch_model.safetensors"],
      );
      writeComponentFiles(
        directory,
        "vae",
        ["config.json"],
        ["diffusion_pytorch_model.safetensors"],
      );
      writeComponentFiles(directory, "scheduler", ["scheduler_config.json"]);
      writeJson(join(directory, "scheduler", "scheduler_config.json"), {
        _class_name: "FlowMatchEulerDiscreteScheduler",
        shift: 3,
        use_dynamic_shifting: false,
      });
      writeComponentFiles(directory, "text_encoder", ["config.json"], ["model.safetensors"]);
      writeComponentFiles(directory, "tokenizer", ["tokenizer.json"]);

      const manifest = await loadDiffusionSnapshotManifest(directory);

      expect(manifest.modelIndex.kind).toBe("z-image");
      expect(manifest.schedulerConfig.kind).toBe("flow-match-euler");
      expect(manifest.schedulerConfig.config).toMatchObject({ shift: 3 });
      expect(
        manifest.components.find((component) => component.name === "tokenizer")?.metadataPaths,
      ).toEqual([join(directory, "tokenizer", "tokenizer.json")]);
    });

    await withTempDirectory("mlxts-diffusion-flux-unsupported-scheduler-", async (directory) => {
      writeJson(join(directory, "model_index.json"), {
        _class_name: "FluxPipeline",
        transformer: ["diffusers", "FluxTransformer2DModel"],
        vae: ["diffusers", "AutoencoderKL"],
        scheduler: ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        text_encoder: ["transformers", "CLIPTextModel"],
        tokenizer: ["transformers", "CLIPTokenizer"],
        text_encoder_2: ["transformers", "T5EncoderModel"],
        tokenizer_2: ["transformers", "T5TokenizerFast"],
      });
      writeComponentFiles(
        directory,
        "transformer",
        ["config.json"],
        ["diffusion_pytorch_model.safetensors"],
      );
      writeComponentFiles(
        directory,
        "vae",
        ["config.json"],
        ["diffusion_pytorch_model.safetensors"],
      );
      writeComponentFiles(directory, "scheduler", ["scheduler_config.json"]);
      writeJson(join(directory, "scheduler", "scheduler_config.json"), {
        _class_name: "FlowMatchEulerDiscreteScheduler",
        stochastic_sampling: true,
      });
      writeComponentFiles(directory, "text_encoder", ["config.json"], ["model.safetensors"]);
      writeComponentFiles(directory, "tokenizer", ["vocab.json", "merges.txt"]);
      writeComponentFiles(directory, "text_encoder_2", ["config.json"], ["model.safetensors"]);
      writeComponentFiles(directory, "tokenizer_2", ["spiece.model"]);

      await expect(loadDiffusionSnapshotManifest(directory)).rejects.toThrow("stochastic_sampling");
    });
  });
});
