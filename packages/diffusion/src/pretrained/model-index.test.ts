import { describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "fs";
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
    });

    expect(parsed.kind).toBe("stable-diffusion-xl");
    expect(parsed.pipelineConfig).toEqual({ force_zeros_for_empty_prompt: true });
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
    expect(() => parseDiffusionModelIndex({ _class_name: "QwenImageEditPipeline" })).toThrow(
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
