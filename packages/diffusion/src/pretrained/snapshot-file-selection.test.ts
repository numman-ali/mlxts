import { describe, expect, test } from "bun:test";

import {
  isSupportedRemoteFile,
  type RemoteDiffusionSnapshotFile,
  selectSupportedRemoteFiles,
} from "./snapshot-file-selection";

function file(relativePath: string, size = 1): RemoteDiffusionSnapshotFile {
  return { relativePath, size };
}

describe("Diffusers snapshot remote file selection", () => {
  test("selects requested component weight variants and skips root monoliths", () => {
    const selected = selectSupportedRemoteFiles(
      "segmind/SSD-1B",
      "abc",
      [
        file("README.md"),
        file("SSD-1B.safetensors", 4_000),
        file("model_index.json"),
        file("scheduler/scheduler_config.json"),
        file("text_encoder/config.json"),
        file("text_encoder/model.safetensors", 200),
        file("text_encoder/model.fp16.safetensors", 100),
        file("text_encoder_2/config.json"),
        file("text_encoder_2/model.safetensors", 2_000),
        file("text_encoder_2/model.fp16.safetensors", 1_000),
        file("tokenizer/vocab.json"),
        file("tokenizer/merges.txt"),
        file("unet/config.json"),
        file("unet/diffusion_pytorch_model.safetensors", 5_000),
        file("unet/diffusion_pytorch_model.fp16.safetensors", 2_500),
        file("vae/config.json"),
        file("vae/diffusion_pytorch_model.safetensors", 300),
        file("vae/diffusion_pytorch_model.fp16.safetensors", 150),
      ],
      { variant: "fp16" },
    );

    expect(selected.map((entry) => entry.relativePath)).toEqual([
      "model_index.json",
      "scheduler/scheduler_config.json",
      "text_encoder_2/config.json",
      "text_encoder_2/model.fp16.safetensors",
      "text_encoder/config.json",
      "text_encoder/model.fp16.safetensors",
      "tokenizer/merges.txt",
      "tokenizer/vocab.json",
      "unet/config.json",
      "unet/diffusion_pytorch_model.fp16.safetensors",
      "vae/config.json",
      "vae/diffusion_pytorch_model.fp16.safetensors",
    ]);
  });

  test("defaults to component-local unvariant weights", () => {
    const selected = selectSupportedRemoteFiles("black-forest-labs/FLUX.1-schnell", "abc", [
      file("model_index.json"),
      file("transformer/config.json"),
      file("transformer/diffusion_pytorch_model.safetensors"),
      file("transformer/diffusion_pytorch_model.fp16.safetensors"),
      file("vae/config.json"),
      file("vae/diffusion_pytorch_model.safetensors"),
    ]);

    expect(selected.map((entry) => entry.relativePath)).toEqual([
      "model_index.json",
      "transformer/config.json",
      "transformer/diffusion_pytorch_model.safetensors",
      "vae/config.json",
      "vae/diffusion_pytorch_model.safetensors",
    ]);
  });

  test("rejects ambiguous variant-only component weights without an explicit variant", () => {
    expect(() =>
      selectSupportedRemoteFiles("example/image", "abc", [
        file("model_index.json"),
        file("transformer/config.json"),
        file("transformer/diffusion_pytorch_model.fp16.safetensors"),
        file("transformer/diffusion_pytorch_model.bf16.safetensors"),
      ]),
    ).toThrow("multiple safetensors variants");
  });

  test("selects a single variant-only component when no default exists", () => {
    const selected = selectSupportedRemoteFiles("example/image", "abc", [
      file("model_index.json"),
      file("transformer/config.json"),
      file("transformer/diffusion_pytorch_model.fp16.safetensors"),
    ]);

    expect(selected.map((entry) => entry.relativePath)).toEqual([
      "model_index.json",
      "transformer/config.json",
      "transformer/diffusion_pytorch_model.fp16.safetensors",
    ]);
  });

  test("falls back to unvariant component weights when requested variant is absent", () => {
    const selected = selectSupportedRemoteFiles(
      "black-forest-labs/FLUX.1-schnell",
      "abc",
      [
        file("model_index.json"),
        file("transformer/config.json"),
        file("transformer/diffusion_pytorch_model.safetensors"),
      ],
      { variant: "fp16" },
    );

    expect(selected.map((entry) => entry.relativePath)).toEqual([
      "model_index.json",
      "transformer/config.json",
      "transformer/diffusion_pytorch_model.safetensors",
    ]);
  });

  test("keeps sharded component indexes with matching shards", () => {
    const selected = selectSupportedRemoteFiles(
      "Tongyi-MAI/Z-Image",
      "abc",
      [
        file("model_index.json"),
        file("text_encoder/config.json"),
        file("text_encoder/generation_config.json"),
        file("text_encoder/model.safetensors.index.json"),
        file("text_encoder/model-00001-of-00002.safetensors"),
        file("text_encoder/model-00002-of-00002.safetensors"),
        file("transformer/config.json"),
        file("transformer/diffusion_pytorch_model.fp16.safetensors.index.json"),
        file("transformer/diffusion_pytorch_model.fp16-00001-of-00002.safetensors"),
        file("transformer/diffusion_pytorch_model.fp16-00002-of-00002.safetensors"),
        file("transformer/diffusion_pytorch_model.bf16.safetensors.index.json"),
        file("transformer/diffusion_pytorch_model.bf16-00001-of-00002.safetensors"),
        file("vae/config.json"),
        file("vae/diffusion_pytorch_model.safetensors"),
      ],
      { variant: "fp16" },
    );

    expect(selected.map((entry) => entry.relativePath)).toEqual([
      "model_index.json",
      "text_encoder/config.json",
      "text_encoder/generation_config.json",
      "text_encoder/model-00001-of-00002.safetensors",
      "text_encoder/model-00002-of-00002.safetensors",
      "text_encoder/model.safetensors.index.json",
      "transformer/config.json",
      "transformer/diffusion_pytorch_model.fp16-00001-of-00002.safetensors",
      "transformer/diffusion_pytorch_model.fp16-00002-of-00002.safetensors",
      "transformer/diffusion_pytorch_model.fp16.safetensors.index.json",
      "vae/config.json",
      "vae/diffusion_pytorch_model.safetensors",
    ]);
  });

  test("rejects requested variants when no matching or default component weights exist", () => {
    expect(() =>
      selectSupportedRemoteFiles(
        "example/image",
        "abc",
        [
          file("model_index.json"),
          file("transformer/config.json"),
          file("transformer/diffusion_pytorch_model.bf16.safetensors"),
        ],
        { variant: "fp16" },
      ),
    ).toThrow('has no "fp16" or default safetensors variant');
  });

  test("rejects repos without component safetensors", () => {
    expect(() =>
      selectSupportedRemoteFiles("example/config-only", "abc", [
        file("model_index.json"),
        file("scheduler/scheduler_config.json"),
        file("SSD-1B.safetensors"),
      ]),
    ).toThrow("component artifacts");
  });

  test("reports only metadata and component-local safetensors as supported remote files", () => {
    expect(isSupportedRemoteFile("model_index.json")).toBe(true);
    expect(isSupportedRemoteFile("tokenizer/added_tokens.json")).toBe(true);
    expect(isSupportedRemoteFile("transformer/diffusion_pytorch_model.safetensors")).toBe(true);
    expect(isSupportedRemoteFile("transformer/config.pb")).toBe(false);
    expect(isSupportedRemoteFile("diffusion_pytorch_model.safetensors")).toBe(false);
  });
});
