import {
  type DiffusionSnapshotManifest,
  loadDiffusionSnapshotManifest,
  loadStableDiffusion3ComponentConfigs,
} from "@mlxts/diffusion";
import { loadCLIP, loadSentencePiece } from "@mlxts/tokenizers";
import { loadCLIPTextModelWithProjection, loadT5EncoderModel } from "@mlxts/transformers";

import { encodeStableDiffusion3Prompt } from "./conditioning-runtime";
import type {
  LoadStableDiffusion3PromptConditionerOptions,
  StableDiffusion3PromptConditioner,
  StableDiffusion3PromptConditionerComponents,
  StableDiffusion3PromptConditioning,
  StableDiffusion3PromptConditioningOptions,
} from "./conditioning-types";

export type {
  LoadStableDiffusion3PromptConditionerOptions,
  StableDiffusion3CLIPTextEncoder,
  StableDiffusion3Prompt,
  StableDiffusion3PromptConditioner,
  StableDiffusion3PromptConditionerComponents,
  StableDiffusion3PromptConditioning,
  StableDiffusion3PromptConditioningOptions,
  StableDiffusion3T5TextEncoder,
} from "./conditioning-types";

class ExampleStableDiffusion3PromptConditioner implements StableDiffusion3PromptConditioner {
  #disposed = false;

  readonly pipelineKind = "stable-diffusion-3";

  constructor(private readonly components: StableDiffusion3PromptConditionerComponents) {}

  encodePrompt(
    options: StableDiffusion3PromptConditioningOptions,
  ): StableDiffusion3PromptConditioning {
    this.assertOpen();
    return encodeStableDiffusion3Prompt(this.components, options);
  }

  [Symbol.dispose](): void {
    if (this.#disposed) {
      return;
    }
    this.#disposed = true;
    this.components.textEncoder[Symbol.dispose]();
    this.components.textEncoder2[Symbol.dispose]();
    this.components.textEncoder3[Symbol.dispose]();
  }

  private assertOpen(): void {
    if (this.#disposed) {
      throw new Error("StableDiffusion3PromptConditioner has been disposed.");
    }
  }
}

function componentDirectory(
  manifest: DiffusionSnapshotManifest,
  name:
    | "tokenizer"
    | "tokenizer_2"
    | "tokenizer_3"
    | "text_encoder"
    | "text_encoder_2"
    | "text_encoder_3",
): string {
  const component = manifest.components.find((candidate) => candidate.name === name);
  if (component?.directory === undefined || !component.enabled) {
    throw new Error(`Stable Diffusion 3 snapshot is missing enabled ${name} component.`);
  }
  return component.directory;
}

/** Create a prompt conditioner from already loaded CLIP and T5 components. */
export function createStableDiffusion3PromptConditioner(
  components: StableDiffusion3PromptConditionerComponents,
): StableDiffusion3PromptConditioner {
  return new ExampleStableDiffusion3PromptConditioner(components);
}

/** Load CLIP, T5, and tokenizer components from a local Diffusers SD3 snapshot. */
export async function loadStableDiffusion3PromptConditionerFromSnapshot(
  snapshotDirectory: string,
  options: LoadStableDiffusion3PromptConditionerOptions = {},
): Promise<StableDiffusion3PromptConditioner> {
  const manifest = await loadDiffusionSnapshotManifest(snapshotDirectory);
  if (manifest.modelIndex.kind !== "stable-diffusion-3") {
    throw new Error(
      `Stable Diffusion 3 prompt conditioning is not implemented for ${manifest.modelIndex.kind}.`,
    );
  }
  const configs = await loadStableDiffusion3ComponentConfigs(manifest);

  const tokenizer = loadCLIP(componentDirectory(manifest, "tokenizer"));
  const textEncoder = await loadCLIPTextModelWithProjection(
    componentDirectory(manifest, "text_encoder"),
    options.textEncoder,
  );
  try {
    const tokenizer2 = loadCLIP(componentDirectory(manifest, "tokenizer_2"));
    const textEncoder2 = await loadCLIPTextModelWithProjection(
      componentDirectory(manifest, "text_encoder_2"),
      options.textEncoder2,
    );
    try {
      const tokenizer3 = loadSentencePiece(componentDirectory(manifest, "tokenizer_3"));
      const textEncoder3 = await loadT5EncoderModel(
        componentDirectory(manifest, "text_encoder_3"),
        options.textEncoder3,
      );
      return createStableDiffusion3PromptConditioner({
        tokenizer,
        tokenizer2,
        tokenizer3,
        textEncoder,
        textEncoder2,
        textEncoder3,
        jointAttentionDim: configs.transformer.jointAttentionDim,
        pooledProjectionDim: configs.transformer.pooledProjectionDim,
      });
    } catch (error) {
      textEncoder2[Symbol.dispose]();
      throw error;
    }
  } catch (error) {
    textEncoder[Symbol.dispose]();
    throw error;
  }
}
