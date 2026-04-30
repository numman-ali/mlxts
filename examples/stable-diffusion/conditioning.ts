import {
  type DiffusionPipelineKind,
  type DiffusionSnapshotManifest,
  loadDiffusionSnapshotManifest,
} from "@mlxts/diffusion";
import { loadCLIP } from "@mlxts/tokenizers";
import { loadCLIPTextModel, loadCLIPTextModelWithProjection } from "@mlxts/transformers";

import { encodeSDPrompt, encodeXLPrompt } from "./conditioning-runtime";
import type {
  LoadStableDiffusionPromptConditionerOptions,
  StableDiffusionPromptConditioner,
  StableDiffusionPromptConditionerComponents,
  StableDiffusionPromptConditioning,
  StableDiffusionPromptConditioningOptions,
} from "./conditioning-types";

export type {
  LoadStableDiffusionPromptConditionerOptions,
  StableDiffusionProjectedTextEncoder,
  StableDiffusionPrompt,
  StableDiffusionPromptConditioner,
  StableDiffusionPromptConditionerComponents,
  StableDiffusionPromptConditioning,
  StableDiffusionPromptConditioningOptions,
  StableDiffusionTextEncoder,
} from "./conditioning-types";

class ExamplePromptConditioner implements StableDiffusionPromptConditioner {
  #disposed = false;

  constructor(private readonly components: StableDiffusionPromptConditionerComponents) {}

  get pipelineKind(): "stable-diffusion" | "stable-diffusion-xl" {
    return this.components.pipelineKind;
  }

  encodePrompt(
    options: StableDiffusionPromptConditioningOptions,
  ): StableDiffusionPromptConditioning {
    this.assertOpen();
    return this.pipelineKind === "stable-diffusion-xl"
      ? encodeXLPrompt(this.components, options)
      : encodeSDPrompt(this.components, options);
  }

  [Symbol.dispose](): void {
    if (this.#disposed) {
      return;
    }
    this.#disposed = true;
    this.components.textEncoder[Symbol.dispose]();
    this.components.textEncoder2?.[Symbol.dispose]();
  }

  private assertOpen(): void {
    if (this.#disposed) {
      throw new Error("StableDiffusionPromptConditioner has been disposed.");
    }
  }
}

function componentDirectory(
  manifest: DiffusionSnapshotManifest,
  name: "tokenizer" | "tokenizer_2" | "text_encoder" | "text_encoder_2",
): string {
  const component = manifest.components.find((candidate) => candidate.name === name);
  if (component?.directory === undefined || !component.enabled) {
    throw new Error(`Stable Diffusion snapshot is missing enabled ${name} component.`);
  }
  return component.directory;
}

function forceZerosForEmptyPrompt(manifest: DiffusionSnapshotManifest): boolean {
  const value = manifest.modelIndex.pipelineConfig.force_zeros_for_empty_prompt;
  return value === undefined ? true : value === true;
}

/** Create a prompt conditioner from already loaded CLIP components. */
export function createStableDiffusionPromptConditioner(
  components: StableDiffusionPromptConditionerComponents,
): StableDiffusionPromptConditioner {
  return new ExamplePromptConditioner(components);
}

/** Load CLIP tokenizers and text encoders from a local Diffusers snapshot. */
export async function loadStableDiffusionPromptConditionerFromSnapshot(
  snapshotDirectory: string,
  options: LoadStableDiffusionPromptConditionerOptions = {},
): Promise<StableDiffusionPromptConditioner> {
  const manifest = await loadDiffusionSnapshotManifest(snapshotDirectory);
  if (
    manifest.modelIndex.kind !== "stable-diffusion" &&
    manifest.modelIndex.kind !== "stable-diffusion-xl"
  ) {
    throw new Error(`Prompt conditioning is not implemented for ${manifest.modelIndex.kind}.`);
  }
  const pipelineKind: Extract<DiffusionPipelineKind, "stable-diffusion" | "stable-diffusion-xl"> =
    manifest.modelIndex.kind;
  const tokenizer = loadCLIP(componentDirectory(manifest, "tokenizer"));
  const textEncoder = await loadCLIPTextModel(
    componentDirectory(manifest, "text_encoder"),
    options.textEncoder,
  );
  if (pipelineKind === "stable-diffusion") {
    return createStableDiffusionPromptConditioner({
      pipelineKind,
      tokenizer,
      textEncoder,
    });
  }
  try {
    const tokenizer2 = loadCLIP(componentDirectory(manifest, "tokenizer_2"));
    const textEncoder2 = await loadCLIPTextModelWithProjection(
      componentDirectory(manifest, "text_encoder_2"),
      options.textEncoder2,
    );
    return createStableDiffusionPromptConditioner({
      pipelineKind,
      tokenizer,
      textEncoder,
      tokenizer2,
      textEncoder2,
      forceZerosForEmptyPrompt: forceZerosForEmptyPrompt(manifest),
    });
  } catch (error) {
    textEncoder[Symbol.dispose]();
    throw error;
  }
}
