import { type DiffusionSnapshotManifest, loadDiffusionSnapshotManifest } from "@mlxts/diffusion";
import {
  loadInteractionProfile,
  loadPretrainedTokenizer,
  loadQwen3CausalLM,
} from "@mlxts/transformers";

import { encodeZImagePrompt } from "./conditioning-runtime";
import type {
  LoadZImagePromptConditionerOptions,
  ZImagePromptConditioner,
  ZImagePromptConditionerComponents,
  ZImagePromptConditioning,
  ZImagePromptConditioningOptions,
} from "./conditioning-types";

export type {
  LoadZImagePromptConditionerOptions,
  ZImagePromptConditioner,
  ZImagePromptConditionerComponents,
  ZImagePromptConditioning,
  ZImagePromptConditioningOptions,
  ZImageQwen3TextEncoder,
  ZImageQwen3TextModel,
  ZImageTextModelOutput,
} from "./conditioning-types";

class ExampleZImagePromptConditioner implements ZImagePromptConditioner {
  #disposed = false;

  readonly pipelineKind = "z-image";

  constructor(private readonly components: ZImagePromptConditionerComponents) {}

  encodePrompt(options: ZImagePromptConditioningOptions): ZImagePromptConditioning {
    this.assertOpen();
    return encodeZImagePrompt(this.components, options);
  }

  [Symbol.dispose](): void {
    if (this.#disposed) {
      return;
    }
    this.#disposed = true;
    this.components.textEncoder[Symbol.dispose]();
  }

  private assertOpen(): void {
    if (this.#disposed) {
      throw new Error("ZImagePromptConditioner has been disposed.");
    }
  }
}

function componentDirectory(
  manifest: DiffusionSnapshotManifest,
  name: "tokenizer" | "text_encoder",
): string {
  const component = manifest.components.find((candidate) => candidate.name === name);
  if (component?.directory === undefined || !component.enabled) {
    throw new Error(`Z-Image snapshot is missing enabled ${name} component.`);
  }
  return component.directory;
}

/** Create a prompt conditioner from already loaded tokenizer and Qwen3 components. */
export function createZImagePromptConditioner(
  components: ZImagePromptConditionerComponents,
): ZImagePromptConditioner {
  return new ExampleZImagePromptConditioner(components);
}

/** Load Qwen3 prompt conditioning components from a local Diffusers Z-Image snapshot. */
export async function loadZImagePromptConditionerFromSnapshot(
  snapshotDirectory: string,
  options: LoadZImagePromptConditionerOptions = {},
): Promise<ZImagePromptConditioner> {
  const manifest = await loadDiffusionSnapshotManifest(snapshotDirectory);
  if (manifest.modelIndex.kind !== "z-image") {
    throw new Error(
      `Z-Image prompt conditioning is not implemented for ${manifest.modelIndex.kind}.`,
    );
  }
  const tokenizerDirectory = componentDirectory(manifest, "tokenizer");
  const textEncoderDirectory = componentDirectory(manifest, "text_encoder");
  const tokenizer = await loadPretrainedTokenizer(tokenizerDirectory, options.tokenizer);
  const interactionProfile = await loadInteractionProfile(tokenizerDirectory, options.chatTemplate);
  const textEncoder = await loadQwen3CausalLM(textEncoderDirectory, options.textEncoder);
  return createZImagePromptConditioner({
    tokenizer,
    interactionProfile,
    textEncoder,
  });
}
