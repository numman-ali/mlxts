import { type DiffusionSnapshotManifest, loadDiffusionSnapshotManifest } from "@mlxts/diffusion";
import {
  loadInteractionProfile,
  loadPretrainedTokenizer,
  loadQwen3CausalLM,
} from "@mlxts/transformers";

import { encodeFlux2KleinPrompt } from "./conditioning-runtime";
import type {
  Flux2KleinPromptConditioner,
  Flux2KleinPromptConditionerComponents,
  Flux2KleinPromptConditioning,
  Flux2KleinPromptConditioningOptions,
  LoadFlux2KleinPromptConditionerOptions,
} from "./conditioning-types";

export type {
  Flux2KleinPromptConditioner,
  Flux2KleinPromptConditionerComponents,
  Flux2KleinPromptConditioning,
  Flux2KleinPromptConditioningOptions,
  Flux2KleinQwen3TextEncoder,
  Flux2KleinQwen3TextModel,
  Flux2KleinTextModelOutput,
  LoadFlux2KleinPromptConditionerOptions,
} from "./conditioning-types";

class ExampleFlux2KleinPromptConditioner implements Flux2KleinPromptConditioner {
  #disposed = false;

  readonly pipelineKind = "flux2-klein";

  constructor(private readonly components: Flux2KleinPromptConditionerComponents) {}

  encodePrompt(options: Flux2KleinPromptConditioningOptions): Flux2KleinPromptConditioning {
    this.assertOpen();
    return encodeFlux2KleinPrompt(this.components, options);
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
      throw new Error("Flux2KleinPromptConditioner has been disposed.");
    }
  }
}

function componentDirectory(
  manifest: DiffusionSnapshotManifest,
  name: "tokenizer" | "text_encoder",
): string {
  const component = manifest.components.find((candidate) => candidate.name === name);
  if (component?.directory === undefined || !component.enabled) {
    throw new Error(`FLUX.2 Klein snapshot is missing enabled ${name} component.`);
  }
  return component.directory;
}

/** Create a prompt conditioner from already loaded tokenizer and Qwen3 components. */
export function createFlux2KleinPromptConditioner(
  components: Flux2KleinPromptConditionerComponents,
): Flux2KleinPromptConditioner {
  return new ExampleFlux2KleinPromptConditioner(components);
}

/** Load Qwen3 prompt conditioning components from a local Diffusers FLUX.2 Klein snapshot. */
export async function loadFlux2KleinPromptConditionerFromSnapshot(
  snapshotDirectory: string,
  options: LoadFlux2KleinPromptConditionerOptions = {},
): Promise<Flux2KleinPromptConditioner> {
  const manifest = await loadDiffusionSnapshotManifest(snapshotDirectory);
  if (manifest.modelIndex.kind !== "flux2-klein") {
    throw new Error(
      `FLUX.2 Klein prompt conditioning is not implemented for ${manifest.modelIndex.kind}.`,
    );
  }
  const tokenizerDirectory = componentDirectory(manifest, "tokenizer");
  const textEncoderDirectory = componentDirectory(manifest, "text_encoder");
  const tokenizer = await loadPretrainedTokenizer(tokenizerDirectory, options.tokenizer);
  const interactionProfile = await loadInteractionProfile(tokenizerDirectory, options.chatTemplate);
  const textEncoder = await loadQwen3CausalLM(textEncoderDirectory, options.textEncoder);
  return createFlux2KleinPromptConditioner({
    tokenizer,
    interactionProfile,
    textEncoder,
  });
}
