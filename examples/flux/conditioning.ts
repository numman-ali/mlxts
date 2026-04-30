import { type DiffusionSnapshotManifest, loadDiffusionSnapshotManifest } from "@mlxts/diffusion";
import { loadCLIP, loadSentencePiece } from "@mlxts/tokenizers";
import { loadCLIPTextModel, loadT5EncoderModel } from "@mlxts/transformers";

import { encodeFluxPrompt } from "./conditioning-runtime";
import type {
  FluxPromptConditioner,
  FluxPromptConditionerComponents,
  FluxPromptConditioning,
  FluxPromptConditioningOptions,
  LoadFluxPromptConditionerOptions,
} from "./conditioning-types";

export type {
  FluxCLIPTextEncoder,
  FluxPrompt,
  FluxPromptConditioner,
  FluxPromptConditionerComponents,
  FluxPromptConditioning,
  FluxPromptConditioningOptions,
  FluxT5TextEncoder,
  LoadFluxPromptConditionerOptions,
} from "./conditioning-types";

class ExampleFluxPromptConditioner implements FluxPromptConditioner {
  #disposed = false;

  readonly pipelineKind = "flux";

  constructor(private readonly components: FluxPromptConditionerComponents) {}

  encodePrompt(options: FluxPromptConditioningOptions): FluxPromptConditioning {
    this.assertOpen();
    return encodeFluxPrompt(this.components, options);
  }

  [Symbol.dispose](): void {
    if (this.#disposed) {
      return;
    }
    this.#disposed = true;
    this.components.textEncoder[Symbol.dispose]();
    this.components.textEncoder2[Symbol.dispose]();
  }

  private assertOpen(): void {
    if (this.#disposed) {
      throw new Error("FluxPromptConditioner has been disposed.");
    }
  }
}

function componentDirectory(
  manifest: DiffusionSnapshotManifest,
  name: "tokenizer" | "tokenizer_2" | "text_encoder" | "text_encoder_2",
): string {
  const component = manifest.components.find((candidate) => candidate.name === name);
  if (component?.directory === undefined || !component.enabled) {
    throw new Error(`FLUX snapshot is missing enabled ${name} component.`);
  }
  return component.directory;
}

/** Create a prompt conditioner from already loaded CLIP and T5 components. */
export function createFluxPromptConditioner(
  components: FluxPromptConditionerComponents,
): FluxPromptConditioner {
  return new ExampleFluxPromptConditioner(components);
}

/** Load CLIP, T5, and tokenizers from a local Diffusers FLUX snapshot. */
export async function loadFluxPromptConditionerFromSnapshot(
  snapshotDirectory: string,
  options: LoadFluxPromptConditionerOptions = {},
): Promise<FluxPromptConditioner> {
  const manifest = await loadDiffusionSnapshotManifest(snapshotDirectory);
  if (manifest.modelIndex.kind !== "flux") {
    throw new Error(`FLUX prompt conditioning is not implemented for ${manifest.modelIndex.kind}.`);
  }
  const tokenizer = loadCLIP(componentDirectory(manifest, "tokenizer"));
  const textEncoder = await loadCLIPTextModel(
    componentDirectory(manifest, "text_encoder"),
    options.textEncoder,
  );
  try {
    const tokenizer2 = loadSentencePiece(componentDirectory(manifest, "tokenizer_2"));
    const textEncoder2 = await loadT5EncoderModel(
      componentDirectory(manifest, "text_encoder_2"),
      options.textEncoder2,
    );
    return createFluxPromptConditioner({
      tokenizer,
      textEncoder,
      tokenizer2,
      textEncoder2,
    });
  } catch (error) {
    textEncoder[Symbol.dispose]();
    throw error;
  }
}
