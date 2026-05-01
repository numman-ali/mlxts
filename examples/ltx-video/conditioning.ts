import { type DiffusionSnapshotManifest, loadDiffusionSnapshotManifest } from "@mlxts/diffusion";
import { loadSentencePiece } from "@mlxts/tokenizers";
import { loadT5EncoderModel } from "@mlxts/transformers";

import { encodeLtxVideoPrompt } from "./conditioning-runtime";
import type {
  LoadLtxVideoPromptConditionerOptions,
  LtxVideoPromptConditioner,
  LtxVideoPromptConditionerComponents,
  LtxVideoPromptConditioning,
  LtxVideoPromptConditioningOptions,
} from "./conditioning-types";

export type {
  LoadLtxVideoPromptConditionerOptions,
  LtxVideoPrompt,
  LtxVideoPromptConditioner,
  LtxVideoPromptConditionerComponents,
  LtxVideoPromptConditioning,
  LtxVideoPromptConditioningOptions,
  LtxVideoT5TextEncoder,
} from "./conditioning-types";

class ExampleLtxVideoPromptConditioner implements LtxVideoPromptConditioner {
  #disposed = false;

  readonly pipelineKind = "ltx-video";

  constructor(private readonly components: LtxVideoPromptConditionerComponents) {}

  encodePrompt(options: LtxVideoPromptConditioningOptions): LtxVideoPromptConditioning {
    this.assertOpen();
    return encodeLtxVideoPrompt(this.components, options);
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
      throw new Error("LtxVideoPromptConditioner has been disposed.");
    }
  }
}

function componentDirectory(
  manifest: DiffusionSnapshotManifest,
  name: "tokenizer" | "text_encoder",
): string {
  const component = manifest.components.find((candidate) => candidate.name === name);
  if (component?.directory === undefined || !component.enabled) {
    throw new Error(`LTX-Video snapshot is missing enabled ${name} component.`);
  }
  return component.directory;
}

/** Create a prompt conditioner from already loaded SentencePiece and T5 components. */
export function createLtxVideoPromptConditioner(
  components: LtxVideoPromptConditionerComponents,
): LtxVideoPromptConditioner {
  return new ExampleLtxVideoPromptConditioner(components);
}

/** Load T5 prompt conditioning components from a local Diffusers LTX-Video snapshot. */
export async function loadLtxVideoPromptConditionerFromSnapshot(
  snapshotDirectory: string,
  options: LoadLtxVideoPromptConditionerOptions = {},
): Promise<LtxVideoPromptConditioner> {
  const manifest = await loadDiffusionSnapshotManifest(snapshotDirectory);
  if (manifest.modelIndex.kind !== "ltx-video") {
    throw new Error(
      `LTX-Video prompt conditioning is not implemented for ${manifest.modelIndex.kind}.`,
    );
  }
  const tokenizer = loadSentencePiece(componentDirectory(manifest, "tokenizer"));
  const textEncoder = await loadT5EncoderModel(
    componentDirectory(manifest, "text_encoder"),
    options.textEncoder,
  );
  return createLtxVideoPromptConditioner({
    tokenizer,
    textEncoder,
  });
}
