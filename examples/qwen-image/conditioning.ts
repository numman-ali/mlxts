import { type DiffusionSnapshotManifest, loadDiffusionSnapshotManifest } from "@mlxts/diffusion";
import { loadPretrainedTokenizer, loadQwen2_5VLTextEncoder } from "@mlxts/transformers";

import { encodeQwenImagePrompt } from "./conditioning-runtime";
import type {
  LoadQwenImagePromptConditionerOptions,
  QwenImagePromptConditioner,
  QwenImagePromptConditionerComponents,
  QwenImagePromptConditioning,
  QwenImagePromptConditioningOptions,
} from "./conditioning-types";

export type {
  LoadQwenImagePromptConditionerOptions,
  QwenImagePromptConditioner,
  QwenImagePromptConditionerComponents,
  QwenImagePromptConditioning,
  QwenImagePromptConditioningOptions,
  QwenImageQwen2_5VLTextEncoder,
  QwenImageQwen2_5VLTextModel,
  QwenImageTextModelOutput,
} from "./conditioning-types";

class ExampleQwenImagePromptConditioner implements QwenImagePromptConditioner {
  #disposed = false;

  readonly pipelineKind = "qwen-image";

  constructor(private readonly components: QwenImagePromptConditionerComponents) {}

  encodePrompt(options: QwenImagePromptConditioningOptions): QwenImagePromptConditioning {
    this.assertOpen();
    return encodeQwenImagePrompt(this.components, options);
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
      throw new Error("QwenImagePromptConditioner has been disposed.");
    }
  }
}

function componentDirectory(
  manifest: DiffusionSnapshotManifest,
  name: "tokenizer" | "text_encoder",
): string {
  const component = manifest.components.find((candidate) => candidate.name === name);
  if (component?.directory === undefined || !component.enabled) {
    throw new Error(`Qwen-Image snapshot is missing enabled ${name} component.`);
  }
  return component.directory;
}

/** Create a prompt conditioner from already loaded tokenizer and Qwen2.5-VL components. */
export function createQwenImagePromptConditioner(
  components: QwenImagePromptConditionerComponents,
): QwenImagePromptConditioner {
  return new ExampleQwenImagePromptConditioner(components);
}

/** Load Qwen2.5-VL prompt conditioning components from a local Diffusers Qwen-Image snapshot. */
export async function loadQwenImagePromptConditionerFromSnapshot(
  snapshotDirectory: string,
  options: LoadQwenImagePromptConditionerOptions = {},
): Promise<QwenImagePromptConditioner> {
  const manifest = await loadDiffusionSnapshotManifest(snapshotDirectory);
  if (manifest.modelIndex.kind !== "qwen-image") {
    throw new Error(
      `Qwen-Image prompt conditioning is not implemented for ${manifest.modelIndex.kind}.`,
    );
  }
  const tokenizerDirectory = componentDirectory(manifest, "tokenizer");
  const textEncoderDirectory = componentDirectory(manifest, "text_encoder");
  const tokenizer = await loadPretrainedTokenizer(tokenizerDirectory, options.tokenizer);
  const textEncoder = await loadQwen2_5VLTextEncoder(textEncoderDirectory, options.textEncoder);
  return createQwenImagePromptConditioner({
    tokenizer,
    textEncoder,
  });
}
