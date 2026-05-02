import {
  type DiffusionSnapshotManifest,
  loadDiffusionSnapshotManifest,
  loadLtx2TextConnectorsFromSnapshot,
} from "@mlxts/diffusion";
import { Module } from "@mlxts/nn";
import { Gemma3TextCausalLM, loadCausalLM, loadPretrainedTokenizer } from "@mlxts/transformers";

import { encodeLtx2Prompt } from "./conditioning-ltx2-runtime";
import type {
  LoadLtx2PromptConditionerOptions,
  Ltx2PromptConditioner,
  Ltx2PromptConditionerComponents,
  Ltx2PromptConditioning,
  Ltx2PromptConditioningOptions,
} from "./conditioning-ltx2-types";

export type {
  LoadLtx2PromptConditionerOptions,
  Ltx2Gemma3TextEncoder,
  Ltx2Gemma3TextModel,
  Ltx2Prompt,
  Ltx2PromptConditioner,
  Ltx2PromptConditionerComponents,
  Ltx2PromptConditioning,
  Ltx2PromptConditioningOptions,
  Ltx2PromptConnector,
} from "./conditioning-ltx2-types";

class ExampleLtx2PromptConditioner implements Ltx2PromptConditioner {
  #disposed = false;

  readonly pipelineKind = "ltx2";

  constructor(private readonly components: Ltx2PromptConditionerComponents) {}

  encodePrompt(options: Ltx2PromptConditioningOptions): Ltx2PromptConditioning {
    this.assertOpen();
    return encodeLtx2Prompt(this.components, options);
  }

  [Symbol.dispose](): void {
    if (this.#disposed) {
      return;
    }
    this.#disposed = true;
    this.components.textEncoder[Symbol.dispose]();
    this.components.connectors[Symbol.dispose]();
  }

  private assertOpen(): void {
    if (this.#disposed) {
      throw new Error("Ltx2PromptConditioner has been disposed.");
    }
  }
}

function componentDirectory(
  manifest: DiffusionSnapshotManifest,
  name: "tokenizer" | "text_encoder" | "connectors",
): string {
  const component = manifest.components.find((candidate) => candidate.name === name);
  if (component?.directory === undefined || !component.enabled) {
    throw new Error(`LTX-2 snapshot is missing enabled ${name} component.`);
  }
  return component.directory;
}

/** Create an LTX-2 prompt conditioner from already loaded Gemma and connector components. */
export function createLtx2PromptConditioner(
  components: Ltx2PromptConditionerComponents,
): Ltx2PromptConditioner {
  return new ExampleLtx2PromptConditioner(components);
}

/** Load Gemma tokenizer/text encoder and LTX-2 connectors from a local Diffusers snapshot. */
export async function loadLtx2PromptConditionerFromSnapshot(
  snapshotDirectory: string,
  options: LoadLtx2PromptConditionerOptions = {},
): Promise<Ltx2PromptConditioner> {
  const manifest = await loadDiffusionSnapshotManifest(snapshotDirectory);
  if (manifest.modelIndex.kind !== "ltx2") {
    throw new Error(
      `LTX-2 prompt conditioning is not implemented for ${manifest.modelIndex.kind}.`,
    );
  }

  const tokenizer = await loadPretrainedTokenizer(
    componentDirectory(manifest, "tokenizer"),
    options.tokenizer,
  );
  const textEncoder = await loadCausalLM(
    componentDirectory(manifest, "text_encoder"),
    options.textEncoder,
  );
  if (!(textEncoder instanceof Gemma3TextCausalLM)) {
    if (textEncoder instanceof Module) {
      textEncoder[Symbol.dispose]();
    }
    throw new Error("LTX-2 prompt conditioning requires a Gemma 3 text encoder.");
  }

  try {
    const connectors = await loadLtx2TextConnectorsFromSnapshot(manifest, options.connectors);
    return createLtx2PromptConditioner({
      tokenizer,
      textEncoder,
      connectors,
    });
  } catch (error) {
    textEncoder[Symbol.dispose]();
    throw error;
  }
}
