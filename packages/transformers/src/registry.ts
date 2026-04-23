/**
 * Explicit family registry for pretrained model dispatch.
 * @module
 */

import { gemmaFamily } from "./families/gemma/config";
import { gemma3TextFamily } from "./families/gemma3/config";
import { gemma4Family, gemma4TextFamily } from "./families/gemma4/config";
import { llamaFamily } from "./families/llama/config";
import { mistralFamily } from "./families/mistral/config";
import { mistral3Family } from "./families/mistral3/config";
import { phiFamily } from "./families/phi/config";
import { qwen3_5Family, qwen3_5TextFamily } from "./families/qwen3_5/config";
import type { FamilyRegistration } from "./types";
import { UnsupportedModelError } from "./types";

const REGISTRATIONS = [
  llamaFamily,
  mistralFamily,
  mistral3Family,
  gemmaFamily,
  gemma3TextFamily,
  gemma4TextFamily,
  gemma4Family,
  phiFamily,
  qwen3_5Family,
  qwen3_5TextFamily,
] as const;

export const FAMILY_REGISTRY: ReadonlyMap<string, FamilyRegistration> = new Map(
  REGISTRATIONS.flatMap((registration) =>
    registration.modelTypes.map((modelType) => [modelType, registration] as const),
  ),
);

/** Resolve a Hugging Face `model_type` into a supported family registration. */
export function resolveFamily(modelType: string): FamilyRegistration {
  const registration = FAMILY_REGISTRY.get(modelType);
  if (registration !== undefined) {
    return registration;
  }
  throw new UnsupportedModelError(
    modelType,
    [...FAMILY_REGISTRY.keys()].sort((left, right) => left.localeCompare(right)),
  );
}
