export type ServeCliModelOption = {
  source: string;
  modelId: string;
};

function requireNonEmptyModelPart(part: string, value: string): string {
  if (value.trim() === "") {
    throw new Error(`Expected --model to include a non-empty ${part}.`);
  }
  return value;
}

/** Parse one repeatable CLI model source entry. */
export function parseModelFlagValue(rawValue: string): ServeCliModelOption {
  const raw = rawValue.trim();
  const separator = raw.indexOf("=");
  if (separator === -1) {
    return {
      source: requireNonEmptyModelPart("source", raw),
      modelId: raw,
    };
  }

  const modelId = requireNonEmptyModelPart("model id", raw.slice(0, separator).trim());
  const source = requireNonEmptyModelPart("source", raw.slice(separator + 1).trim());
  return { source, modelId };
}

/** Require served model ids to be unique within one CLI serve command. */
export function requireDistinctModelIds(models: readonly ServeCliModelOption[]): void {
  const seen = new Set<string>();
  for (const model of models) {
    if (seen.has(model.modelId)) {
      throw new Error(`model id "${model.modelId}" is duplicated.`);
    }
    seen.add(model.modelId);
  }
}

/** Require lazy-pinned models to be part of the resolved serve set. */
export function requirePinnedModelsExist(
  models: readonly ServeCliModelOption[],
  pinnedModels: readonly string[],
): void {
  const servedIds = new Set(models.map((model) => model.modelId));
  for (const modelId of pinnedModels) {
    if (!servedIds.has(modelId)) {
      throw new Error(`Pinned model "${modelId}" is not part of this serve command.`);
    }
  }
}
