/**
 * OpenAI-compatible model listing protocol adapter.
 * @module
 */

export type ServedModelInfo = {
  id: string;
  ownedBy?: string;
  created?: number;
};

export type OpenAIModelInfo = {
  id: string;
  object: "model";
  created: number;
  owned_by: string;
};

export type OpenAIModelsResponse = {
  object: "list";
  data: OpenAIModelInfo[];
};

/** Format one served model as OpenAI-compatible model metadata. */
export function formatOpenAIModelResponse(
  model: ServedModelInfo,
  options: { created: number },
): OpenAIModelInfo {
  return {
    id: model.id,
    object: "model",
    created: model.created ?? options.created,
    owned_by: model.ownedBy ?? "mlxts",
  };
}

/** Format served model metadata as an OpenAI-compatible model list. */
export function formatOpenAIModelsResponse(
  models: readonly ServedModelInfo[],
  options: { created: number },
): OpenAIModelsResponse {
  return {
    object: "list",
    data: models.map((model) => formatOpenAIModelResponse(model, options)),
  };
}
