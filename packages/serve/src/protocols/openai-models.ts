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

/** Format served model metadata as an OpenAI-compatible model list. */
export function formatOpenAIModelsResponse(
  models: readonly ServedModelInfo[],
  options: { created: number },
): OpenAIModelsResponse {
  return {
    object: "list",
    data: models.map((model) => ({
      id: model.id,
      object: "model",
      created: model.created ?? options.created,
      owned_by: model.ownedBy ?? "mlxts",
    })),
  };
}
