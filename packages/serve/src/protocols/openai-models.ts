/**
 * OpenAI-compatible model listing protocol adapter.
 * @module
 */

import { ServeError } from "../errors";

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

/** Parse a model id from the OpenAI model retrieve route path. */
export function parseOpenAIModelIdPath(pathname: string): string {
  try {
    return decodeURIComponent(pathname.slice("/v1/models/".length));
  } catch {
    throw new ServeError("OpenAI models: model id path must be valid URL encoding.", {
      param: "model",
    });
  }
}

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
