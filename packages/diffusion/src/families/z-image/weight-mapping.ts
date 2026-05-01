/**
 * Z-Image transformer Diffusers weight-name mapping.
 * @module
 */

import type { ZImageTransformerConfig } from "./config";

export type ZImageWeightPlan = {
  kind: "direct";
  path: string;
};

const TOP_LEVEL_WEIGHT_PATHS = new Map<string, string>([
  ["x_pad_token", "xPadToken"],
  ["cap_pad_token", "capPadToken"],
  ["t_embedder.mlp.0.weight", "timeEmbedding.linear1.weight"],
  ["t_embedder.mlp.0.bias", "timeEmbedding.linear1.bias"],
  ["t_embedder.mlp.2.weight", "timeEmbedding.linear2.weight"],
  ["t_embedder.mlp.2.bias", "timeEmbedding.linear2.bias"],
  ["cap_embedder.0.weight", "captionEmbedder.norm.weight"],
  ["cap_embedder.1.weight", "captionEmbedder.projection.weight"],
  ["cap_embedder.1.bias", "captionEmbedder.projection.bias"],
]);

const BLOCK_WEIGHT_PATHS = new Map<string, string>([
  ["attention.to_q", "attention.query"],
  ["attention.to_k", "attention.key"],
  ["attention.to_v", "attention.value"],
  ["attention.to_out.0", "attention.projection"],
  ["attention.norm_q", "attention.queryNorm"],
  ["attention.norm_k", "attention.keyNorm"],
  ["feed_forward.w1", "feedForward.w1"],
  ["feed_forward.w2", "feedForward.w2"],
  ["feed_forward.w3", "feedForward.w3"],
  ["attention_norm1", "attentionNorm1"],
  ["attention_norm2", "attentionNorm2"],
  ["ffn_norm1", "ffnNorm1"],
  ["ffn_norm2", "ffnNorm2"],
  ["adaLN_modulation.0", "adaLnModulation"],
]);

function direct(path: string): ZImageWeightPlan {
  return { kind: "direct", path };
}

function geometryIndex(config: ZImageTransformerConfig, key: string): number | null {
  const index = config.patchGeometries.findIndex(
    (geometry) => `${geometry.patchSize}-${geometry.framePatchSize}` === key,
  );
  return index === -1 ? null : index;
}

function zImageGeometryWeightPlan(
  config: ZImageTransformerConfig,
  checkpointName: string,
): ZImageWeightPlan | null {
  const imageMatch = checkpointName.match(/^all_x_embedder\.([^.]+)\.(weight|bias)$/);
  if (imageMatch !== null) {
    const [, key, leaf] = imageMatch;
    if (key === undefined || leaf === undefined) {
      return null;
    }
    const index = geometryIndex(config, key);
    return index === null ? null : direct(`imageEmbedders.${index}.${leaf}`);
  }

  const finalMatch = checkpointName.match(
    /^all_final_layer\.([^.]+)\.(linear|adaLN_modulation\.[01])\.(weight|bias)$/,
  );
  if (finalMatch === null) {
    return null;
  }
  const [, key, local, leaf] = finalMatch;
  if (key === undefined || local === undefined || leaf === undefined) {
    return null;
  }
  const index = geometryIndex(config, key);
  if (index === null) {
    return null;
  }
  const target = local === "linear" ? "output" : "modulation";
  return direct(`finalLayers.${index}.${target}.${leaf}`);
}

function zImageBlockWeightPlan(checkpointName: string): ZImageWeightPlan | null {
  const match = checkpointName.match(
    /^(noise_refiner|context_refiner|layers)\.(\d+)\.(.+)\.(weight|bias)$/,
  );
  if (match === null) {
    return null;
  }
  const [, blockGroup, index, local, leaf] = match;
  if (
    blockGroup === undefined ||
    index === undefined ||
    local === undefined ||
    leaf === undefined
  ) {
    return null;
  }
  const group =
    blockGroup === "noise_refiner"
      ? "noiseRefiner"
      : blockGroup === "context_refiner"
        ? "contextRefiner"
        : "layers";
  const path = BLOCK_WEIGHT_PATHS.get(local);
  return path === undefined ? null : direct(`${group}.${index}.${path}.${leaf}`);
}

export function zImageTransformerWeightPlan(
  config: ZImageTransformerConfig,
  checkpointName: string,
): ZImageWeightPlan | null {
  if (checkpointName.trim() === "" || checkpointName.includes("num_batches_tracked")) {
    return null;
  }

  const geometryPlan = zImageGeometryWeightPlan(config, checkpointName);
  if (geometryPlan !== null) {
    return geometryPlan;
  }

  const blockPlan = zImageBlockWeightPlan(checkpointName);
  if (blockPlan !== null) {
    return blockPlan;
  }

  const path = TOP_LEVEL_WEIGHT_PATHS.get(checkpointName);
  return path === undefined ? null : direct(path);
}

/** Map a Diffusers Z-Image transformer tensor name onto the package parameter tree. */
export function zImageTransformerWeightPath(
  config: ZImageTransformerConfig,
  checkpointName: string,
): string | null {
  return zImageTransformerWeightPlan(config, checkpointName)?.path ?? null;
}
