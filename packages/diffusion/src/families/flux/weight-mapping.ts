/**
 * FLUX.1 transformer Diffusers weight-name mapping.
 * @module
 */

export type DirectWeightPlan = {
  kind: "direct";
  path: string;
};

export type FusedWeightPlan = {
  kind: "fused";
  path: string;
  partIndex: number;
  partCount: number;
};

export type FluxWeightPlan = DirectWeightPlan | FusedWeightPlan;

const DOUBLE_DIRECT_WEIGHT_PATHS = new Map<string, string>([
  ["norm1.linear", "imageModulation.linear"],
  ["norm1_context.linear", "textModulation.linear"],
  ["ff.net.0.proj", "imageFeedForward.linear1"],
  ["ff.net.2", "imageFeedForward.linear2"],
  ["ff_context.net.0.proj", "textFeedForward.linear1"],
  ["ff_context.net.2", "textFeedForward.linear2"],
  ["attn.norm_q", "imageAttention.norm.queryNorm"],
  ["attn.norm_k", "imageAttention.norm.keyNorm"],
  ["attn.norm_added_q", "textAttention.norm.queryNorm"],
  ["attn.norm_added_k", "textAttention.norm.keyNorm"],
  ["attn.to_out.0", "imageAttention.projection"],
  ["attn.to_add_out", "textAttention.projection"],
]);

const DOUBLE_IMAGE_QKV_PARTS = ["attn.to_q", "attn.to_k", "attn.to_v"];
const DOUBLE_TEXT_QKV_PARTS = ["attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"];

const SINGLE_DIRECT_WEIGHT_PATHS = new Map<string, string>([
  ["norm.linear", "modulation.linear"],
  ["attn.norm_q", "norm.queryNorm"],
  ["attn.norm_k", "norm.keyNorm"],
  ["proj_out", "linear2"],
]);

const SINGLE_LINEAR1_PARTS = ["attn.to_q", "attn.to_k", "attn.to_v", "proj_mlp"];

const TOP_LEVEL_WEIGHT_PATHS = new Map<string, string>([
  ["x_embedder.weight", "imageProjection.weight"],
  ["x_embedder.bias", "imageProjection.bias"],
  ["context_embedder.weight", "textProjection.weight"],
  ["context_embedder.bias", "textProjection.bias"],
  ["time_text_embed.timestep_embedder.linear_1.weight", "timeEmbedding.linear1.weight"],
  ["time_text_embed.timestep_embedder.linear_1.bias", "timeEmbedding.linear1.bias"],
  ["time_text_embed.timestep_embedder.linear_2.weight", "timeEmbedding.linear2.weight"],
  ["time_text_embed.timestep_embedder.linear_2.bias", "timeEmbedding.linear2.bias"],
  ["time_text_embed.guidance_embedder.linear_1.weight", "guidanceEmbedding.linear1.weight"],
  ["time_text_embed.guidance_embedder.linear_1.bias", "guidanceEmbedding.linear1.bias"],
  ["time_text_embed.guidance_embedder.linear_2.weight", "guidanceEmbedding.linear2.weight"],
  ["time_text_embed.guidance_embedder.linear_2.bias", "guidanceEmbedding.linear2.bias"],
  ["time_text_embed.text_embedder.linear_1.weight", "vectorEmbedding.linear1.weight"],
  ["time_text_embed.text_embedder.linear_1.bias", "vectorEmbedding.linear1.bias"],
  ["time_text_embed.text_embedder.linear_2.weight", "vectorEmbedding.linear2.weight"],
  ["time_text_embed.text_embedder.linear_2.bias", "vectorEmbedding.linear2.bias"],
  ["norm_out.linear.weight", "finalLayer.modulation.weight"],
  ["norm_out.linear.bias", "finalLayer.modulation.bias"],
  ["proj_out.weight", "finalLayer.projection.weight"],
  ["proj_out.bias", "finalLayer.projection.bias"],
]);

function direct(path: string): DirectWeightPlan {
  return { kind: "direct", path };
}

function fused(path: string, partIndex: number, partCount: number): FusedWeightPlan {
  return { kind: "fused", path, partIndex, partCount };
}

function fusedPartPlan(
  local: string,
  targetPath: string,
  leaf: string,
  parts: readonly string[],
): FusedWeightPlan | null {
  const partIndex = parts.indexOf(local);
  if (partIndex === -1) {
    return null;
  }
  return fused(`${targetPath}.${leaf}`, partIndex, parts.length);
}

function fluxDoubleBlockWeightPlan(checkpointName: string): FluxWeightPlan | null {
  const match = checkpointName.match(/^transformer_blocks\.(\d+)\.(.+)\.(weight|bias)$/);
  if (match === null) {
    return null;
  }
  const [, index, local, leaf] = match;
  if (index === undefined || local === undefined || leaf === undefined) {
    return null;
  }
  const prefix = `doubleBlocks.${index}`;
  const directPath = DOUBLE_DIRECT_WEIGHT_PATHS.get(local);
  if (directPath !== undefined) {
    return direct(`${prefix}.${directPath}.${leaf}`);
  }

  return (
    fusedPartPlan(local, `${prefix}.imageAttention.qkv`, leaf, DOUBLE_IMAGE_QKV_PARTS) ??
    fusedPartPlan(local, `${prefix}.textAttention.qkv`, leaf, DOUBLE_TEXT_QKV_PARTS)
  );
}

function fluxSingleBlockWeightPlan(checkpointName: string): FluxWeightPlan | null {
  const match = checkpointName.match(/^single_transformer_blocks\.(\d+)\.(.+)\.(weight|bias)$/);
  if (match === null) {
    return null;
  }
  const [, index, local, leaf] = match;
  if (index === undefined || local === undefined || leaf === undefined) {
    return null;
  }
  const prefix = `singleBlocks.${index}`;
  const directPath = SINGLE_DIRECT_WEIGHT_PATHS.get(local);
  if (directPath !== undefined) {
    return direct(`${prefix}.${directPath}.${leaf}`);
  }

  return fusedPartPlan(local, `${prefix}.linear1`, leaf, SINGLE_LINEAR1_PARTS);
}

export function fluxTransformerWeightPlan(checkpointName: string): FluxWeightPlan | null {
  if (checkpointName.trim() === "" || checkpointName.includes("num_batches_tracked")) {
    return null;
  }

  const doublePlan = fluxDoubleBlockWeightPlan(checkpointName);
  if (doublePlan !== null) {
    return doublePlan;
  }

  const singlePlan = fluxSingleBlockWeightPlan(checkpointName);
  if (singlePlan !== null) {
    return singlePlan;
  }

  const path = TOP_LEVEL_WEIGHT_PATHS.get(checkpointName);
  return path === undefined ? null : direct(path);
}

/** Map a Diffusers FLUX transformer tensor name onto the package parameter tree. */
export function fluxTransformerWeightPath(checkpointName: string): string | null {
  return fluxTransformerWeightPlan(checkpointName)?.path ?? null;
}
