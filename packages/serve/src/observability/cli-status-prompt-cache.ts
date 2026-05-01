/**
 * Prompt-prefix cache formatting for the finite status CLI.
 * @module
 */

import type { ServeInfoResponse } from "../http/route-info";

function toon(value: string | number | boolean | null): string {
  return typeof value === "string" ? JSON.stringify(value) : String(value);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

/** Format the compact prompt-prefix cache status line. */
export function formatDefaultPromptPrefixCache(info: ServeInfoResponse): string[] {
  const cache = info.prompt_prefix_cache;
  if (cache === null) {
    return ["prompt_prefix_cache: null"];
  }
  return [
    `prompt_prefix_cache: ${toon(
      `snapshots=${cache.total_retained_snapshots}/token_blocks=${cache.total_token_blocks}`,
    )}`,
  ];
}

/** Format full prompt-prefix cache retention details for the status CLI. */
export function formatPromptPrefixCache(info: ServeInfoResponse): string[] {
  const cache = info.prompt_prefix_cache;
  if (cache === null) {
    return ["prompt_prefix_cache: null"];
  }
  return [
    "prompt_prefix_cache:",
    `  retained_snapshots: ${toon(cache.total_retained_snapshots)}`,
    `  retained_snapshot_bytes: ${toon(cache.total_retained_snapshot_bytes)}`,
    `  indexed_block_hashes: ${toon(cache.total_indexed_block_hashes)}`,
    `  token_blocks: ${toon(cache.total_token_blocks)}`,
    `  token_block_references: ${toon(cache.total_token_block_references)}`,
    `  unique_token_count: ${toon(cache.total_unique_token_count)}`,
    `  referenced_token_count: ${toon(cache.total_referenced_token_count)}`,
    `prompt_prefix_cache_models[${cache.models.length}]{id,retained_snapshots,retained_snapshot_bytes,indexed_block_hashes,block_size,token_blocks,token_block_references,unique_token_count,referenced_token_count}:`,
    ...cache.models.map(
      (model) =>
        `  ${[
          toon(model.id),
          toon(model.retained_snapshots),
          toon(model.retained_snapshot_bytes),
          toon(model.indexed_block_hashes),
          toon(model.token_blocks.block_size),
          toon(model.token_blocks.block_count),
          toon(model.token_blocks.block_references),
          toon(model.token_blocks.unique_token_count),
          toon(model.token_blocks.referenced_token_count),
        ].join(",")}`,
    ),
  ];
}

/** Validate the optional prompt-prefix cache section from `/info`. */
export function hasPromptPrefixCacheInfo(
  value: unknown,
): value is ServeInfoResponse["prompt_prefix_cache"] {
  if (value === null) {
    return true;
  }
  if (!isRecord(value)) {
    return false;
  }
  if (
    typeof value.total_retained_snapshots !== "number" ||
    typeof value.total_retained_snapshot_bytes !== "number" ||
    typeof value.total_indexed_block_hashes !== "number" ||
    typeof value.total_token_blocks !== "number" ||
    typeof value.total_token_block_references !== "number" ||
    typeof value.total_unique_token_count !== "number" ||
    typeof value.total_referenced_token_count !== "number" ||
    !Array.isArray(value.models)
  ) {
    return false;
  }
  return value.models.every((model) => {
    if (!isRecord(model) || typeof model.id !== "string") {
      return false;
    }
    const blocks = model.token_blocks;
    return (
      typeof model.retained_snapshots === "number" &&
      typeof model.retained_snapshot_bytes === "number" &&
      typeof model.indexed_block_hashes === "number" &&
      isRecord(blocks) &&
      typeof blocks.block_size === "number" &&
      typeof blocks.block_count === "number" &&
      typeof blocks.block_references === "number" &&
      typeof blocks.unique_token_count === "number" &&
      typeof blocks.referenced_token_count === "number"
    );
  });
}
