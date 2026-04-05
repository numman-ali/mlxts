import { iterateSafetensors, type MxArray } from "@mlxts/core";

import { inspectSnapshot } from "./snapshot";
import type {
  ResolvedSnapshot,
  SafetensorWeightIterationOptions,
  SnapshotInspection,
} from "./types";

/** Parsed weight-map metadata for a sharded safetensors snapshot. */
export type SafetensorIndex = {
  weight_map: Record<string, string>;
};

/** Parse a safetensors shard index payload if it has a valid weight map. */
export function parseSafetensorIndex(value: Record<string, unknown>): SafetensorIndex | null {
  const weightMap = value.weight_map;
  if (typeof weightMap !== "object" || weightMap === null || Array.isArray(weightMap)) {
    return null;
  }

  const parsed: Record<string, string> = {};
  for (const [key, shard] of Object.entries(weightMap)) {
    if (typeof shard !== "string") {
      throw new Error(
        `inspectSnapshot: safetensors index entry "${key}" must point to a shard filename.`,
      );
    }
    parsed[key] = shard;
  }
  return { weight_map: parsed };
}

function shardPathsFromInspection(
  snapshot: ResolvedSnapshot,
  inspection: SnapshotInspection,
): string[] {
  const parsedIndex = parseSafetensorIndex(inspection.safetensorsIndex);
  if (parsedIndex !== null) {
    const shardSet = new Set(Object.values(parsedIndex.weight_map));
    return [...shardSet]
      .map((name) => {
        const match = snapshot.files.find((file) => file.relativePath === name);
        if (match === undefined) {
          throw new Error(
            `iterateSafetensorWeights: missing shard "${name}" from resolved snapshot.`,
          );
        }
        return match.localPath;
      })
      .sort((left, right) => left.localeCompare(right));
  }

  return inspection.model.safetensorPaths;
}

/** Return the concrete safetensor shard paths for a resolved snapshot. */
export function listSafetensorShardPaths(snapshot: ResolvedSnapshot): string[] {
  return shardPathsFromInspection(snapshot, inspectSnapshot(snapshot));
}

/** Iterate all safetensor weights shard by shard. */
export async function* iterateSafetensorWeights(
  snapshot: ResolvedSnapshot,
  options: SafetensorWeightIterationOptions = {},
): AsyncGenerator<{ name: string; tensor: MxArray; shardPath: string }, void, void> {
  for (const shardPath of listSafetensorShardPaths(snapshot)) {
    const iterateOptions =
      options.include === undefined
        ? {}
        : { include: (name: string) => options.include?.(name, shardPath) ?? false };
    for await (const { name, tensor } of iterateSafetensors(shardPath, iterateOptions)) {
      yield { name, tensor, shardPath };
    }
  }
}
