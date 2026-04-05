import { iterateSafetensors, loadSafetensors, type MxArray } from "@mlxts/core";
import { inspectSnapshot } from "./inspect";
import type {
  LoadedShardSet,
  ResolvedSnapshot,
  SafetensorWeight,
  SafetensorWeightIterationOptions,
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
        `inspectSnapshot: safetensors index entry "${key}" must point to a shard filename`,
      );
    }
    parsed[key] = shard;
  }
  return { weight_map: parsed };
}

function shardPaths(snapshot: ResolvedSnapshot): string[] {
  const inspection = inspectSnapshot(snapshot);
  const parsedIndex = parseSafetensorIndex(inspection.safetensorsIndex);
  if (parsedIndex !== null) {
    const shardSet = new Set(Object.values(parsedIndex.weight_map));
    return [...shardSet]
      .map((name) => {
        const match = snapshot.files.find((file) => file.relativePath === name);
        if (match === undefined) {
          throw new Error(
            `iterateSafetensorWeights: missing shard "${name}" from resolved snapshot`,
          );
        }
        return match.localPath;
      })
      .sort((left, right) => left.localeCompare(right));
  }

  return inspection.model.safetensorPaths;
}

/** Iterate all safetensor weights shard by shard. */
export async function* iterateSafetensorWeights(
  snapshot: ResolvedSnapshot,
  options: SafetensorWeightIterationOptions = {},
): AsyncGenerator<SafetensorWeight, void, void> {
  for (const path of shardPaths(snapshot)) {
    const iterateOptions =
      options.include === undefined
        ? {}
        : { include: (name: string) => options.include?.(name, path) ?? false };
    for await (const { name, tensor } of iterateSafetensors(path, iterateOptions)) {
      yield { name, tensor, shardPath: path };
    }
  }
}

/** Load every safetensor shard in a resolved snapshot into one tensor map. */
export async function loadSafetensorShardSet(snapshot: ResolvedSnapshot): Promise<LoadedShardSet> {
  const tensors: Record<string, MxArray> = {};
  const metadata: Record<string, string> = {};
  const paths = shardPaths(snapshot);

  for (const path of paths) {
    const loaded = await loadSafetensors(path);
    Object.assign(metadata, loaded.metadata);
    for (const [name, tensor] of Object.entries(loaded.tensors)) {
      tensors[name] = tensor;
    }
  }

  return {
    tensors,
    metadata,
    shardPaths: paths,
  };
}
