import type { GgufMetadataValue, LoadedGguf, MxArray } from "@mlxts/core";
import { loadGguf, saveGguf } from "@mlxts/core";

export type LoadedGgufCheckpoint = LoadedGguf;
export type GgufCheckpointMetadataValue = GgufMetadataValue;

/** Load a GGUF checkpoint through MLX-native GGUF import. */
export function loadGgufCheckpoint(path: string): LoadedGgufCheckpoint {
  return loadGguf(path);
}

/** Save named tensors to a GGUF checkpoint with optional string metadata. */
export function saveGgufCheckpoint(
  tensors: Record<string, MxArray>,
  path: string,
  metadata: Record<string, string> = {},
): void {
  saveGguf(tensors, path, metadata);
}
