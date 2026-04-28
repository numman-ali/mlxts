/**
 * Cross-family cache layer taxonomy.
 * @module
 */

/** Semantic cache-state kind retained by one decoder layer. */
export type CacheLayerKind = "full" | "sliding" | "linear-recurrent";

/** Map family-native attention layer labels onto the shared cache taxonomy. */
export function cacheLayerKindFromAttentionType(layerType: string): CacheLayerKind | null {
  switch (layerType) {
    case "full_attention":
      return "full";
    case "sliding_attention":
      return "sliding";
    case "linear_attention":
      return "linear-recurrent";
    default:
      return null;
  }
}

/** Convert validated family attention layer labels into cache layer kinds. */
export function cacheLayerKindsFromAttentionTypes(
  layerTypes: readonly string[],
  context: string,
): CacheLayerKind[] {
  return layerTypes.map((layerType, index) => {
    const kind = cacheLayerKindFromAttentionType(layerType);
    if (kind === null) {
      throw new Error(
        `${context}: unsupported attention layer type ${String(layerType)} at ${index}.`,
      );
    }
    return kind;
  });
}

/** Convert per-layer sliding-window settings into cache layer kinds. */
export function cacheLayerKindsFromWindowSizes(
  layerWindowSizes: readonly (number | undefined)[],
): CacheLayerKind[] {
  return layerWindowSizes.map((windowSize) => (windowSize === undefined ? "full" : "sliding"));
}

/** Return repeated cache layer kinds for uniform cache implementations. */
export function repeatedCacheLayerKinds(
  layerCount: number,
  kind: CacheLayerKind,
): CacheLayerKind[] {
  return Array.from({ length: layerCount }, () => kind);
}
