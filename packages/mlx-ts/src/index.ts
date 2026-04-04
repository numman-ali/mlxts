/**
 * Transitional mlx-ts barrel.
 *
 * This package remains as a compatibility shim during the Phase 5 restructure.
 * Delete it once all consumers import from `@mlxts/*` directly.
 *
 * @module mlx-ts
 */

// biome-ignore lint/performance/noReExportAll: Temporary compatibility shim while @mlxts/* packages settle.
export * from "@mlxts/core";
// biome-ignore lint/performance/noReExportAll: Temporary compatibility shim while @mlxts/* packages settle.
export * as nn from "@mlxts/nn";
export {
  crossEntropy,
  Dropout,
  Embedding,
  gelu,
  LayerNorm,
  Linear,
  Module,
  mse,
  relu,
  silu,
} from "@mlxts/nn";
export type { AdamWCheckpoint } from "@mlxts/optimizers";
// biome-ignore lint/performance/noReExportAll: Temporary compatibility shim while @mlxts/* packages settle.
export * as optimizers from "@mlxts/optimizers";
export { Adam, AdamW, SGD } from "@mlxts/optimizers";
