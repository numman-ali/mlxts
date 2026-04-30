/**
 * Pre-load memory estimates for source-backed model serving.
 * @module
 */

import { existsSync, readdirSync, statSync } from "fs";
import { join } from "path";

import { ServeError } from "../errors";
import type { GenerationMemoryUsage } from "../types";

export const MODEL_LOAD_MEMORY_HEADROOM = 1.25;
const HIDDEN_DIRECTORIES = new Set([".cache", ".git"]);

export type ModelLoadMemoryPreflightOptions = {
  modelId: string;
  source: string;
  gpuMemoryUtilization: number;
  memory?: GenerationMemoryUsage | undefined;
};

export type ModelLoadMemoryEstimate = {
  safetensorBytes: number;
  estimatedBytes: number;
};

function formatBytes(bytes: number): string {
  if (bytes >= 1e9) {
    return `${(bytes / 1e9).toFixed(2)} GB`;
  }
  if (bytes >= 1e6) {
    return `${(bytes / 1e6).toFixed(2)} MB`;
  }
  if (bytes >= 1e3) {
    return `${(bytes / 1e3).toFixed(2)} KB`;
  }
  return `${bytes} B`;
}

function localSafetensorBytes(directory: string): number | undefined {
  if (!existsSync(directory)) {
    return undefined;
  }
  let total = 0;

  function safetensorFileSize(path: string, name: string): number {
    if (!name.endsWith(".safetensors")) {
      return 0;
    }
    const stats = statSync(path);
    return stats.isFile() ? stats.size : 0;
  }

  function visit(currentDirectory: string): void {
    for (const entry of readdirSync(currentDirectory, { withFileTypes: true })) {
      const path = join(currentDirectory, entry.name);
      if (!entry.isDirectory()) {
        total += safetensorFileSize(path, entry.name);
        continue;
      }
      if (HIDDEN_DIRECTORIES.has(entry.name)) {
        continue;
      }
      visit(path);
    }
  }

  visit(directory);
  return total;
}

/** Estimate model load memory from local safetensor files plus serving headroom. */
export function estimateModelLoadMemory(source: string): ModelLoadMemoryEstimate | undefined {
  let safetensorBytes: number | undefined;
  try {
    safetensorBytes = localSafetensorBytes(source);
  } catch {
    return undefined;
  }
  if (safetensorBytes === undefined || safetensorBytes === 0) {
    return undefined;
  }
  return {
    safetensorBytes,
    estimatedBytes: Math.ceil(safetensorBytes * MODEL_LOAD_MEMORY_HEADROOM),
  };
}

/** Reject a model load when the local source estimate clearly exceeds the active MLX budget. */
export function requireModelLoadMemoryBudget(options: ModelLoadMemoryPreflightOptions): void {
  const memory = options.memory;
  if (memory === undefined) {
    return;
  }

  const estimate = estimateModelLoadMemory(options.source);
  if (estimate === undefined) {
    return;
  }

  const budgetBytes = Math.floor(memory.limitBytes * options.gpuMemoryUtilization);
  const availableBytes = Math.max(0, budgetBytes - memory.activeBytes);
  if (estimate.estimatedBytes <= availableBytes) {
    return;
  }

  throw new ServeError(
    [
      `Model "${options.modelId}" is estimated to need ${formatBytes(estimate.estimatedBytes)} before loading.`,
      `The active MLX memory budget has ${formatBytes(availableBytes)} available.`,
      `Lower the number of served models, raise gpuMemoryUtilization if appropriate, or serve this model separately.`,
    ].join(" "),
    {
      code: "model_load_memory_exceeded",
      status: 503,
    },
  );
}
