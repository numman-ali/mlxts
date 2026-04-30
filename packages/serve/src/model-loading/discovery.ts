/**
 * Local checkpoint discovery for source-backed model serving.
 * @module
 */

import { FAMILY_REGISTRY } from "@mlxts/transformers";
import { existsSync, readdirSync, readFileSync, statSync } from "fs";
import { basename, join, relative, resolve, sep } from "path";

const HIDDEN_DIRECTORIES = new Set([".cache", ".git", "node_modules"]);

/** Local checkpoint metadata discovered from a model-root scan. */
export type DiscoveredLocalModelSource = {
  source: string;
  modelId: string;
  modelType: string;
  architectures: readonly string[];
  hasVisionConfig: boolean;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function stringArray(value: unknown): readonly string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter((item): item is string => typeof item === "string");
}

function requireDirectory(path: string): string {
  const resolved = resolve(path);
  try {
    if (statSync(resolved).isDirectory()) {
      return resolved;
    }
  } catch {
    throw new Error(`Model root "${path}" does not exist or is not readable.`);
  }
  throw new Error(`Model root "${path}" must be a directory.`);
}

function sortedDirectoryEntries(directory: string) {
  return readdirSync(directory, { withFileTypes: true }).sort((left, right) =>
    left.name.localeCompare(right.name),
  );
}

function shouldVisitDirectory(name: string): boolean {
  return !HIDDEN_DIRECTORIES.has(name) && !name.startsWith(".");
}

function candidateDirectories(root: string): string[] {
  const candidates = [root];
  for (const child of sortedDirectoryEntries(root)) {
    if (!child.isDirectory() || !shouldVisitDirectory(child.name)) {
      continue;
    }
    const childPath = join(root, child.name);
    candidates.push(childPath);
    for (const grandchild of sortedDirectoryEntries(childPath)) {
      if (grandchild.isDirectory() && shouldVisitDirectory(grandchild.name)) {
        candidates.push(join(childPath, grandchild.name));
      }
    }
  }
  return candidates;
}

function hasSafetensorWeights(directory: string): boolean {
  for (const entry of sortedDirectoryEntries(directory)) {
    if (entry.isDirectory() || !entry.name.endsWith(".safetensors")) {
      continue;
    }
    try {
      if (statSync(join(directory, entry.name)).isFile()) {
        return true;
      }
    } catch {}
  }
  return false;
}

function readConfig(directory: string): Record<string, unknown> | undefined {
  const configPath = join(directory, "config.json");
  if (!existsSync(configPath)) {
    return undefined;
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(readFileSync(configPath, "utf8"));
  } catch (error) {
    throw new Error(
      `Model config "${configPath}" is not valid JSON: ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
  }
  if (!isRecord(parsed)) {
    throw new Error(`Model config "${configPath}" must contain a JSON object.`);
  }
  return parsed;
}

function modelIdFor(root: string, directory: string): string {
  const relativePath = relative(root, directory);
  if (relativePath === "") {
    return basename(root);
  }
  return relativePath.split(sep).join("/");
}

function supportedModelType(config: Record<string, unknown>): string | undefined {
  const modelType = typeof config.model_type === "string" ? config.model_type : undefined;
  return modelType !== undefined && FAMILY_REGISTRY.has(modelType) ? modelType : undefined;
}

function discoveredModelSource(
  root: string,
  directory: string,
): DiscoveredLocalModelSource | undefined {
  if (!hasSafetensorWeights(directory)) {
    return undefined;
  }
  const config = readConfig(directory);
  if (config === undefined) {
    return undefined;
  }
  const modelType = supportedModelType(config);
  if (modelType === undefined) {
    return undefined;
  }
  return {
    source: directory,
    modelId: modelIdFor(root, directory),
    modelType,
    architectures: stringArray(config.architectures),
    hasVisionConfig: isRecord(config.vision_config),
  };
}

/** Discover supported local autoregressive checkpoints under root and org/model layouts. */
export function discoverLocalModelSources(root: string): DiscoveredLocalModelSource[] {
  const rootDirectory = requireDirectory(root);
  const discovered: DiscoveredLocalModelSource[] = [];
  for (const directory of candidateDirectories(rootDirectory)) {
    const model = discoveredModelSource(rootDirectory, directory);
    if (model !== undefined) {
      discovered.push(model);
    }
  }
  return discovered;
}
