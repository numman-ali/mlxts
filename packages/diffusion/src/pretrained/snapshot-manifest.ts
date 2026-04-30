import { existsSync, readdirSync, statSync } from "fs";
import { join } from "path";

import { DiffusionConfigError } from "../errors";
import {
  type DiffusionModelIndexComponent,
  getDiffusionComponentSpec,
  type ParsedDiffusionModelIndex,
  parseDiffusionModelIndex,
} from "./model-index";
import {
  loadDiffusionSchedulerConfig,
  type ParsedDiffusionSchedulerConfig,
} from "./scheduler-config";

/** Component files discovered under a local diffusion snapshot directory. */
export type DiffusionSnapshotComponent = DiffusionModelIndexComponent & {
  directory?: string;
  metadataPaths: readonly string[];
  weightPaths: readonly string[];
};

/** Local diffusion snapshot inspection without model construction or tensor loading. */
export type DiffusionSnapshotManifest = {
  directory: string;
  modelIndexPath: string;
  modelIndex: ParsedDiffusionModelIndex;
  schedulerConfig: ParsedDiffusionSchedulerConfig;
  components: readonly DiffusionSnapshotComponent[];
};

function expectRecord(value: unknown, context: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new DiffusionConfigError(`${context} must be a JSON object.`);
  }
  return Object.fromEntries(Object.entries(value));
}

function pathIfFile(path: string): string | undefined {
  try {
    if (statSync(path).isFile()) {
      return path;
    }
  } catch {}
  return undefined;
}

function requireDirectory(path: string, context: string): void {
  try {
    if (statSync(path).isDirectory()) {
      return;
    }
  } catch {}
  throw new DiffusionConfigError(`${context} directory is missing: ${path}.`);
}

function listWeightPaths(directory: string): string[] {
  return readdirSync(directory, { withFileTypes: true })
    .filter(
      (entry) =>
        entry.isFile() &&
        (entry.name.endsWith(".safetensors") || entry.name.endsWith(".safetensors.index.json")),
    )
    .map((entry) => join(directory, entry.name))
    .sort((left, right) => left.localeCompare(right));
}

function metadataPathsFor(component: DiffusionModelIndexComponent, directory: string): string[] {
  if (component.role === "scheduler") {
    const path = pathIfFile(join(directory, "scheduler_config.json"));
    return path === undefined ? [] : [path];
  }
  if (component.role === "tokenizer") {
    return [
      "tokenizer.json",
      "tokenizer_config.json",
      "vocab.json",
      "merges.txt",
      "spiece.model",
      "special_tokens_map.json",
    ]
      .map((name) => pathIfFile(join(directory, name)))
      .filter((path): path is string => path !== undefined);
  }
  if (component.role === "image-processor") {
    return ["preprocessor_config.json", "config.json"]
      .map((name) => pathIfFile(join(directory, name)))
      .filter((path): path is string => path !== undefined);
  }
  const path = pathIfFile(join(directory, "config.json"));
  return path === undefined ? [] : [path];
}

function inspectComponent(
  snapshotDirectory: string,
  modelIndex: ParsedDiffusionModelIndex,
  component: DiffusionModelIndexComponent,
): DiffusionSnapshotComponent {
  if (!component.enabled) {
    return { ...component, metadataPaths: [], weightPaths: [] };
  }

  const spec = getDiffusionComponentSpec(modelIndex, component);
  const directory = join(snapshotDirectory, component.subfolder);
  requireDirectory(directory, `model_index.json.${component.name}`);
  const metadataPaths = metadataPathsFor(component, directory);
  const weightPaths = listWeightPaths(directory);

  if (spec.requiresConfig === true && metadataPaths.length === 0) {
    throw new DiffusionConfigError(
      `model_index.json.${component.name} is missing required config metadata.`,
    );
  }
  if (spec.requiresTokenizerFiles === true && metadataPaths.length === 0) {
    throw new DiffusionConfigError(
      `model_index.json.${component.name} is missing tokenizer metadata files.`,
    );
  }
  if (spec.requiresWeights === true && weightPaths.length === 0) {
    throw new DiffusionConfigError(
      `model_index.json.${component.name} is missing safetensors weights.`,
    );
  }

  return {
    ...component,
    directory,
    metadataPaths,
    weightPaths,
  };
}

async function readModelIndex(snapshotDirectory: string): Promise<{
  modelIndexPath: string;
  rawConfig: Record<string, unknown>;
}> {
  const modelIndexPath = join(snapshotDirectory, "model_index.json");
  const file = Bun.file(modelIndexPath);
  if (!(await file.exists())) {
    throw new DiffusionConfigError(`loadDiffusionSnapshotManifest: missing ${modelIndexPath}.`);
  }

  let rawConfig: unknown;
  try {
    rawConfig = await file.json();
  } catch {
    throw new DiffusionConfigError(
      `loadDiffusionSnapshotManifest: ${modelIndexPath} must contain valid JSON.`,
    );
  }
  return { modelIndexPath, rawConfig: expectRecord(rawConfig, "model_index.json") };
}

/** Load a local Diffusers snapshot manifest without constructing models or loading tensors. */
export async function loadDiffusionSnapshotManifest(
  snapshotDirectory: string,
): Promise<DiffusionSnapshotManifest> {
  if (!existsSync(snapshotDirectory)) {
    throw new DiffusionConfigError(`loadDiffusionSnapshotManifest: missing ${snapshotDirectory}.`);
  }
  requireDirectory(snapshotDirectory, "loadDiffusionSnapshotManifest");
  const loaded = await readModelIndex(snapshotDirectory);
  const modelIndex = parseDiffusionModelIndex(loaded.rawConfig);
  const schedulerConfig = await loadDiffusionSchedulerConfig(snapshotDirectory);
  return {
    directory: snapshotDirectory,
    modelIndexPath: loaded.modelIndexPath,
    modelIndex,
    schedulerConfig,
    components: modelIndex.components.map((component) =>
      inspectComponent(snapshotDirectory, modelIndex, component),
    ),
  };
}
