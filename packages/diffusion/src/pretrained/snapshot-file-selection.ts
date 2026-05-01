/** A remote file advertised by a Hugging Face Diffusers snapshot. */
export type RemoteDiffusionSnapshotFile = {
  relativePath: string;
  size: number;
};

/** Options for selecting files from a remote Diffusers snapshot. */
export type DiffusionSnapshotFileSelectionOptions = {
  variant?: string;
};

const REMOTE_SUPPORTED_FILE_NAMES = new Set([
  "model_index.json",
  "scheduler_config.json",
  "config.json",
  "preprocessor_config.json",
  "generation_config.json",
  "tokenizer.json",
  "tokenizer_config.json",
  "processor_config.json",
  "special_tokens_map.json",
  "vocab.json",
  "merges.txt",
  "spiece.model",
  "tokenizer.model",
  "chat_template.jinja",
  "added_tokens.json",
]);

const REMOTE_WEIGHT_COMPONENT_FOLDERS = new Set([
  "unet",
  "vae",
  "audio_vae",
  "text_encoder",
  "text_encoder_2",
  "text_encoder_3",
  "transformer",
  "transformer_2",
  "image_encoder",
  "connectors",
  "latent_upsampler",
  "vocoder",
]);

const SAFETENSORS_SUFFIX = ".safetensors";
const SAFETENSORS_INDEX_SUFFIX = ".safetensors.index.json";
const SHARD_SUFFIX = /-\d{5}-of-\d{5}$/;

function fileName(path: string): string {
  return path.split("/").at(-1) ?? path;
}

function isMetadataFile(path: string): boolean {
  return REMOTE_SUPPORTED_FILE_NAMES.has(fileName(path));
}

function isSafetensorsFile(path: string): boolean {
  return path.endsWith(SAFETENSORS_SUFFIX) || path.endsWith(SAFETENSORS_INDEX_SUFFIX);
}

function componentFolder(path: string): string | undefined {
  const [folder, name] = path.split("/");
  if (folder === undefined || name === undefined) {
    return undefined;
  }
  return REMOTE_WEIGHT_COMPONENT_FOLDERS.has(folder) ? folder : undefined;
}

function isComponentWeightFile(path: string): boolean {
  return componentFolder(path) !== undefined && isSafetensorsFile(path);
}

function trimWeightSuffix(name: string): string | undefined {
  if (name.endsWith(SAFETENSORS_INDEX_SUFFIX)) {
    return name.slice(0, -SAFETENSORS_INDEX_SUFFIX.length);
  }
  if (name.endsWith(SAFETENSORS_SUFFIX)) {
    return name.slice(0, -SAFETENSORS_SUFFIX.length).replace(SHARD_SUFFIX, "");
  }
  return undefined;
}

function weightVariant(path: string): string | null {
  const stem = trimWeightSuffix(fileName(path));
  if (stem === undefined) {
    return null;
  }
  const separator = stem.lastIndexOf(".");
  return separator === -1 ? null : stem.slice(separator + 1);
}

function groupWeightsByComponent(
  remoteFiles: readonly RemoteDiffusionSnapshotFile[],
): Map<string, RemoteDiffusionSnapshotFile[]> {
  const groups = new Map<string, RemoteDiffusionSnapshotFile[]>();
  for (const file of remoteFiles) {
    const folder = componentFolder(file.relativePath);
    if (folder === undefined || !isSafetensorsFile(file.relativePath)) {
      continue;
    }
    const group = groups.get(folder) ?? [];
    group.push(file);
    groups.set(folder, group);
  }
  return groups;
}

function groupWeightsByVariant(
  weights: readonly RemoteDiffusionSnapshotFile[],
): Map<string | null, RemoteDiffusionSnapshotFile[]> {
  const groups = new Map<string | null, RemoteDiffusionSnapshotFile[]>();
  for (const file of weights) {
    const variant = weightVariant(file.relativePath);
    const group = groups.get(variant) ?? [];
    group.push(file);
    groups.set(variant, group);
  }
  return groups;
}

function sortedFiles(files: Iterable<RemoteDiffusionSnapshotFile>): RemoteDiffusionSnapshotFile[] {
  return [...files].sort((left, right) => left.relativePath.localeCompare(right.relativePath));
}

function selectComponentWeights(
  component: string,
  weights: readonly RemoteDiffusionSnapshotFile[],
  requestedVariant: string | undefined,
): RemoteDiffusionSnapshotFile[] {
  const byVariant = groupWeightsByVariant(weights);
  if (requestedVariant !== undefined) {
    const requested = byVariant.get(requestedVariant);
    if (requested !== undefined) {
      return sortedFiles(requested);
    }
    const defaultWeights = byVariant.get(null);
    if (defaultWeights !== undefined) {
      return sortedFiles(defaultWeights);
    }
    throw new Error(
      `resolveDiffusionSnapshot: component ${component} has no "${requestedVariant}" or default safetensors variant.`,
    );
  }

  const defaultWeights = byVariant.get(null);
  if (defaultWeights !== undefined) {
    return sortedFiles(defaultWeights);
  }

  if (byVariant.size === 1) {
    const [onlyVariant] = byVariant.values();
    if (onlyVariant !== undefined) {
      return sortedFiles(onlyVariant);
    }
  }

  throw new Error(
    `resolveDiffusionSnapshot: component ${component} has multiple safetensors variants; pass --variant to select one.`,
  );
}

/** Select package-loadable Diffusers files from a remote Hugging Face listing. */
export function selectSupportedRemoteFiles(
  source: string,
  resolvedRevision: string,
  remoteFiles: readonly RemoteDiffusionSnapshotFile[],
  options: DiffusionSnapshotFileSelectionOptions = {},
): RemoteDiffusionSnapshotFile[] {
  const metadata = remoteFiles.filter((file) => isMetadataFile(file.relativePath));
  if (!metadata.some((file) => file.relativePath === "model_index.json")) {
    throw new Error(
      `resolveDiffusionSnapshot: repo ${source}@${resolvedRevision} did not contain model_index.json.`,
    );
  }

  const selectedWeights: RemoteDiffusionSnapshotFile[] = [];
  for (const [component, weights] of groupWeightsByComponent(remoteFiles)) {
    selectedWeights.push(...selectComponentWeights(component, weights, options.variant));
  }
  if (selectedWeights.length === 0) {
    throw new Error(
      `resolveDiffusionSnapshot: repo ${source}@${resolvedRevision} did not contain supported Diffusers component artifacts.`,
    );
  }

  return sortedFiles([...metadata, ...selectedWeights]);
}

export function isSupportedRemoteFile(path: string): boolean {
  return isMetadataFile(path) || isComponentWeightFile(path);
}
