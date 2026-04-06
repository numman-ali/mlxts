export type RemoteSnapshotFile = {
  relativePath: string;
  size: number;
};

const REMOTE_SUPPORTED_FILE_NAMES = new Set([
  "config.json",
  "generation_config.json",
  "processor_config.json",
  "chat_template.jinja",
  "model.safetensors.index.json",
  "tokenizer.json",
  "tokenizer.model",
  "tekken.json",
  "tokenizer_config.json",
  "special_tokens_map.json",
]);

function isSupportedRemoteSnapshotFile(path: string): boolean {
  return path.endsWith(".safetensors") || REMOTE_SUPPORTED_FILE_NAMES.has(path);
}

export function selectSupportedRemoteSnapshotFiles(
  source: string,
  resolvedRevision: string,
  remoteFiles: readonly RemoteSnapshotFile[],
): RemoteSnapshotFile[] {
  const selected = remoteFiles.filter((file) => isSupportedRemoteSnapshotFile(file.relativePath));
  if (selected.length === 0) {
    throw new Error(
      `resolvePretrainedSource: repo ${source}@${resolvedRevision} did not contain supported safetensors or tokenizer artifacts.`,
    );
  }
  return selected;
}
