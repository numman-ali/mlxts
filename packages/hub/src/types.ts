import type { MxArray } from "@mlxts/core";

/** Supported Hugging Face repository kinds. */
export type HubRepoType = "model" | "dataset";

/** Options for resolving or downloading a snapshot. */
export type SnapshotOptions = {
  revision?: string;
  repoType?: HubRepoType;
  files?: string[];
  include?: string[];
  exclude?: string[];
  token?: string;
  localFilesOnly?: boolean;
  cacheDir?: string;
  forceDownload?: boolean;
};

/** A single file present inside a resolved snapshot. */
export type SnapshotFile = {
  relativePath: string;
  localPath: string;
  size: number;
  etag?: string;
  sha256?: string;
};

/** A resolved local or remote snapshot directory and its files. */
export type ResolvedSnapshot = {
  source: "local" | "hub";
  repoId?: string;
  repoType: HubRepoType;
  requestedRevision?: string;
  resolvedRevision?: string;
  directory: string;
  files: SnapshotFile[];
};

/** Model artifacts discovered inside a snapshot. */
export type ModelArtifacts = {
  configPath?: string;
  generationConfigPath?: string;
  safetensorsIndexPath?: string;
  safetensorPaths: string[];
  ggufPaths: string[];
};

/** Tokenizer artifacts discovered inside a snapshot. */
export type TokenizerArtifacts = {
  tokenizerJsonPath?: string;
  tekkenJsonPath?: string;
  tokenizerModelPath?: string;
  tokenizerConfigPath?: string;
  specialTokensMapPath?: string;
};

/** Fully inspected snapshot metadata and parsed JSON sidecars. */
export type SnapshotInspection = {
  snapshot: ResolvedSnapshot;
  model: ModelArtifacts;
  tokenizer: TokenizerArtifacts;
  config: Record<string, unknown>;
  generationConfig: Record<string, unknown>;
  tokenizerConfig: Record<string, unknown>;
  specialTokensMap: Record<string, unknown>;
  safetensorsIndex: Record<string, unknown>;
};

/** A single weight tensor yielded from a safetensor shard iteration. */
export type SafetensorWeight = {
  name: string;
  tensor: MxArray;
  shardPath: string;
};

/** Optional filtering applied while iterating safetensor weights. */
export type SafetensorWeightIterationOptions = {
  include?: (name: string, shardPath: string) => boolean;
};

/** All tensors and metadata loaded from one resolved shard set. */
export type LoadedShardSet = {
  tensors: Record<string, MxArray>;
  metadata: Record<string, string>;
  shardPaths: string[];
};

/** One GGUF tensor entry from the parsed header. */
export type GgufTensorInfo = {
  name: string;
  dimensions: number[];
  type: number;
  offset: number;
};

/** Parsed GGUF header metadata and tensor table. */
export type GgufHeader = {
  version: number;
  metadata: Record<string, unknown>;
  tensors: GgufTensorInfo[];
};
