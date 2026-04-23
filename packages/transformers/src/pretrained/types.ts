import type { TokenizerFormat } from "@mlxts/tokenizers";

/** Options shared by pretrained local-path and Hugging Face Hub resolution. */
export type LoadSourceOptions = {
  revision?: string;
  accessToken?: string;
  cacheDir?: string;
  localFilesOnly?: boolean;
  onProgress?: (event: PretrainedLoadProgressEvent) => void;
};

/** Structured loader progress events for downloads, cache reuse, and model/tokenizer load phases. */
export type PretrainedLoadProgressEvent =
  | {
      stage: "resolve";
      status: "start";
      source: string;
    }
  | {
      stage: "resolve";
      status: "complete";
      sourceKind: "local" | "hub";
      directory: string;
      fileCount: number;
      totalBytes: number;
      repoId?: string;
      resolvedRevision?: string;
    }
  | {
      stage: "download";
      status: "cached" | "start" | "complete";
      repoId: string;
      relativePath: string;
      size: number;
      index: number;
      totalFiles: number;
      completedFiles: number;
      completedBytes: number;
      totalBytes: number;
    }
  | {
      stage: "model";
      status: "weights-start" | "weights-complete";
      shardCount: number;
    }
  | {
      stage: "tokenizer";
      status: "start" | "complete";
      directory: string;
      format?: TokenizerFormat;
    };

/** A concrete file inside a resolved local pretrained snapshot. */
export type SnapshotFile = {
  relativePath: string;
  localPath: string;
  size: number;
};

/** A resolved local pretrained source, whether it started as a local path or a Hub repo id. */
export type ResolvedSnapshot = {
  source: "local" | "hub";
  directory: string;
  files: SnapshotFile[];
  totalBytes: number;
  repoId?: string;
  requestedRevision?: string;
  resolvedRevision?: string;
};

/** Standard model-side artifacts discovered inside a resolved snapshot. */
export type ModelArtifacts = {
  configPath?: string;
  generationConfigPath?: string;
  processorConfigPath?: string;
  preprocessorConfigPath?: string;
  videoPreprocessorConfigPath?: string;
  chatTemplatePath?: string;
  safetensorsIndexPath?: string;
  safetensorPaths: string[];
};

/** Standard tokenizer-side artifacts discovered inside a resolved snapshot. */
export type TokenizerArtifacts = {
  tokenizerJsonPath?: string;
  tekkenJsonPath?: string;
  tokenizerModelPath?: string;
  tokenizerConfigPath?: string;
  specialTokensMapPath?: string;
};

/** Parsed model/tokenizer sidecars and file paths for a resolved snapshot. */
export type SnapshotInspection = {
  snapshot: ResolvedSnapshot;
  model: ModelArtifacts;
  tokenizer: TokenizerArtifacts;
  config: Record<string, unknown>;
  generationConfig: Record<string, unknown>;
  processorConfig: Record<string, unknown>;
  preprocessorConfig: Record<string, unknown>;
  videoPreprocessorConfig: Record<string, unknown>;
  tokenizerConfig: Record<string, unknown>;
  specialTokensMap: Record<string, unknown>;
  safetensorsIndex: Record<string, unknown>;
};

/** Options that control shard-by-shard safetensor iteration for pretrained loading. */
export type SafetensorWeightIterationOptions = {
  include?: (name: string, shardPath: string) => boolean;
};
