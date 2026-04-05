import { existsSync, readFileSync } from "fs";
import type {
  ModelArtifacts,
  ResolvedSnapshot,
  SnapshotInspection,
  TokenizerArtifacts,
} from "./types";

function readJsonFile(path: string | undefined): Record<string, unknown> {
  if (path === undefined || !existsSync(path)) {
    return {};
  }
  const payload: unknown = JSON.parse(readFileSync(path, "utf8"));
  if (typeof payload !== "object" || payload === null || Array.isArray(payload)) {
    throw new Error(`inspectSnapshot: expected "${path}" to contain a JSON object`);
  }
  return Object.fromEntries(Object.entries(payload));
}

function findSnapshotFile(snapshot: ResolvedSnapshot, relativePath: string): string | undefined {
  const match = snapshot.files.find((file) => file.relativePath === relativePath);
  return match?.localPath;
}

function collectModelArtifacts(snapshot: ResolvedSnapshot): ModelArtifacts {
  const safetensorPaths = snapshot.files
    .filter((file) => file.relativePath.endsWith(".safetensors"))
    .map((file) => file.localPath)
    .sort((left, right) => left.localeCompare(right));
  const ggufPaths = snapshot.files
    .filter((file) => file.relativePath.endsWith(".gguf"))
    .map((file) => file.localPath)
    .sort((left, right) => left.localeCompare(right));

  const artifacts: ModelArtifacts = {
    safetensorPaths,
    ggufPaths,
  };
  const configPath = findSnapshotFile(snapshot, "config.json");
  if (configPath !== undefined) {
    artifacts.configPath = configPath;
  }
  const generationConfigPath = findSnapshotFile(snapshot, "generation_config.json");
  if (generationConfigPath !== undefined) {
    artifacts.generationConfigPath = generationConfigPath;
  }
  const safetensorsIndexPath = findSnapshotFile(snapshot, "model.safetensors.index.json");
  if (safetensorsIndexPath !== undefined) {
    artifacts.safetensorsIndexPath = safetensorsIndexPath;
  }
  return artifacts;
}

function collectTokenizerArtifacts(snapshot: ResolvedSnapshot): TokenizerArtifacts {
  const artifacts: TokenizerArtifacts = {};
  const tokenizerJsonPath = findSnapshotFile(snapshot, "tokenizer.json");
  if (tokenizerJsonPath !== undefined) {
    artifacts.tokenizerJsonPath = tokenizerJsonPath;
  }
  const tekkenJsonPath = findSnapshotFile(snapshot, "tekken.json");
  if (tekkenJsonPath !== undefined) {
    artifacts.tekkenJsonPath = tekkenJsonPath;
  }
  const tokenizerModelPath = findSnapshotFile(snapshot, "tokenizer.model");
  if (tokenizerModelPath !== undefined) {
    artifacts.tokenizerModelPath = tokenizerModelPath;
  }
  const tokenizerConfigPath = findSnapshotFile(snapshot, "tokenizer_config.json");
  if (tokenizerConfigPath !== undefined) {
    artifacts.tokenizerConfigPath = tokenizerConfigPath;
  }
  const specialTokensMapPath = findSnapshotFile(snapshot, "special_tokens_map.json");
  if (specialTokensMapPath !== undefined) {
    artifacts.specialTokensMapPath = specialTokensMapPath;
  }
  return artifacts;
}

/** Inspect a resolved snapshot and parse the common model/tokenizer JSON artifacts. */
export function inspectSnapshot(snapshot: ResolvedSnapshot): SnapshotInspection {
  const model = collectModelArtifacts(snapshot);
  const tokenizer = collectTokenizerArtifacts(snapshot);

  return {
    snapshot,
    model,
    tokenizer,
    config: readJsonFile(model.configPath),
    generationConfig: readJsonFile(model.generationConfigPath),
    tokenizerConfig: readJsonFile(tokenizer.tokenizerConfigPath),
    specialTokensMap: readJsonFile(tokenizer.specialTokensMapPath),
    safetensorsIndex: readJsonFile(model.safetensorsIndexPath),
  };
}
