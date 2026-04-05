import {
  downloadFileToCacheDir,
  getHFHubCachePath,
  getRepoFolderName,
  listFiles,
  modelInfo,
} from "@huggingface/hub";
import { existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from "fs";
import { join, resolve } from "path";

import type {
  LoadSourceOptions,
  ModelArtifacts,
  ResolvedSnapshot,
  SnapshotFile,
  SnapshotInspection,
  TokenizerArtifacts,
} from "./types";

const DEFAULT_REVISION = "main";
const HIDDEN_DIRECTORIES = new Set([".cache", ".git"]);
const HIDDEN_FILES = new Set([".DS_Store"]);
const COMMIT_HASH = /^[0-9a-f]{40}$/;

type RemoteSnapshotFile = {
  relativePath: string;
  size: number;
};

function emitProgress(
  options: LoadSourceOptions,
  event: Parameters<NonNullable<LoadSourceOptions["onProgress"]>>[0],
): void {
  options.onProgress?.(event);
}

function expandHome(path: string): string {
  if (path === "~") {
    return Bun.env.HOME ?? process.cwd();
  }
  if (path.startsWith("~/")) {
    return join(Bun.env.HOME ?? process.cwd(), path.slice(2));
  }
  return path;
}

function resolveExistingLocalDirectory(source: string): string | null {
  const resolved = resolve(expandHome(source));
  if (!existsSync(resolved)) {
    return null;
  }
  if (!statSync(resolved).isDirectory()) {
    throw new Error(`resolvePretrainedSource: local path "${resolved}" must be a directory.`);
  }
  return resolved;
}

function looksLikeLocalPath(source: string): boolean {
  return (
    source.startsWith("/") ||
    source.startsWith("./") ||
    source.startsWith("../") ||
    source === "." ||
    source === ".." ||
    source.startsWith("~/")
  );
}

function readJsonFile(path: string | undefined): Record<string, unknown> {
  if (path === undefined || !existsSync(path)) {
    return {};
  }

  const payload: unknown = JSON.parse(readFileSync(path, "utf8"));
  if (typeof payload !== "object" || payload === null || Array.isArray(payload)) {
    throw new Error(`inspectSnapshot: expected "${path}" to contain a JSON object.`);
  }
  return Object.fromEntries(Object.entries(payload));
}

function scanSnapshotFiles(directory: string): SnapshotFile[] {
  const resolvedDirectory = resolve(directory);
  const files: SnapshotFile[] = [];

  function walk(currentDirectory: string): void {
    const entries = readdirSync(currentDirectory, { withFileTypes: true });
    for (const entry of entries) {
      if (entry.isDirectory()) {
        if (HIDDEN_DIRECTORIES.has(entry.name)) {
          continue;
        }
        walk(join(currentDirectory, entry.name));
        continue;
      }

      if (HIDDEN_FILES.has(entry.name)) {
        continue;
      }

      const fullPath = join(currentDirectory, entry.name);
      const stats = statSync(fullPath);
      files.push({
        relativePath: fullPath.slice(resolvedDirectory.length + 1),
        localPath: fullPath,
        size: stats.size,
      });
    }
  }

  walk(resolvedDirectory);
  files.sort((left, right) => left.relativePath.localeCompare(right.relativePath));
  return files;
}

function snapshotTotals(files: readonly SnapshotFile[]): number {
  return files.reduce((total, file) => total + file.size, 0);
}

function hfAccessTokenPath(): string | null {
  const home = Bun.env.HOME;
  if (home === undefined || home === "") {
    return null;
  }
  return join(home, ".cache", "huggingface", "token");
}

function resolveAccessToken(explicit: string | undefined): string | undefined {
  const candidate =
    explicit ??
    Bun.env.HF_TOKEN ??
    Bun.env.HF_ACCESS_TOKEN ??
    Bun.env.HUGGING_FACE_HUB_TOKEN ??
    Bun.env.HUGGINGFACE_HUB_TOKEN;
  if (candidate !== undefined && candidate.trim() !== "") {
    return candidate.trim();
  }

  const tokenPath = hfAccessTokenPath();
  if (tokenPath !== null && existsSync(tokenPath)) {
    const token = readFileSync(tokenPath, "utf8").trim();
    if (token !== "") {
      return token;
    }
  }

  return undefined;
}

function repoFolder(cacheDir: string, repoId: string): string {
  return join(cacheDir, getRepoFolderName({ name: repoId, type: "model" }));
}

function requestedRevision(options: LoadSourceOptions): string {
  return options.revision ?? DEFAULT_REVISION;
}

function resolveCachedSnapshotDirectory(
  repoId: string,
  cacheDir: string,
  revision: string,
): string | null {
  const directory = repoFolder(cacheDir, repoId);
  const resolvedRevision = COMMIT_HASH.test(revision)
    ? revision
    : existsSync(join(directory, "refs", revision))
      ? readFileSync(join(directory, "refs", revision), "utf8").trim()
      : "";
  if (resolvedRevision === "") {
    return null;
  }

  const snapshotDirectory = join(directory, "snapshots", resolvedRevision);
  return existsSync(snapshotDirectory) && statSync(snapshotDirectory).isDirectory()
    ? snapshotDirectory
    : null;
}

function writeRevisionRef(
  repoId: string,
  cacheDir: string,
  revision: string,
  resolved: string,
): void {
  if (revision === resolved) {
    return;
  }

  const refPath = join(repoFolder(cacheDir, repoId), "refs", revision);
  mkdirSync(join(repoFolder(cacheDir, repoId), "refs"), { recursive: true });
  writeFileSync(refPath, `${resolved}\n`, "utf8");
}

function cachedHubSnapshot(
  source: string,
  cacheDir: string,
  revision: string,
): ResolvedSnapshot | null {
  const cachedDirectory = resolveCachedSnapshotDirectory(source, cacheDir, revision);
  if (cachedDirectory === null) {
    return null;
  }

  const files = scanSnapshotFiles(cachedDirectory);
  const resolvedRevision = cachedDirectory.split("/").at(-1);
  return {
    source: "hub",
    repoId: source,
    requestedRevision: revision,
    directory: cachedDirectory,
    files,
    totalBytes: snapshotTotals(files),
    ...(resolvedRevision === undefined ? {} : { resolvedRevision }),
  };
}

async function listRemoteSnapshotFiles(
  source: string,
  resolvedRevision: string,
  accessToken: string | undefined,
): Promise<RemoteSnapshotFile[]> {
  const remoteFiles: RemoteSnapshotFile[] = [];
  for await (const entry of listFiles({
    repo: source,
    recursive: true,
    revision: resolvedRevision,
    ...(accessToken === undefined ? {} : { accessToken }),
  })) {
    if (entry.type !== "file") {
      continue;
    }
    remoteFiles.push({ relativePath: entry.path, size: entry.size });
  }
  remoteFiles.sort((left, right) => left.relativePath.localeCompare(right.relativePath));
  if (remoteFiles.length === 0) {
    throw new Error(
      `resolvePretrainedSource: repo ${source}@${resolvedRevision} did not contain files.`,
    );
  }
  return remoteFiles;
}

async function downloadRemoteSnapshotFiles(
  source: string,
  cacheDir: string,
  resolvedRevision: string,
  accessToken: string | undefined,
  remoteFiles: readonly RemoteSnapshotFile[],
  options: LoadSourceOptions,
): Promise<void> {
  const totalFiles = remoteFiles.length;
  const totalBytes = remoteFiles.reduce((total, file) => total + file.size, 0);
  let completedFiles = 0;
  let completedBytes = 0;

  for (const [index, file] of remoteFiles.entries()) {
    const snapshotPath = join(
      repoFolder(cacheDir, source),
      "snapshots",
      resolvedRevision,
      file.relativePath,
    );
    const alreadyCached = existsSync(snapshotPath);

    emitProgress(options, {
      stage: "download",
      status: alreadyCached ? "cached" : "start",
      repoId: source,
      relativePath: file.relativePath,
      size: file.size,
      index: index + 1,
      totalFiles,
      completedFiles,
      completedBytes,
      totalBytes,
    });

    await downloadFileToCacheDir({
      repo: source,
      path: file.relativePath,
      revision: resolvedRevision,
      cacheDir,
      ...(accessToken === undefined ? {} : { accessToken }),
    });

    completedFiles += 1;
    completedBytes += file.size;

    emitProgress(options, {
      stage: "download",
      status: "complete",
      repoId: source,
      relativePath: file.relativePath,
      size: file.size,
      index: index + 1,
      totalFiles,
      completedFiles,
      completedBytes,
      totalBytes,
    });
  }
}

async function resolveRemoteSnapshot(
  source: string,
  options: LoadSourceOptions,
): Promise<ResolvedSnapshot> {
  const cacheDir = resolve(expandHome(options.cacheDir ?? getHFHubCachePath()));
  const revision = requestedRevision(options);
  const accessToken = resolveAccessToken(options.accessToken);

  if (options.localFilesOnly === true) {
    const snapshot = cachedHubSnapshot(source, cacheDir, revision);
    if (snapshot === null) {
      throw new Error(
        `resolvePretrainedSource: no cached snapshot for ${source}@${revision} and localFilesOnly was requested.`,
      );
    }
    return snapshot;
  }

  const info = await modelInfo({
    name: source,
    revision,
    additionalFields: ["sha"],
    ...(accessToken === undefined ? {} : { accessToken }),
  });
  const resolvedRevision = info.sha;
  if (typeof resolvedRevision !== "string" || resolvedRevision === "") {
    throw new Error(`resolvePretrainedSource: model ${source} did not resolve to a commit hash.`);
  }

  writeRevisionRef(source, cacheDir, revision, resolvedRevision);
  const remoteFiles = await listRemoteSnapshotFiles(source, resolvedRevision, accessToken);
  await downloadRemoteSnapshotFiles(
    source,
    cacheDir,
    resolvedRevision,
    accessToken,
    remoteFiles,
    options,
  );

  const directory = join(repoFolder(cacheDir, source), "snapshots", resolvedRevision);
  const files = scanSnapshotFiles(directory);
  return {
    source: "hub",
    repoId: source,
    requestedRevision: revision,
    resolvedRevision,
    directory,
    files,
    totalBytes: snapshotTotals(files),
  };
}

/** Resolve a local directory or ensure a Hugging Face model repo is present in the local cache. */
export async function resolvePretrainedSnapshot(
  source: string,
  options: LoadSourceOptions = {},
): Promise<ResolvedSnapshot> {
  emitProgress(options, { stage: "resolve", status: "start", source });

  const localDirectory = resolveExistingLocalDirectory(source);
  if (localDirectory !== null) {
    const files = scanSnapshotFiles(localDirectory);
    const snapshot: ResolvedSnapshot = {
      source: "local",
      directory: localDirectory,
      files,
      totalBytes: snapshotTotals(files),
    };
    emitProgress(options, {
      stage: "resolve",
      status: "complete",
      sourceKind: snapshot.source,
      directory: snapshot.directory,
      fileCount: snapshot.files.length,
      totalBytes: snapshot.totalBytes,
    });
    return snapshot;
  }

  if (looksLikeLocalPath(source)) {
    throw new Error(
      `resolvePretrainedSource: local path "${resolve(expandHome(source))}" does not exist.`,
    );
  }

  if (!source.includes("/")) {
    throw new Error(
      `resolvePretrainedSource: "${source}" is neither a local directory nor a Hugging Face model id.`,
    );
  }

  const snapshot = await resolveRemoteSnapshot(source, options);
  emitProgress(options, {
    stage: "resolve",
    status: "complete",
    sourceKind: snapshot.source,
    directory: snapshot.directory,
    fileCount: snapshot.files.length,
    totalBytes: snapshot.totalBytes,
    ...(snapshot.repoId === undefined ? {} : { repoId: snapshot.repoId }),
    ...(snapshot.resolvedRevision === undefined
      ? {}
      : { resolvedRevision: snapshot.resolvedRevision }),
  });
  return snapshot;
}

/** Resolve a local directory or a Hub model id to a concrete local directory path. */
export async function resolvePretrainedSource(
  source: string,
  options: LoadSourceOptions = {},
): Promise<string> {
  const snapshot = await resolvePretrainedSnapshot(source, options);
  return snapshot.directory;
}

function findSnapshotFile(snapshot: ResolvedSnapshot, relativePath: string): string | undefined {
  return snapshot.files.find((file) => file.relativePath === relativePath)?.localPath;
}

function collectModelArtifacts(snapshot: ResolvedSnapshot): ModelArtifacts {
  const artifacts: ModelArtifacts = {
    safetensorPaths: snapshot.files
      .filter((file) => file.relativePath.endsWith(".safetensors"))
      .map((file) => file.localPath)
      .sort((left, right) => left.localeCompare(right)),
  };
  const configPath = findSnapshotFile(snapshot, "config.json");
  if (configPath !== undefined) {
    artifacts.configPath = configPath;
  }
  const generationConfigPath = findSnapshotFile(snapshot, "generation_config.json");
  if (generationConfigPath !== undefined) {
    artifacts.generationConfigPath = generationConfigPath;
  }
  const processorConfigPath = findSnapshotFile(snapshot, "processor_config.json");
  if (processorConfigPath !== undefined) {
    artifacts.processorConfigPath = processorConfigPath;
  }
  const chatTemplatePath = findSnapshotFile(snapshot, "chat_template.jinja");
  if (chatTemplatePath !== undefined) {
    artifacts.chatTemplatePath = chatTemplatePath;
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

/** Inspect a resolved snapshot and parse its standard model/tokenizer sidecars. */
export function inspectSnapshot(snapshot: ResolvedSnapshot): SnapshotInspection {
  const model = collectModelArtifacts(snapshot);
  const tokenizer = collectTokenizerArtifacts(snapshot);

  return {
    snapshot,
    model,
    tokenizer,
    config: readJsonFile(model.configPath),
    generationConfig: readJsonFile(model.generationConfigPath),
    processorConfig: readJsonFile(model.processorConfigPath),
    tokenizerConfig: readJsonFile(tokenizer.tokenizerConfigPath),
    specialTokensMap: readJsonFile(tokenizer.specialTokensMapPath),
    safetensorsIndex: readJsonFile(model.safetensorsIndexPath),
  };
}
