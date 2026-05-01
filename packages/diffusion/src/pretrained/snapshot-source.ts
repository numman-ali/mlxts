import {
  downloadFileToCacheDir,
  getHFHubCachePath,
  getRepoFolderName,
  listFiles,
  modelInfo,
} from "@huggingface/hub";
import { existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from "fs";
import { dirname, join, resolve } from "path";
import {
  type DiffusionSnapshotFileSelectionOptions,
  type RemoteDiffusionSnapshotFile,
  selectSupportedRemoteFiles,
} from "./snapshot-file-selection";

/** Options for resolving a Diffusers snapshot from a local path or Hugging Face model id. */
export type ResolveDiffusionSnapshotOptions = {
  revision?: string;
  accessToken?: string;
  cacheDir?: string;
  localFilesOnly?: boolean;
  variant?: string;
  onProgress?: (event: DiffusionSnapshotResolveProgressEvent) => void;
};

/** Structured resolver progress for agent-facing image proof commands. */
export type DiffusionSnapshotResolveProgressEvent =
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
    };

/** A concrete file in a resolved Diffusers snapshot. */
export type ResolvedDiffusionSnapshotFile = {
  relativePath: string;
  localPath: string;
  size: number;
};

/** A resolved Diffusers snapshot rooted at a local directory. */
export type ResolvedDiffusionSnapshot = {
  source: "local" | "hub";
  directory: string;
  files: ResolvedDiffusionSnapshotFile[];
  totalBytes: number;
  repoId?: string;
  requestedRevision?: string;
  resolvedRevision?: string;
};

const DEFAULT_REVISION = "main";
const HIDDEN_DIRECTORIES = new Set([".cache", ".git"]);
const COMMIT_HASH = /^[0-9a-f]{40}$/;

function emitProgress(
  options: ResolveDiffusionSnapshotOptions,
  event: DiffusionSnapshotResolveProgressEvent,
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
    throw new Error(`resolveDiffusionSnapshot: local path "${resolved}" must be a directory.`);
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

function scanSnapshotFiles(directory: string): ResolvedDiffusionSnapshotFile[] {
  const resolvedDirectory = resolve(directory);
  const files: ResolvedDiffusionSnapshotFile[] = [];

  function walk(currentDirectory: string): void {
    const entries = readdirSync(currentDirectory, { withFileTypes: true });
    for (const entry of entries) {
      if (entry.isDirectory()) {
        if (!HIDDEN_DIRECTORIES.has(entry.name)) {
          walk(join(currentDirectory, entry.name));
        }
        continue;
      }
      if (entry.name === ".DS_Store") {
        continue;
      }

      const fullPath = join(currentDirectory, entry.name);
      files.push({
        relativePath: fullPath.slice(resolvedDirectory.length + 1),
        localPath: fullPath,
        size: statSync(fullPath).size,
      });
    }
  }

  walk(resolvedDirectory);
  files.sort((left, right) => left.relativePath.localeCompare(right.relativePath));
  return files;
}

function snapshotTotals(files: readonly ResolvedDiffusionSnapshotFile[]): number {
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

function requestedRevision(options: ResolveDiffusionSnapshotOptions): string {
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
  mkdirSync(dirname(refPath), { recursive: true });
  writeFileSync(refPath, `${resolved}\n`, "utf8");
}

function cachedHubSnapshot(
  source: string,
  cacheDir: string,
  revision: string,
): ResolvedDiffusionSnapshot | null {
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
): Promise<RemoteDiffusionSnapshotFile[]> {
  const remoteFiles: RemoteDiffusionSnapshotFile[] = [];
  for await (const entry of listFiles({
    repo: source,
    recursive: true,
    revision: resolvedRevision,
    ...(accessToken === undefined ? {} : { accessToken }),
  })) {
    if (entry.type === "file") {
      remoteFiles.push({ relativePath: entry.path, size: entry.size });
    }
  }
  remoteFiles.sort((left, right) => left.relativePath.localeCompare(right.relativePath));
  if (remoteFiles.length === 0) {
    throw new Error(
      `resolveDiffusionSnapshot: repo ${source}@${resolvedRevision} did not contain files.`,
    );
  }
  return remoteFiles;
}

async function downloadRemoteSnapshotFiles(
  source: string,
  cacheDir: string,
  resolvedRevision: string,
  accessToken: string | undefined,
  remoteFiles: readonly RemoteDiffusionSnapshotFile[],
  options: ResolveDiffusionSnapshotOptions,
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

    const debug = console.debug;
    console.debug = () => {};
    try {
      await downloadFileToCacheDir({
        repo: source,
        path: file.relativePath,
        revision: resolvedRevision,
        cacheDir,
        ...(accessToken === undefined ? {} : { accessToken }),
      });
    } finally {
      console.debug = debug;
    }

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
  options: ResolveDiffusionSnapshotOptions,
): Promise<ResolvedDiffusionSnapshot> {
  const cacheDir = resolve(expandHome(options.cacheDir ?? getHFHubCachePath()));
  const revision = requestedRevision(options);
  const accessToken = resolveAccessToken(options.accessToken);

  if (options.localFilesOnly === true) {
    const snapshot = cachedHubSnapshot(source, cacheDir, revision);
    if (snapshot === null) {
      throw new Error(
        `resolveDiffusionSnapshot: no cached snapshot for ${source}@${revision} and localFilesOnly was requested.`,
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
    throw new Error(`resolveDiffusionSnapshot: model ${source} did not resolve to a commit hash.`);
  }

  writeRevisionRef(source, cacheDir, revision, resolvedRevision);
  const remoteFiles = selectSupportedRemoteFiles(
    source,
    resolvedRevision,
    await listRemoteSnapshotFiles(source, resolvedRevision, accessToken),
    fileSelectionOptions(options),
  );
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

function fileSelectionOptions(
  options: ResolveDiffusionSnapshotOptions,
): DiffusionSnapshotFileSelectionOptions {
  return options.variant === undefined ? {} : { variant: options.variant };
}

/** Resolve a local Diffusers directory or Hugging Face model id to a local snapshot. */
export async function resolveDiffusionSnapshot(
  source: string,
  options: ResolveDiffusionSnapshotOptions = {},
): Promise<ResolvedDiffusionSnapshot> {
  emitProgress(options, { stage: "resolve", status: "start", source });

  const localDirectory = resolveExistingLocalDirectory(source);
  if (localDirectory !== null) {
    const files = scanSnapshotFiles(localDirectory);
    const snapshot: ResolvedDiffusionSnapshot = {
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
      `resolveDiffusionSnapshot: local path "${resolve(expandHome(source))}" does not exist.`,
    );
  }
  if (!source.includes("/")) {
    throw new Error(
      `resolveDiffusionSnapshot: "${source}" is neither a local directory nor a Hugging Face model id.`,
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

/** Resolve a local Diffusers directory or Hub model id to a concrete local directory. */
export async function resolveDiffusionSnapshotDirectory(
  source: string,
  options: ResolveDiffusionSnapshotOptions = {},
): Promise<string> {
  const snapshot = await resolveDiffusionSnapshot(source, options);
  return snapshot.directory;
}
