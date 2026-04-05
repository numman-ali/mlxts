import { existsSync, readFileSync, statSync, writeFileSync } from "fs";
import { dirname, join, resolve } from "path";
import { fetchFile, fetchJson, resolveFileUrl } from "./http";
import {
  defaultCacheDir,
  ensureDirectory,
  localSnapshotFiles,
  refsDir,
  snapshotDir,
} from "./paths";
import { shouldIncludePath } from "./patterns";
import type { HubRepoType, ResolvedSnapshot, SnapshotFile, SnapshotOptions } from "./types";

type HubSibling = {
  relativePath: string;
  size: number;
  etag?: string;
  sha256?: string;
};

type SnapshotManifest = {
  repoId: string;
  repoType: HubRepoType;
  requestedRevision: string;
  resolvedRevision: string;
  files: SnapshotFile[];
  downloadedAt: string;
};

const DEFAULT_INCLUDE = [
  "config.json",
  "generation_config.json",
  "tokenizer.json",
  "tekken.json",
  "tokenizer.model",
  "tokenizer_config.json",
  "special_tokens_map.json",
  "model*.safetensors",
  "model.safetensors.index.json",
] as const;

function looksLikeLocalDirectory(source: string): boolean {
  return existsSync(source) && statSync(source).isDirectory();
}

function normalizeRemoteSource(source: string): string {
  if (!source.includes("/")) {
    throw new Error(
      `resolveSnapshot: "${source}" is neither a local directory nor a HuggingFace repo id`,
    );
  }
  return source;
}

function snapshotManifestPath(directory: string): string {
  return join(directory, ".mlxts-snapshot.json");
}

function asRecord(value: unknown): Record<string, unknown> | null {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return null;
  }
  return Object.fromEntries(Object.entries(value));
}

function siblingSize(raw: Record<string, unknown>, lfs: Record<string, unknown>): number {
  if (typeof raw.size === "number" && Number.isFinite(raw.size)) {
    return raw.size;
  }
  if (typeof lfs.size === "number" && Number.isFinite(lfs.size)) {
    return lfs.size;
  }
  return 0;
}

function parseSibling(entry: unknown): HubSibling | null {
  const raw = asRecord(entry);
  if (raw === null) {
    return null;
  }
  const relativePath = raw.rfilename;
  if (typeof relativePath !== "string") {
    return null;
  }

  const lfs = asRecord(raw.lfs) ?? {};
  const sibling: HubSibling = { relativePath, size: siblingSize(raw, lfs) };
  const etag =
    typeof raw.etag === "string" ? raw.etag : typeof lfs.oid === "string" ? lfs.oid : undefined;
  if (etag !== undefined) {
    sibling.etag = etag;
  }
  const sha256 = typeof lfs.sha256 === "string" ? lfs.sha256 : undefined;
  if (sha256 !== undefined) {
    sibling.sha256 = sha256;
  }
  return sibling;
}

function parseHubSiblings(payload: Record<string, unknown>): HubSibling[] {
  const siblingsValue = payload.siblings;
  if (!Array.isArray(siblingsValue)) {
    throw new Error("resolveSnapshot: HuggingFace API payload did not include a siblings array");
  }

  const siblings: HubSibling[] = [];
  for (const entry of siblingsValue) {
    const parsed = parseSibling(entry);
    if (parsed !== null) {
      siblings.push(parsed);
    }
  }
  siblings.sort((left, right) => left.relativePath.localeCompare(right.relativePath));
  return siblings;
}

function parseSnapshotFile(entry: unknown): SnapshotFile | null {
  const record = asRecord(entry);
  if (record === null) {
    return null;
  }
  if (
    typeof record.relativePath !== "string" ||
    typeof record.localPath !== "string" ||
    typeof record.size !== "number"
  ) {
    return null;
  }

  const file: SnapshotFile = {
    relativePath: record.relativePath,
    localPath: record.localPath,
    size: record.size,
  };
  if (typeof record.etag === "string") {
    file.etag = record.etag;
  }
  if (typeof record.sha256 === "string") {
    file.sha256 = record.sha256;
  }
  return file;
}

function parseManifest(payload: unknown): SnapshotManifest | null {
  const manifest = asRecord(payload);
  if (manifest === null) {
    return null;
  }
  if (
    typeof manifest.repoId !== "string" ||
    typeof manifest.repoType !== "string" ||
    typeof manifest.requestedRevision !== "string" ||
    typeof manifest.resolvedRevision !== "string" ||
    !Array.isArray(manifest.files) ||
    typeof manifest.downloadedAt !== "string"
  ) {
    return null;
  }

  const files = manifest.files
    .map((entry) => parseSnapshotFile(entry))
    .filter((entry): entry is SnapshotFile => entry !== null);

  return {
    repoId: manifest.repoId,
    repoType: manifest.repoType === "dataset" ? "dataset" : "model",
    requestedRevision: manifest.requestedRevision,
    resolvedRevision: manifest.resolvedRevision,
    files,
    downloadedAt: manifest.downloadedAt,
  };
}

function readManifest(directory: string): SnapshotManifest | null {
  const path = snapshotManifestPath(directory);
  if (!existsSync(path)) {
    return null;
  }

  const payload: unknown = JSON.parse(readFileSync(path, "utf8"));
  return parseManifest(payload);
}

function writeManifest(directory: string, manifest: SnapshotManifest): void {
  writeFileSync(snapshotManifestPath(directory), `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
}

function cachedRevisionPath(
  cacheDir: string,
  repoType: HubRepoType,
  repoId: string,
  revision: string,
): string {
  return join(refsDir(cacheDir, repoType, repoId), `${revision}.txt`);
}

function readPinnedRevision(
  cacheDir: string,
  repoType: HubRepoType,
  repoId: string,
  revision: string,
): string | undefined {
  const path = cachedRevisionPath(cacheDir, repoType, repoId, revision);
  if (!existsSync(path)) {
    return undefined;
  }
  const value = readFileSync(path, "utf8").trim();
  return value === "" ? undefined : value;
}

function writePinnedRevision(
  cacheDir: string,
  repoType: HubRepoType,
  repoId: string,
  revision: string,
  resolvedRevision: string,
): void {
  ensureDirectory(refsDir(cacheDir, repoType, repoId));
  writeFileSync(
    cachedRevisionPath(cacheDir, repoType, repoId, revision),
    `${resolvedRevision}\n`,
    "utf8",
  );
}

function selectedFiles(remoteFiles: HubSibling[], options: SnapshotOptions): HubSibling[] {
  const include = [...(options.include ?? DEFAULT_INCLUDE)];
  const exclude = options.exclude ?? [];
  const files = remoteFiles.filter((entry) =>
    shouldIncludePath(entry.relativePath, options.files, include, exclude),
  );
  if (files.length === 0) {
    throw new Error("resolveSnapshot: no remote files matched the requested filters");
  }
  return files;
}

function snapshotFileFromResponse(
  file: HubSibling,
  localPath: string,
  response: Response,
): SnapshotFile {
  const stats = statSync(localPath);
  const snapshotFile: SnapshotFile = {
    relativePath: file.relativePath,
    localPath,
    size: stats.size,
  };
  const responseEtag = response.headers.get("etag");
  if (responseEtag !== null && responseEtag !== "") {
    snapshotFile.etag = responseEtag;
  } else if (file.etag !== undefined) {
    snapshotFile.etag = file.etag;
  }
  if (file.sha256 !== undefined) {
    snapshotFile.sha256 = file.sha256;
  }
  return snapshotFile;
}

function existingSnapshotFile(file: HubSibling, localPath: string): SnapshotFile {
  const stats = statSync(localPath);
  const snapshotFile: SnapshotFile = {
    relativePath: file.relativePath,
    localPath,
    size: stats.size,
  };
  if (file.etag !== undefined) {
    snapshotFile.etag = file.etag;
  }
  if (file.sha256 !== undefined) {
    snapshotFile.sha256 = file.sha256;
  }
  return snapshotFile;
}

async function ensureDownloadedFiles(
  repoId: string,
  repoType: HubRepoType,
  resolvedRevision: string,
  files: HubSibling[],
  directory: string,
  token: string | undefined,
  forceDownload: boolean,
): Promise<SnapshotFile[]> {
  const downloaded: SnapshotFile[] = [];
  for (const file of files) {
    const localPath = join(directory, file.relativePath);
    ensureDirectory(dirname(localPath));

    if (!existsSync(localPath) || forceDownload) {
      const response = await fetchFile(
        resolveFileUrl(repoId, repoType, resolvedRevision, file.relativePath),
        token,
      );
      const bytes = new Uint8Array(await response.arrayBuffer());
      await Bun.write(localPath, bytes);
      downloaded.push(snapshotFileFromResponse(file, localPath, response));
      continue;
    }

    downloaded.push(existingSnapshotFile(file, localPath));
  }
  downloaded.sort((left, right) => left.relativePath.localeCompare(right.relativePath));
  return downloaded;
}

function localSnapshot(source: string, repoType: HubRepoType): ResolvedSnapshot {
  const directory = resolve(source);
  return {
    source: "local",
    repoType,
    directory,
    files: localSnapshotFiles(directory),
  };
}

function requestedRevision(options: SnapshotOptions): string {
  return options.revision ?? "main";
}

function requestedToken(options: SnapshotOptions): string | undefined {
  return options.token ?? Bun.env.HF_TOKEN;
}

/** Resolve a local directory or materialize a filtered HuggingFace snapshot into cache. */
export async function resolveSnapshot(
  source: string,
  options: SnapshotOptions = {},
): Promise<ResolvedSnapshot> {
  const repoType = options.repoType ?? "model";
  if (looksLikeLocalDirectory(source)) {
    return localSnapshot(source, repoType);
  }

  const repoId = normalizeRemoteSource(source);
  const revision = requestedRevision(options);
  const cacheDir = defaultCacheDir(options.cacheDir);
  const token = requestedToken(options);

  let resolvedRevision = readPinnedRevision(cacheDir, repoType, repoId, revision);
  let siblings: HubSibling[] | undefined;

  if (resolvedRevision === undefined && options.localFilesOnly === true) {
    throw new Error(
      `resolveSnapshot: no cached revision for ${repoId}@${revision} and localFilesOnly was requested`,
    );
  }

  if (resolvedRevision === undefined || options.forceDownload === true) {
    const payload = await fetchJson(repoId, repoType, revision, token);
    const sha = payload.sha;
    if (typeof sha !== "string" || sha === "") {
      throw new Error("resolveSnapshot: HuggingFace API payload did not include a resolved sha");
    }
    resolvedRevision = sha;
    siblings = parseHubSiblings(payload);
    writePinnedRevision(cacheDir, repoType, repoId, revision, resolvedRevision);
  }

  const directory = snapshotDir(cacheDir, repoType, repoId, resolvedRevision);
  ensureDirectory(directory);
  const manifest = readManifest(directory);

  if (siblings === undefined) {
    if (manifest !== null) {
      return {
        source: "hub",
        repoId,
        repoType,
        requestedRevision: revision,
        resolvedRevision,
        directory,
        files: manifest.files,
      };
    }
    if (options.localFilesOnly === true) {
      throw new Error(
        `resolveSnapshot: cached snapshot for ${repoId}@${resolvedRevision} is unavailable`,
      );
    }
    const payload = await fetchJson(repoId, repoType, resolvedRevision, token);
    siblings = parseHubSiblings(payload);
  }

  const files = await ensureDownloadedFiles(
    repoId,
    repoType,
    resolvedRevision,
    selectedFiles(siblings, options),
    directory,
    token,
    options.forceDownload === true,
  );
  writeManifest(directory, {
    repoId,
    repoType,
    requestedRevision: revision,
    resolvedRevision,
    files,
    downloadedAt: new Date().toISOString(),
  });

  return {
    source: "hub",
    repoId,
    repoType,
    requestedRevision: revision,
    resolvedRevision,
    directory,
    files,
  };
}

/** Download a HuggingFace repo snapshot into cache. */
export async function downloadSnapshot(
  repoId: string,
  options: SnapshotOptions = {},
): Promise<ResolvedSnapshot> {
  return resolveSnapshot(repoId, options);
}
