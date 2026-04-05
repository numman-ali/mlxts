import { existsSync, mkdirSync, readdirSync, statSync } from "fs";
import { join, resolve } from "path";
import type { HubRepoType, SnapshotFile } from "./types";

const DEFAULT_CACHE_ROOT = "~/.cache/mlxts/hub";

function homeDirectory(): string {
  return Bun.env.HOME ?? process.cwd();
}

/** Expand a leading `~` to the active user's home directory. */
export function expandHome(path: string): string {
  if (!path.startsWith("~")) {
    return path;
  }
  if (path === "~") {
    return homeDirectory();
  }
  if (path.startsWith("~/")) {
    return join(homeDirectory(), path.slice(2));
  }
  return path;
}

/** Resolve the default local cache directory for Hugging Face snapshots. */
export function defaultCacheDir(override?: string): string {
  return resolve(expandHome(override ?? Bun.env.MLXTS_HUB_CACHE ?? DEFAULT_CACHE_ROOT));
}

function sanitizeRepoId(repoId: string): string {
  return repoId.replaceAll("/", "--");
}

/** Return the cache root for a specific Hub repo and repo type. */
export function repoCacheDir(cacheDir: string, repoType: HubRepoType, repoId: string): string {
  const prefix = repoType === "dataset" ? "datasets" : "models";
  return join(cacheDir, `${prefix}--${sanitizeRepoId(repoId)}`);
}

/** Return the snapshot directory for a resolved repo revision. */
export function snapshotDir(
  cacheDir: string,
  repoType: HubRepoType,
  repoId: string,
  revision: string,
): string {
  return join(repoCacheDir(cacheDir, repoType, repoId), "snapshots", revision);
}

/** Return the refs directory that stores revision pointers for a cached repo. */
export function refsDir(cacheDir: string, repoType: HubRepoType, repoId: string): string {
  return join(repoCacheDir(cacheDir, repoType, repoId), "refs");
}

/** Create a directory if it does not already exist. */
export function ensureDirectory(path: string): void {
  mkdirSync(path, { recursive: true });
}

/** Walk a local snapshot directory and return every file relative to its root. */
export function localSnapshotFiles(directory: string): SnapshotFile[] {
  const resolved = resolve(directory);
  const files: SnapshotFile[] = [];

  function walk(currentDirectory: string): void {
    const entries = readdirSync(currentDirectory, { withFileTypes: true });
    for (const entry of entries) {
      const fullPath = join(currentDirectory, entry.name);
      if (entry.isDirectory()) {
        walk(fullPath);
        continue;
      }
      const stats = statSync(fullPath);
      files.push({
        relativePath: fullPath.slice(resolved.length + 1),
        localPath: fullPath,
        size: stats.size,
      });
    }
  }

  if (!existsSync(resolved)) {
    throw new Error(`resolveSnapshot: local path "${resolved}" does not exist`);
  }
  if (!statSync(resolved).isDirectory()) {
    throw new Error(`resolveSnapshot: local path "${resolved}" must be a directory`);
  }

  walk(resolved);
  files.sort((left, right) => left.relativePath.localeCompare(right.relativePath));
  return files;
}
