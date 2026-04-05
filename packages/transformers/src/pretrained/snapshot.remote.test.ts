import { afterEach, describe, expect, mock, test } from "bun:test";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

import { inspectSnapshot, resolvePretrainedSnapshot } from "./snapshot";
import type { PretrainedLoadProgressEvent } from "./types";

const actualHub = await import("@huggingface/hub");
const tempRoots: string[] = [];
const originalHome = Bun.env.HOME;

function createTempDir(prefix: string): string {
  const directory = mkdtempSync(join(tmpdir(), prefix));
  tempRoots.push(directory);
  return directory;
}

function repoFolderName(repoId: string): string {
  return `models--${repoId.replaceAll("/", "--")}`;
}

function writeSnapshotFile(
  cacheDir: string,
  repoId: string,
  revision: string,
  relativePath: string,
  content: string | Uint8Array,
): void {
  const filePath = join(cacheDir, repoFolderName(repoId), "snapshots", revision, relativePath);
  mkdirSync(join(filePath, ".."), { recursive: true });
  writeFileSync(filePath, content);
}

afterEach(() => {
  mock.module("@huggingface/hub", () => actualHub);
  Bun.env.HOME = originalHome;
  while (tempRoots.length > 0) {
    const directory = tempRoots.pop();
    if (directory !== undefined) {
      rmSync(directory, { recursive: true, force: true });
    }
  }
});

describe("resolvePretrainedSnapshot remote resolution", () => {
  test("uses a cached Hub snapshot when localFilesOnly is set", async () => {
    const cacheDir = createTempDir("mlxts-transformers-hf-cache-");
    const repoId = "google/gemma-4-E2B-it";
    const revision = "main";
    const resolvedRevision = "1234567890abcdef1234567890abcdef12345678";
    const refPath = join(cacheDir, repoFolderName(repoId), "refs", revision);
    mkdirSync(join(refPath, ".."), { recursive: true });
    writeFileSync(refPath, `${resolvedRevision}\n`);
    writeSnapshotFile(
      cacheDir,
      repoId,
      resolvedRevision,
      "config.json",
      JSON.stringify({ model_type: "gemma4" }),
    );
    writeSnapshotFile(
      cacheDir,
      repoId,
      resolvedRevision,
      "model.safetensors",
      new Uint8Array([1, 2, 3]),
    );

    mock.module("@huggingface/hub", () => ({
      ...actualHub,
      getHFHubCachePath: () => cacheDir,
      getRepoFolderName: ({ name }: { name: string }) => repoFolderName(name),
    }));

    const snapshot = await resolvePretrainedSnapshot(repoId, { localFilesOnly: true });

    expect(snapshot.source).toBe("hub");
    expect(snapshot.repoId).toBe(repoId);
    expect(snapshot.resolvedRevision).toBe(resolvedRevision);
    expect(snapshot.files.map((file) => file.relativePath)).toEqual([
      "config.json",
      "model.safetensors",
    ]);
  });

  test("rejects missing cached snapshots when localFilesOnly is set", async () => {
    const cacheDir = createTempDir("mlxts-transformers-hf-cache-missing-");

    mock.module("@huggingface/hub", () => ({
      ...actualHub,
      getHFHubCachePath: () => cacheDir,
      getRepoFolderName: ({ name }: { name: string }) => repoFolderName(name),
    }));

    await expect(
      resolvePretrainedSnapshot("google/gemma-4-E2B-it", { localFilesOnly: true }),
    ).rejects.toThrow(
      "resolvePretrainedSource: no cached snapshot for google/gemma-4-E2B-it@main and localFilesOnly was requested.",
    );
  });

  test("downloads a Hub snapshot, reads the Hugging Face token file, and emits progress", async () => {
    const homeDir = createTempDir("mlxts-transformers-home-");
    const cacheDir = join(homeDir, "custom-hf-cache");
    Bun.env.HOME = homeDir;
    mkdirSync(join(homeDir, ".cache", "huggingface"), { recursive: true });
    writeFileSync(join(homeDir, ".cache", "huggingface", "token"), "hf_token_from_file\n");

    const repoId = "google/gemma-4-E2B-it";
    const resolvedRevision = "fedcba0987654321fedcba0987654321fedcba09";
    const filePayloads = new Map<string, string | Uint8Array>([
      ["config.json", JSON.stringify({ model_type: "gemma4" })],
      ["tokenizer_config.json", JSON.stringify({ chat_template: "{{ messages[0]['content'] }}" })],
      ["model.safetensors", new Uint8Array([1, 2, 3, 4])],
    ]);
    const cachedConfig = filePayloads.get("config.json");
    if (cachedConfig === undefined) {
      throw new Error("Expected cached config payload");
    }
    writeSnapshotFile(cacheDir, repoId, resolvedRevision, "config.json", cachedConfig);

    const accessTokens: string[] = [];
    mock.module("@huggingface/hub", () => ({
      ...actualHub,
      getHFHubCachePath: () => cacheDir,
      getRepoFolderName: ({ name }: { name: string }) => repoFolderName(name),
      modelInfo: async ({ accessToken }: { accessToken?: string }) => {
        if (accessToken !== undefined) {
          accessTokens.push(accessToken);
        }
        return { sha: resolvedRevision };
      },
      listFiles: async function* () {
        yield { type: "file", path: "config.json", size: 24 };
        yield { type: "file", path: "tokenizer_config.json", size: 56 };
        yield { type: "file", path: "model.safetensors", size: 4 };
      },
      downloadFileToCacheDir: async ({
        repo,
        path,
        revision,
        cacheDir: targetCacheDir,
      }: {
        repo: string;
        path: string;
        revision: string;
        cacheDir: string;
      }) => {
        const payload = filePayloads.get(path);
        if (payload === undefined) {
          throw new Error(`Unexpected file request for ${path}`);
        }
        writeSnapshotFile(targetCacheDir, repo, revision, path, payload);
      },
    }));

    const events: PretrainedLoadProgressEvent[] = [];
    const snapshot = await resolvePretrainedSnapshot(repoId, {
      onProgress: (event) => {
        events.push(event);
      },
      cacheDir: "~/custom-hf-cache",
    });

    expect(accessTokens).toEqual(["hf_token_from_file"]);
    expect(snapshot.source).toBe("hub");
    expect(snapshot.directory).toBe(
      join(homeDir, "custom-hf-cache", repoFolderName(repoId), "snapshots", resolvedRevision),
    );
    expect(existsSync(join(snapshot.directory, "tokenizer_config.json"))).toBe(true);
    expect(inspectSnapshot(snapshot).tokenizer.tokenizerConfigPath).toBe(
      join(snapshot.directory, "tokenizer_config.json"),
    );

    const downloadStatuses = events
      .filter(
        (event): event is Extract<PretrainedLoadProgressEvent, { stage: "download" }> =>
          event.stage === "download",
      )
      .map((event) => `${event.relativePath}:${event.status}`);
    expect(downloadStatuses).toEqual([
      "config.json:cached",
      "config.json:complete",
      "model.safetensors:start",
      "model.safetensors:complete",
      "tokenizer_config.json:start",
      "tokenizer_config.json:complete",
    ]);

    const resolveComplete = events.find(
      (
        event,
      ): event is Extract<PretrainedLoadProgressEvent, { stage: "resolve"; status: "complete" }> =>
        event.stage === "resolve" && event.status === "complete",
    );
    expect(resolveComplete).toBeDefined();
    expect(resolveComplete?.directory).toBe(snapshot.directory);
    expect(resolveComplete?.repoId).toBe(repoId);
    expect(resolveComplete?.resolvedRevision).toBe(resolvedRevision);

    const revisionRef = readFileSync(
      join(homeDir, "custom-hf-cache", repoFolderName(repoId), "refs", "main"),
      "utf8",
    ).trim();
    expect(revisionRef).toBe(resolvedRevision);
  });
});
