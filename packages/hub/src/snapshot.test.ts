import { afterEach, describe, expect, test } from "bun:test";
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  statSync,
  writeFileSync,
} from "fs";
import { tmpdir } from "os";
import { join } from "path";

import { downloadSnapshot, resolveSnapshot } from "./snapshot";

const tempRoots: string[] = [];

function createTempDir(prefix: string): string {
  const directory = mkdtempSync(join(tmpdir(), prefix));
  tempRoots.push(directory);
  return directory;
}

afterEach(() => {
  while (tempRoots.length > 0) {
    const directory = tempRoots.pop();
    if (directory !== undefined) {
      rmSync(directory, { recursive: true, force: true });
    }
  }
});

function withFetchStub(
  originalFetch: typeof fetch,
  impl: (input: URL | RequestInfo, init?: RequestInit | BunFetchRequestInit) => Promise<Response>,
): typeof fetch {
  return Object.assign(impl, {
    preconnect: originalFetch.preconnect.bind(originalFetch),
  });
}

function requestUrl(input: URL | RequestInfo): string {
  return typeof input === "string" ? input : input instanceof URL ? input.toString() : input.url;
}

function responseForDownloadedSnapshot(url: string): Response {
  if (url.endsWith("/api/models/test-org/test-model/revision/main")) {
    return new Response(
      JSON.stringify({
        sha: "commit-123",
        siblings: [
          { rfilename: "config.json" },
          { rfilename: "tekken.json" },
          { rfilename: "tokenizer.json" },
          { rfilename: "model.safetensors" },
        ],
      }),
      { status: 200 },
    );
  }
  if (url.includes("/resolve/commit-123/config.json")) {
    return new Response(JSON.stringify({ model_type: "llama" }), {
      status: 200,
      headers: { etag: "cfg" },
    });
  }
  if (url.includes("/resolve/commit-123/tokenizer.json")) {
    return new Response(JSON.stringify({ model: { type: "BPE", vocab: {}, merges: [] } }), {
      status: 200,
      headers: { etag: "tok" },
    });
  }
  if (url.includes("/resolve/commit-123/tekken.json")) {
    return new Response(JSON.stringify({ config: { pattern: "\\p{L}+" } }), {
      status: 200,
      headers: { etag: "tekken" },
    });
  }
  if (url.includes("/resolve/commit-123/model.safetensors")) {
    return new Response(new Uint8Array([9, 8, 7]), {
      status: 200,
      headers: { etag: "weights" },
    });
  }
  throw new Error(`Unexpected fetch url: ${url}`);
}

describe("resolveSnapshot", () => {
  test("resolves a local snapshot directory with the default file filter", async () => {
    const directory = createTempDir("mlxts-hub-local-");
    await Bun.write(join(directory, "config.json"), JSON.stringify({ model_type: "llama" }));
    await Bun.write(
      join(directory, "tokenizer.json"),
      JSON.stringify({ model: { type: "BPE", vocab: {}, merges: [] } }),
    );
    await Bun.write(join(directory, "model.safetensors"), new Uint8Array([1, 2, 3]));
    await Bun.write(join(directory, "notes.txt"), "skip me");

    const snapshot = await resolveSnapshot(directory);

    expect(snapshot.source).toBe("local");
    expect(snapshot.files.map((file) => file.relativePath)).toEqual([
      "config.json",
      "model.safetensors",
      "notes.txt",
      "tokenizer.json",
    ]);
  });

  test("downloads and caches a remote snapshot", async () => {
    const cacheDir = createTempDir("mlxts-hub-cache-");
    const originalFetch = globalThis.fetch;
    globalThis.fetch = withFetchStub(originalFetch, async (input, init) => {
      const url = requestUrl(input);
      try {
        return responseForDownloadedSnapshot(url);
      } catch (error) {
        throw new Error(`Unexpected fetch url: ${url} ${JSON.stringify(init ?? {})}`, {
          cause: error,
        });
      }
    });

    try {
      const snapshot = await downloadSnapshot("test-org/test-model", { cacheDir });
      expect(snapshot.resolvedRevision).toBe("commit-123");
      expect(snapshot.files.map((file) => file.relativePath)).toEqual([
        "config.json",
        "model.safetensors",
        "tekken.json",
        "tokenizer.json",
      ]);
      expect(existsSync(join(cacheDir, "models--test-org--test-model", "refs", "main.txt"))).toBe(
        true,
      );
      expect(
        existsSync(
          join(
            cacheDir,
            "models--test-org--test-model",
            "snapshots",
            "commit-123",
            ".mlxts-snapshot.json",
          ),
        ),
      ).toBe(true);

      globalThis.fetch = withFetchStub(originalFetch, async () => {
        throw new Error("cached resolve should not fetch");
      });

      const cached = await resolveSnapshot("test-org/test-model", {
        cacheDir,
        localFilesOnly: true,
      });
      expect(cached.resolvedRevision).toBe("commit-123");
      expect(
        readFileSync(
          join(cacheDir, "models--test-org--test-model", "refs", "main.txt"),
          "utf8",
        ).trim(),
      ).toBe("commit-123");
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test("rejects invalid remote ids and missing local-only cache revisions", async () => {
    const cacheDir = createTempDir("mlxts-hub-local-only-");

    await expect(resolveSnapshot("llama", { cacheDir })).rejects.toThrow(
      "is neither a local directory nor a HuggingFace repo id",
    );
    await expect(
      resolveSnapshot("test-org/test-model", {
        cacheDir,
        localFilesOnly: true,
      }),
    ).rejects.toThrow("no cached revision");
  });

  test("uses cached manifests when a pinned revision exists and ignores malformed manifest entries", async () => {
    const cacheDir = createTempDir("mlxts-hub-manifest-");
    const refsPath = join(cacheDir, "models--test-org--test-model", "refs");
    const snapshotPath = join(cacheDir, "models--test-org--test-model", "snapshots", "commit-123");
    mkdirSync(refsPath, { recursive: true });
    mkdirSync(snapshotPath, { recursive: true });
    writeFileSync(join(refsPath, "main.txt"), "commit-123\n");
    writeFileSync(
      join(snapshotPath, ".mlxts-snapshot.json"),
      `${JSON.stringify(
        {
          repoId: "test-org/test-model",
          repoType: "model",
          requestedRevision: "main",
          resolvedRevision: "commit-123",
          files: [
            null,
            { relativePath: 123, localPath: "/tmp/ignored", size: 1 },
            {
              relativePath: "model.safetensors",
              localPath: "/tmp/model.safetensors",
              size: 3,
              etag: "etag-from-manifest",
              sha256: "sha-from-manifest",
            },
          ],
          downloadedAt: "2026-04-05T00:00:00.000Z",
        },
        null,
        2,
      )}\n`,
    );

    const originalFetch = globalThis.fetch;
    globalThis.fetch = withFetchStub(originalFetch, async () => {
      throw new Error("cached manifest resolve should not fetch");
    });

    try {
      const snapshot = await resolveSnapshot("test-org/test-model", {
        cacheDir,
        localFilesOnly: true,
      });
      expect(snapshot.resolvedRevision).toBe("commit-123");
      expect(snapshot.files).toEqual([
        {
          relativePath: "model.safetensors",
          localPath: "/tmp/model.safetensors",
          size: 3,
          etag: "etag-from-manifest",
          sha256: "sha-from-manifest",
        },
      ]);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test("validates snapshot API payloads and cached snapshot availability", async () => {
    const cacheDir = createTempDir("mlxts-hub-remote-errors-");
    const originalFetch = globalThis.fetch;

    globalThis.fetch = withFetchStub(originalFetch, async (input) => {
      const url = requestUrl(input);
      if (url.endsWith("/api/models/test-org/test-model/revision/main")) {
        return new Response(JSON.stringify({ sha: "commit-1", siblings: {} }), { status: 200 });
      }
      throw new Error(`Unexpected fetch url: ${url}`);
    });
    try {
      await expect(downloadSnapshot("test-org/test-model", { cacheDir })).rejects.toThrow(
        "did not include a siblings array",
      );
    } finally {
      globalThis.fetch = originalFetch;
    }

    globalThis.fetch = withFetchStub(originalFetch, async (input) => {
      const url = requestUrl(input);
      if (url.endsWith("/api/models/test-org/test-model/revision/main")) {
        return new Response(JSON.stringify({ sha: "", siblings: [] }), { status: 200 });
      }
      throw new Error(`Unexpected fetch url: ${url}`);
    });
    try {
      await expect(downloadSnapshot("test-org/test-model", { cacheDir })).rejects.toThrow(
        "did not include a resolved sha",
      );
    } finally {
      globalThis.fetch = originalFetch;
    }

    globalThis.fetch = withFetchStub(originalFetch, async (input) => {
      const url = requestUrl(input);
      if (url.endsWith("/api/models/test-org/test-model/revision/main")) {
        return new Response(
          JSON.stringify({
            sha: "commit-2",
            siblings: [
              null,
              { rfilename: 123 },
              {
                rfilename: "model.safetensors",
                lfs: { oid: "etag-from-lfs", sha256: "sha-from-lfs", size: 3 },
              },
            ],
          }),
          { status: 200 },
        );
      }
      if (url.includes("/resolve/commit-2/model.safetensors")) {
        return new Response(new Uint8Array([1, 2, 3]), { status: 200 });
      }
      throw new Error(`Unexpected fetch url: ${url}`);
    });
    try {
      const snapshot = await downloadSnapshot("test-org/test-model", {
        cacheDir,
        include: ["model*.safetensors"],
      });
      expect(snapshot.files).toEqual([
        {
          relativePath: "model.safetensors",
          localPath: join(
            cacheDir,
            "models--test-org--test-model",
            "snapshots",
            "commit-2",
            "model.safetensors",
          ),
          size: 3,
          etag: "etag-from-lfs",
          sha256: "sha-from-lfs",
        },
      ]);

      await expect(
        downloadSnapshot("test-org/test-model", {
          cacheDir,
          files: ["tokenizer.json"],
          forceDownload: true,
        }),
      ).rejects.toThrow("no remote files matched");
    } finally {
      globalThis.fetch = originalFetch;
    }

    const cachedOnlyDir = createTempDir("mlxts-hub-cached-only-");
    const cachedRefs = join(cachedOnlyDir, "models--test-org--test-model", "refs");
    mkdirSync(cachedRefs, { recursive: true });
    writeFileSync(join(cachedRefs, "main.txt"), "commit-only\n");

    await expect(
      resolveSnapshot("test-org/test-model", {
        cacheDir: cachedOnlyDir,
        localFilesOnly: true,
      }),
    ).rejects.toThrow("cached snapshot for test-org/test-model@commit-only is unavailable");
  });

  test("rebuilds malformed cached manifests from pinned revisions without re-downloading existing files", async () => {
    const cacheDir = createTempDir("mlxts-hub-rebuild-manifest-");
    const refsPath = join(cacheDir, "models--test-org--test-model", "refs");
    const snapshotPath = join(cacheDir, "models--test-org--test-model", "snapshots", "commit-123");
    mkdirSync(refsPath, { recursive: true });
    mkdirSync(snapshotPath, { recursive: true });
    writeFileSync(join(refsPath, "main.txt"), "commit-123\n");
    writeFileSync(join(snapshotPath, "config.json"), JSON.stringify({ model_type: "llama" }));
    writeFileSync(
      join(snapshotPath, "tokenizer.json"),
      JSON.stringify({ model: { type: "BPE", vocab: {}, merges: [] } }),
    );
    writeFileSync(join(snapshotPath, "model.safetensors"), new Uint8Array([1, 2, 3]));

    const originalFetch = globalThis.fetch;
    globalThis.fetch = withFetchStub(originalFetch, async (input) => {
      const url = requestUrl(input);
      if (url.endsWith("/api/models/test-org/test-model/revision/commit-123")) {
        return new Response(
          JSON.stringify({
            siblings: [
              { rfilename: "tokenizer.json", size: 40 },
              { rfilename: "config.json", size: 23 },
              {
                rfilename: "model.safetensors",
                size: 3,
                etag: "weights-direct",
                lfs: { sha256: "sha-direct" },
              },
            ],
          }),
          { status: 200 },
        );
      }
      throw new Error(`Unexpected fetch url: ${url}`);
    });

    try {
      writeFileSync(join(snapshotPath, ".mlxts-snapshot.json"), "[]\n");
      const rebuiltFromArray = await resolveSnapshot("test-org/test-model", { cacheDir });
      expect(rebuiltFromArray.files).toEqual([
        {
          relativePath: "config.json",
          localPath: join(snapshotPath, "config.json"),
          size: statSync(join(snapshotPath, "config.json")).size,
        },
        {
          relativePath: "model.safetensors",
          localPath: join(snapshotPath, "model.safetensors"),
          size: 3,
          etag: "weights-direct",
          sha256: "sha-direct",
        },
        {
          relativePath: "tokenizer.json",
          localPath: join(snapshotPath, "tokenizer.json"),
          size: statSync(join(snapshotPath, "tokenizer.json")).size,
        },
      ]);

      writeFileSync(
        join(snapshotPath, ".mlxts-snapshot.json"),
        `${JSON.stringify({
          repoId: "test-org/test-model",
          repoType: "model",
          requestedRevision: "main",
          resolvedRevision: "commit-123",
          files: [],
        })}\n`,
      );
      const rebuiltFromObject = await resolveSnapshot("test-org/test-model", { cacheDir });
      expect(rebuiltFromObject.files[1]).toEqual({
        relativePath: "model.safetensors",
        localPath: join(snapshotPath, "model.safetensors"),
        size: 3,
        etag: "weights-direct",
        sha256: "sha-direct",
      });
    } finally {
      globalThis.fetch = originalFetch;
    }
  });
});
