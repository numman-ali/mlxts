import { afterEach, describe, expect, mock, test } from "bun:test";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { dirname, join } from "path";

import {
  type DiffusionSnapshotResolveProgressEvent,
  resolveDiffusionSnapshot,
} from "./snapshot-source";

const actualHub = await import("@huggingface/hub");
const tempRoots: string[] = [];

function createTempDir(prefix: string): string {
  const directory = mkdtempSync(join(tmpdir(), prefix));
  tempRoots.push(directory);
  return directory;
}

function repoFolderName(repoId: string): string {
  return `models--${repoId.replaceAll("/", "--")}`;
}

function writeFile(path: string, content: string | Uint8Array): void {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, content);
}

function writeSnapshotFile(
  cacheDir: string,
  repoId: string,
  revision: string,
  relativePath: string,
  content: string | Uint8Array,
): void {
  writeFile(join(cacheDir, repoFolderName(repoId), "snapshots", revision, relativePath), content);
}

afterEach(() => {
  mock.module("@huggingface/hub", () => actualHub);
  while (tempRoots.length > 0) {
    const directory = tempRoots.pop();
    if (directory !== undefined) {
      rmSync(directory, { recursive: true, force: true });
    }
  }
});

describe("resolveDiffusionSnapshot", () => {
  test("scans a local Diffusers directory and ignores hidden files", async () => {
    const snapshotDirectory = createTempDir("mlxts-diffusion-local-");
    writeFile(join(snapshotDirectory, "model_index.json"), "{}");
    writeFile(join(snapshotDirectory, "scheduler", "scheduler_config.json"), "{}");
    writeFile(join(snapshotDirectory, ".cache", "ignored.json"), "{}");
    writeFile(join(snapshotDirectory, ".DS_Store"), "ignored");

    const snapshot = await resolveDiffusionSnapshot(snapshotDirectory);

    expect(snapshot.source).toBe("local");
    expect(snapshot.directory).toBe(snapshotDirectory);
    expect(snapshot.files.map((file) => file.relativePath)).toEqual([
      "model_index.json",
      "scheduler/scheduler_config.json",
    ]);
  });

  test("uses a cached Hub snapshot when localFilesOnly is set", async () => {
    const cacheDir = createTempDir("mlxts-diffusion-hf-cache-");
    const repoId = "runwayml/stable-diffusion-v1-5";
    const resolvedRevision = "1234567890abcdef1234567890abcdef12345678";
    writeFile(join(cacheDir, repoFolderName(repoId), "refs", "main"), `${resolvedRevision}\n`);
    writeSnapshotFile(cacheDir, repoId, resolvedRevision, "model_index.json", "{}");
    writeSnapshotFile(cacheDir, repoId, resolvedRevision, "scheduler/scheduler_config.json", "{}");
    writeSnapshotFile(cacheDir, repoId, resolvedRevision, "unet/model.safetensors", "weights");

    mock.module("@huggingface/hub", () => ({
      ...actualHub,
      getHFHubCachePath: () => cacheDir,
      getRepoFolderName: ({ name }: { name: string }) => repoFolderName(name),
    }));

    const snapshot = await resolveDiffusionSnapshot(repoId, { localFilesOnly: true });

    expect(snapshot.source).toBe("hub");
    expect(snapshot.repoId).toBe(repoId);
    expect(snapshot.resolvedRevision).toBe(resolvedRevision);
    expect(snapshot.files.map((file) => file.relativePath)).toEqual([
      "model_index.json",
      "scheduler/scheduler_config.json",
      "unet/model.safetensors",
    ]);
  });

  test("rejects missing cached snapshots when localFilesOnly is set", async () => {
    const cacheDir = createTempDir("mlxts-diffusion-hf-cache-missing-");

    mock.module("@huggingface/hub", () => ({
      ...actualHub,
      getHFHubCachePath: () => cacheDir,
      getRepoFolderName: ({ name }: { name: string }) => repoFolderName(name),
    }));

    await expect(
      resolveDiffusionSnapshot("runwayml/stable-diffusion-v1-5", { localFilesOnly: true }),
    ).rejects.toThrow("no cached snapshot");
  });

  test("downloads only Diffusers artifacts and emits structured progress", async () => {
    const cacheDir = createTempDir("mlxts-diffusion-hf-download-");
    const repoId = "black-forest-labs/FLUX.1-schnell";
    const resolvedRevision = "fedcba0987654321fedcba0987654321fedcba09";
    const payloads = new Map<string, string | Uint8Array>([
      ["model_index.json", "{}"],
      ["scheduler/scheduler_config.json", "{}"],
      ["text_encoder/config.json", "{}"],
      ["tokenizer/spiece.model", "tokenizer"],
      ["transformer/diffusion_pytorch_model.safetensors", new Uint8Array([1, 2, 3])],
      ["transformer/diffusion_pytorch_model.fp16.safetensors", new Uint8Array([4, 5, 6])],
      ["flux-root.safetensors", new Uint8Array([8])],
      ["README.md", "docs"],
      ["original/consolidated.00.pth", new Uint8Array([9])],
    ]);
    writeSnapshotFile(cacheDir, repoId, resolvedRevision, "model_index.json", "{}");

    const modelInfoTokens: string[] = [];
    mock.module("@huggingface/hub", () => ({
      ...actualHub,
      getHFHubCachePath: () => cacheDir,
      getRepoFolderName: ({ name }: { name: string }) => repoFolderName(name),
      modelInfo: async ({ accessToken }: { accessToken?: string }) => {
        if (accessToken !== undefined) {
          modelInfoTokens.push(accessToken);
        }
        return { sha: resolvedRevision };
      },
      listFiles: async function* () {
        yield { type: "file", path: "README.md", size: 4 };
        yield { type: "file", path: "model_index.json", size: 2 };
        yield { type: "file", path: "scheduler/scheduler_config.json", size: 2 };
        yield { type: "file", path: "text_encoder/config.json", size: 2 };
        yield { type: "file", path: "tokenizer/spiece.model", size: 9 };
        yield { type: "file", path: "flux-root.safetensors", size: 1 };
        yield { type: "file", path: "transformer/diffusion_pytorch_model.safetensors", size: 3 };
        yield {
          type: "file",
          path: "transformer/diffusion_pytorch_model.fp16.safetensors",
          size: 3,
        };
        yield { type: "file", path: "original/consolidated.00.pth", size: 1 };
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
        const payload = payloads.get(path);
        if (payload === undefined) {
          throw new Error(`Unexpected file request for ${path}`);
        }
        console.debug(`Downloading ${path}`);
        writeSnapshotFile(targetCacheDir, repo, revision, path, payload);
      },
    }));

    const events: DiffusionSnapshotResolveProgressEvent[] = [];
    const debugMessages: string[] = [];
    const originalDebug = console.debug;
    console.debug = (...args: unknown[]) => {
      debugMessages.push(args.map(String).join(" "));
    };
    let snapshot: Awaited<ReturnType<typeof resolveDiffusionSnapshot>>;
    try {
      snapshot = await resolveDiffusionSnapshot(repoId, {
        accessToken: "hf_explicit",
        revision: "refs/pr/1",
        variant: "fp16",
        onProgress: (event) => events.push(event),
      });
    } finally {
      console.debug = originalDebug;
    }

    expect(modelInfoTokens).toEqual(["hf_explicit"]);
    expect(debugMessages).toEqual([]);
    expect(readFileSync(join(cacheDir, repoFolderName(repoId), "refs", "refs/pr/1"), "utf8")).toBe(
      `${resolvedRevision}\n`,
    );
    expect(existsSync(join(snapshot.directory, "README.md"))).toBe(false);
    expect(existsSync(join(snapshot.directory, "flux-root.safetensors"))).toBe(false);
    expect(existsSync(join(snapshot.directory, "original", "consolidated.00.pth"))).toBe(false);
    expect(snapshot.files.map((file) => file.relativePath)).toEqual([
      "model_index.json",
      "scheduler/scheduler_config.json",
      "text_encoder/config.json",
      "tokenizer/spiece.model",
      "transformer/diffusion_pytorch_model.fp16.safetensors",
    ]);

    const downloadStatuses = events
      .filter(
        (event): event is Extract<DiffusionSnapshotResolveProgressEvent, { stage: "download" }> =>
          event.stage === "download",
      )
      .map((event) => `${event.relativePath}:${event.status}`);
    expect(downloadStatuses).toEqual([
      "model_index.json:cached",
      "model_index.json:complete",
      "scheduler/scheduler_config.json:start",
      "scheduler/scheduler_config.json:complete",
      "text_encoder/config.json:start",
      "text_encoder/config.json:complete",
      "tokenizer/spiece.model:start",
      "tokenizer/spiece.model:complete",
      "transformer/diffusion_pytorch_model.fp16.safetensors:start",
      "transformer/diffusion_pytorch_model.fp16.safetensors:complete",
    ]);
  });

  test("rejects Hub repos without Diffusers metadata", async () => {
    const cacheDir = createTempDir("mlxts-diffusion-hf-unsupported-");

    mock.module("@huggingface/hub", () => ({
      ...actualHub,
      getHFHubCachePath: () => cacheDir,
      getRepoFolderName: ({ name }: { name: string }) => repoFolderName(name),
      modelInfo: async () => ({ sha: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" }),
      listFiles: async function* () {
        yield { type: "file", path: "config.json", size: 2 };
        yield { type: "file", path: "model.safetensors", size: 3 };
      },
    }));

    await expect(resolveDiffusionSnapshot("example/not-diffusers")).rejects.toThrow(
      "did not contain model_index.json",
    );
  });
});
