import { describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, realpathSync, rmSync, symlinkSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

import { readLocalImageFileBytes, resolveLocalImageRoots } from "./local-image";

async function withTemporaryDirectory<T>(
  name: string,
  work: (directory: string) => Promise<T>,
): Promise<T> {
  const directory = mkdtempSync(join(tmpdir(), `${name}-`));
  try {
    return await work(directory);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
}

describe("local image file transport", () => {
  test("resolves configured roots to canonical directories", async () => {
    await withTemporaryDirectory("mlxts-serve-local-image", async (directory) => {
      expect(resolveLocalImageRoots([directory, directory])).toEqual([realpathSync(directory)]);
    });
  });

  test("validates configured roots before serving requests", async () => {
    await withTemporaryDirectory("mlxts-serve-local-image", async (directory) => {
      writeFileSync(join(directory, "not-root"), new Uint8Array([1]));

      expect(() => resolveLocalImageRoots([" "])).toThrow("non-empty");
      expect(() => resolveLocalImageRoots([join(directory, "not-root")])).toThrow(
        "not a directory",
      );
      await expect(
        readLocalImageFileBytes("pixel.png", {
          localImageRoots: [join(directory, "missing-root")],
          maxBytes: 3,
        }),
      ).rejects.toThrow("existing directories");
    });
  });

  test("reads relative image file ids from configured roots", async () => {
    await withTemporaryDirectory("mlxts-serve-local-image", async (directory) => {
      mkdirSync(join(directory, "nested"));
      writeFileSync(join(directory, "nested", "pixel.png"), new Uint8Array([1, 2, 3]));

      const bytes = await readLocalImageFileBytes("nested/pixel.png", {
        localImageRoots: [directory],
        maxBytes: 3,
      });

      expect(Array.from(bytes)).toEqual([1, 2, 3]);
    });
  });

  test("rejects file ids without configured roots or image extensions", async () => {
    await expect(
      readLocalImageFileBytes("pixel.png", { localImageRoots: [], maxBytes: 3 }),
    ).rejects.toThrow("configured local image root");
    await expect(
      readLocalImageFileBytes("", { localImageRoots: [tmpdir()], maxBytes: 3 }),
    ).rejects.toThrow("non-empty relative path");
    await expect(
      readLocalImageFileBytes("pixel.txt", { localImageRoots: [tmpdir()], maxBytes: 3 }),
    ).rejects.toThrow("supported image extension");
  });

  test("rejects traversal, absolute, and oversized local image file ids", async () => {
    await withTemporaryDirectory("mlxts-serve-local-image", async (directory) => {
      writeFileSync(join(directory, "pixel.png"), new Uint8Array([1, 2, 3]));

      await expect(
        readLocalImageFileBytes("../pixel.png", { localImageRoots: [directory], maxBytes: 3 }),
      ).rejects.toThrow("traversal");
      await expect(
        readLocalImageFileBytes(join(directory, "pixel.png"), {
          localImageRoots: [directory],
          maxBytes: 3,
        }),
      ).rejects.toThrow("relative path");
      await expect(
        readLocalImageFileBytes("pixel.png", { localImageRoots: [directory], maxBytes: 2 }),
      ).rejects.toThrow("exceeding the 2 byte limit");
      await expect(
        readLocalImageFileBytes("missing.png", { localImageRoots: [directory], maxBytes: 3 }),
      ).rejects.toThrow("did not match an image");
    });
  });

  test("rejects aborted and directory-backed local image reads", async () => {
    await withTemporaryDirectory("mlxts-serve-local-image", async (directory) => {
      mkdirSync(join(directory, "folder.png"));
      const controller = new AbortController();
      controller.abort();

      await expect(
        readLocalImageFileBytes("pixel.png", {
          localImageRoots: [directory],
          maxBytes: 3,
          signal: controller.signal,
        }),
      ).rejects.toThrow("cancelled");
      await expect(
        readLocalImageFileBytes("folder.png", { localImageRoots: [directory], maxBytes: 3 }),
      ).rejects.toThrow("resolve to a file");
    });
  });

  test("rejects symlink escapes from configured image roots", async () => {
    await withTemporaryDirectory("mlxts-serve-local-image", async (directory) => {
      const root = join(directory, "root");
      const outside = join(directory, "outside");
      mkdirSync(root);
      mkdirSync(outside);
      writeFileSync(join(outside, "pixel.png"), new Uint8Array([1, 2, 3]));
      symlinkSync(join(outside, "pixel.png"), join(root, "escape.png"));

      await expect(
        readLocalImageFileBytes("escape.png", { localImageRoots: [root], maxBytes: 3 }),
      ).rejects.toThrow("outside the configured image root");
    });
  });
});
