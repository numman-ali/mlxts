import { afterEach, describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

import { loadJsonlDataset } from "./jsonl";

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

describe("loadJsonlDataset", () => {
  test("loads parsed records with a caller-owned parser", async () => {
    const directory = createTempDir("mlxts-data-jsonl-");
    const path = join(directory, "records.jsonl");
    await Bun.write(path, '{"value":1}\n{"value":2}\n');

    const dataset = await loadJsonlDataset(path, (value, lineIndex) => {
      if (typeof value !== "object" || value === null || !("value" in value)) {
        throw new Error(`bad line ${lineIndex}`);
      }
      return value.value;
    });

    expect(dataset.items()).toEqual([1, 2]);
  });
});
