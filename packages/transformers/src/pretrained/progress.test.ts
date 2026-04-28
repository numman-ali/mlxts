import { expect, test } from "bun:test";

import { createProgressReporter } from "./progress";

test("createProgressReporter formats pretrained loader events", () => {
  const lines: string[] = [];
  const report = createProgressReporter((line) => lines.push(line));

  report({ stage: "resolve", status: "start", source: "mlx-community/demo" });
  report({
    stage: "resolve",
    status: "complete",
    sourceKind: "hub",
    directory: "/tmp/demo",
    fileCount: 3,
    totalBytes: 1_500_000,
    repoId: "mlx-community/demo",
    resolvedRevision: "abcdef1234567890",
  });
  report({
    stage: "download",
    status: "complete",
    repoId: "mlx-community/demo",
    relativePath: "model.safetensors",
    size: 2_048,
    index: 1,
    totalFiles: 3,
    completedFiles: 1,
    completedBytes: 2_048,
    totalBytes: 1_500_000,
  });
  report({ stage: "model", status: "weights-start", shardCount: 2 });
  report({ stage: "model", status: "weights-complete", shardCount: 2 });
  report({ stage: "tokenizer", status: "start", directory: "/tmp/demo" });
  report({
    stage: "tokenizer",
    status: "complete",
    directory: "/tmp/demo",
    format: "tokenizer-json",
  });

  expect(lines).toEqual([
    "[resolve] resolving mlx-community/demo",
    "[resolve] mlx-community/demo @ abcdef123456 -> /tmp/demo (3 files, 1.5 MB)",
    "[download] 1/3 complete model.safetensors (2.0 KB) 2.0 KB / 1.5 MB",
    "[model] loading 2 safetensor shard(s)",
    "[model] finished loading 2 safetensor shard(s)",
    "[tokenizer] loading from /tmp/demo",
    "[tokenizer] ready",
  ]);
});
