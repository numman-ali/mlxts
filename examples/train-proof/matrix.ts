#!/usr/bin/env bun

import { defaultQuantizedOutputDir, defaultReportPath } from "./args";

const DEFAULT_MATRIX_SOURCES = [
  "meta-llama/Llama-3.2-1B-Instruct",
  "google/gemma-3-1b-it",
  "google/gemma-4-E2B-it",
  "microsoft/Phi-4-mini-instruct",
  "mistralai/Mistral-7B-Instruct-v0.3",
] as const;

type MatrixArgs = {
  sources: string[];
  passthrough: string[];
};

function parseMatrixArgs(argv: readonly string[]): MatrixArgs {
  const sources: string[] = [];
  const passthrough: string[] = [];

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === undefined) {
      continue;
    }

    if (arg === "--source") {
      const value = argv[index + 1];
      if (value === undefined || value.trim() === "") {
        throw new Error("training proof matrix: --source expects a non-empty value.");
      }
      sources.push(value);
      index += 1;
      continue;
    }

    passthrough.push(arg);
  }

  return {
    sources: sources.length === 0 ? [...DEFAULT_MATRIX_SOURCES] : sources,
    passthrough,
  };
}

function safeSource(source: string): string {
  return source.replace(/[^A-Za-z0-9._-]+/g, "-").replace(/-+/g, "-");
}

async function runOne(source: string, passthrough: readonly string[]): Promise<void> {
  const report = defaultReportPath(source).replace(
    /-report\.json$/,
    `-matrix-${safeSource(source)}-report.json`,
  );
  const quantizedOutput = defaultQuantizedOutputDir(source).replace(
    /-4bit$/,
    `-matrix-${safeSource(source)}-4bit`,
  );
  const process = Bun.spawn(
    [
      "bun",
      "run",
      "examples/train-proof/index.ts",
      "--source",
      source,
      "--quantized-output",
      quantizedOutput,
      "--report",
      report,
      ...passthrough,
    ],
    {
      stdout: "inherit",
      stderr: "inherit",
      stdin: "inherit",
    },
  );

  const exitCode = await process.exited;
  if (exitCode !== 0) {
    throw new Error(
      `training proof matrix: proof run failed for ${source} with exit code ${exitCode}.`,
    );
  }
}

async function main(): Promise<void> {
  const parsed = parseMatrixArgs(Bun.argv.slice(2));
  console.log(`Training proof matrix sources: ${parsed.sources.join(", ")}`);
  for (const source of parsed.sources) {
    console.log(`\n=== proof: ${source} ===`);
    await runOne(source, parsed.passthrough);
  }
}

await main();
