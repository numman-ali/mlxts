#!/usr/bin/env bun

import { defaultAdapterOutputDir, defaultQuantizedOutputDir, defaultReportPath } from "./args";

const DEFAULT_MATRIX_SOURCES = [
  "meta-llama/Llama-3.2-1B-Instruct",
  "google/gemma-3-1b-it",
  "google/gemma-4-E2B-it",
  "microsoft/Phi-4-mini-instruct",
  "mistralai/Mistral-7B-Instruct-v0.3",
] as const;

export type MatrixArgs = {
  sources: string[];
  passthrough: string[];
};

export type MatrixCommand = { kind: "help" } | { kind: "run"; options: MatrixArgs };

export type MatrixRunResult = {
  source: string;
  status: "passed";
  report: string;
  quantizedOutput: string;
  adapterOutput: string;
};

export type MatrixResult = {
  sources: readonly MatrixRunResult[];
  passthroughCount: number;
};

type MatrixRuntime = {
  stdout?: (text: string) => void;
  stderr?: (text: string) => void;
  runMatrix?: (options: MatrixArgs, progress: (line: string) => void) => Promise<MatrixResult>;
};

type MatrixRunSource = (
  source: string,
  passthrough: readonly string[],
  progress: (line: string) => void,
) => Promise<MatrixRunResult>;

class MatrixUsageError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "MatrixUsageError";
  }
}

function readValue(flag: string, value: string | undefined): string {
  if (value === undefined || value.trim() === "" || value.startsWith("--")) {
    throw new MatrixUsageError(`training proof matrix: ${flag} expects a non-empty value.`);
  }
  return value;
}

export function parseMatrixArgs(argv: readonly string[]): MatrixArgs {
  const sources: string[] = [];
  const passthrough: string[] = [];

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === undefined) {
      continue;
    }

    if (arg === "--source") {
      sources.push(readValue(arg, argv[index + 1]));
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

export function parseMatrixCommand(argv: readonly string[]): MatrixCommand {
  if (argv.includes("--help") || argv.includes("-h")) {
    return { kind: "help" };
  }
  return { kind: "run", options: parseMatrixArgs(argv) };
}

function safeSource(source: string): string {
  return source.replace(/[^A-Za-z0-9._-]+/g, "-").replace(/-+/g, "-");
}

function quoteScalar(value: string | number | boolean | null): string {
  return typeof value === "string" ? JSON.stringify(value) : String(value);
}

function formatBlockField(name: string, value: string): string[] {
  const lines = value.split(/\r?\n/);
  return [`  ${name}: |`, ...lines.map((line) => `    ${line}`)];
}

export function formatMatrixUsage(): string {
  return [
    "description: Run the Phase 8 training proof across the local family matrix",
    "usage[2]:",
    "  bun run examples/train-proof/matrix.ts --dataset-source tiny --train-limit 8 --eval-limit 4 --steps 2",
    "  bun run examples/train-proof/matrix.ts --source google/gemma-3-1b-it --stages lora",
    "options[3]{flag,description}:",
    '  "--source <id>","Run one matrix source; repeatable; defaults to the curated family matrix"',
    '  "--help","Show this help"',
    '  "<proof flags>","All other flags pass through to `bun run proof:training`"',
    "exit_codes[3]{code,meaning}:",
    '  0,"matrix completed or help"',
    '  1,"runtime or child proof failure"',
    '  2,"usage error"',
  ].join("\n");
}

export function formatMatrixError(message: string, code: "usage" | "runtime"): string {
  return [
    "error:",
    `  code: ${quoteScalar(code)}`,
    ...formatBlockField("message", message),
    "help[1]:",
    '  "Run `bun run examples/train-proof/matrix.ts --help` for options"',
  ].join("\n");
}

export function formatMatrixSuccess(result: MatrixResult): string {
  const rows = result.sources.map((entry) =>
    [
      quoteScalar(entry.source),
      quoteScalar(entry.status),
      quoteScalar(entry.report),
      quoteScalar(entry.adapterOutput),
      quoteScalar(entry.quantizedOutput),
    ].join(","),
  );
  return [
    "training_proof_matrix:",
    "  status: passed",
    `  sources: ${result.sources.length}`,
    `  passthrough_flags: ${result.passthroughCount}`,
    `runs[${rows.length}]{source,status,report,adapter_output,quantized_output}:`,
    ...rows.map((row) => `  ${row}`),
  ].join("\n");
}

function matrixPaths(source: string): Omit<MatrixRunResult, "status"> {
  const report = defaultReportPath(source).replace(
    /-report\.json$/,
    `-matrix-${safeSource(source)}-report.json`,
  );
  const quantizedOutput = defaultQuantizedOutputDir(source).replace(
    /-4bit$/,
    `-matrix-${safeSource(source)}-4bit`,
  );
  const adapterOutput = defaultAdapterOutputDir(source).replace(
    /-adapters$/,
    `-matrix-${safeSource(source)}-adapters`,
  );
  return { source, report, quantizedOutput, adapterOutput };
}

function inheritedStringEnv(): Record<string, string> {
  return Object.fromEntries(
    Object.entries(process.env).filter((entry): entry is [string, string] => {
      const value = entry[1];
      return typeof value === "string";
    }),
  );
}

async function pipeReadableToProgress(
  stream: ReadableStream<Uint8Array> | null,
  progress: (text: string) => void,
): Promise<void> {
  if (stream === null) {
    return;
  }
  const decoder = new TextDecoder();
  let pending = "";
  for await (const chunk of stream) {
    pending += decoder.decode(chunk, { stream: true });
    let newline = pending.indexOf("\n");
    while (newline !== -1) {
      const line = pending.slice(0, newline);
      if (line !== "") {
        progress(line);
      }
      pending = pending.slice(newline + 1);
      newline = pending.indexOf("\n");
    }
  }
  pending += decoder.decode();
  if (pending !== "") {
    progress(pending);
  }
}

async function runOne(
  source: string,
  passthrough: readonly string[],
  progress: (line: string) => void,
): Promise<MatrixRunResult> {
  const paths = matrixPaths(source);
  progress(`[training-proof-matrix] proof: ${source}`);
  const child = Bun.spawn(
    [
      "bun",
      "run",
      new URL("./index.ts", import.meta.url).pathname,
      "--source",
      source,
      "--quantized-output",
      paths.quantizedOutput,
      "--adapter-output",
      paths.adapterOutput,
      "--report",
      paths.report,
      ...passthrough,
    ],
    {
      env: inheritedStringEnv(),
      stdout: "pipe",
      stderr: "pipe",
      stdin: "ignore",
    },
  );

  const stdout = pipeReadableToProgress(child.stdout, progress);
  const stderr = pipeReadableToProgress(child.stderr, progress);
  const exitCode = await child.exited;
  await Promise.all([stdout, stderr]);
  if (exitCode !== 0) {
    throw new Error(
      `training proof matrix: proof run failed for ${source} with exit code ${exitCode}.`,
    );
  }
  return { ...paths, status: "passed" };
}

export async function runTrainingProofMatrix(
  options: MatrixArgs,
  progress: (line: string) => void = console.error,
  runSource: MatrixRunSource = runOne,
): Promise<MatrixResult> {
  progress(`[training-proof-matrix] sources: ${options.sources.join(", ")}`);
  const results: MatrixRunResult[] = [];
  for (const source of options.sources) {
    results.push(await runSource(source, options.passthrough, progress));
  }
  return { sources: results, passthroughCount: options.passthrough.length };
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export async function runMatrixCommand(
  argv: readonly string[],
  runtime: MatrixRuntime = {},
): Promise<number> {
  const stdout = runtime.stdout ?? console.log;
  const stderr = runtime.stderr ?? console.error;
  const runMatrix = runtime.runMatrix ?? runTrainingProofMatrix;
  let command: MatrixCommand;

  try {
    command = parseMatrixCommand(argv);
  } catch (error) {
    stdout(formatMatrixError(errorMessage(error), "usage"));
    return error instanceof MatrixUsageError ? 2 : 1;
  }

  if (command.kind === "help") {
    stdout(formatMatrixUsage());
    return 0;
  }

  try {
    const result = await runMatrix(command.options, stderr);
    stdout(formatMatrixSuccess(result));
    return 0;
  } catch (error) {
    stdout(formatMatrixError(errorMessage(error), "runtime"));
    if (error instanceof Error && error.stack !== undefined) {
      stderr(error.stack);
    }
    return 1;
  }
}

if (import.meta.main) {
  const exitCode = await runMatrixCommand(Bun.argv.slice(2));
  process.exit(exitCode);
}
