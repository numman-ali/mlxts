import { describe, expect, test } from "bun:test";

import {
  formatMatrixSuccess,
  formatMatrixUsage,
  type MatrixResult,
  runMatrixCommand,
  runTrainingProofMatrix,
} from "./matrix";

function matrixResult(): MatrixResult {
  return {
    passthroughCount: 4,
    sources: [
      {
        source: "local-model",
        status: "passed",
        report: ".tmp/training-proof/local-model-matrix-report.json",
        adapterOutput: ".tmp/training-proof/local-model-matrix-adapters",
        quantizedOutput: ".tmp/training-proof/local-model-matrix-4bit",
      },
    ],
  };
}

describe("training proof matrix command", () => {
  test("help is compact AXI stdout and does not run child proofs", async () => {
    const stdout: string[] = [];
    let runCalls = 0;

    const exitCode = await runMatrixCommand(["--help"], {
      stdout: (text) => stdout.push(text),
      runMatrix: async () => {
        runCalls += 1;
        return matrixResult();
      },
    });

    expect(exitCode).toBe(0);
    expect(runCalls).toBe(0);
    expect(stdout.join("\n")).toBe(formatMatrixUsage());
    expect(stdout.join("\n")).toContain("options[3]");
    expect(stdout.join("\n")).toContain("exit_codes[3]");
  });

  test("usage errors use structured stdout before child proofs", async () => {
    const stdout: string[] = [];
    let runCalls = 0;

    const exitCode = await runMatrixCommand(["--source", "--train-limit"], {
      stdout: (text) => stdout.push(text),
      runMatrix: async () => {
        runCalls += 1;
        return matrixResult();
      },
    });

    expect(exitCode).toBe(2);
    expect(runCalls).toBe(0);
    expect(stdout.join("\n")).toContain('code: "usage"');
    expect(stdout.join("\n")).toContain("--source expects a non-empty value");
  });

  test("success output is structured and child progress stays on stderr", async () => {
    const stdout: string[] = [];
    const stderr: string[] = [];
    const result = matrixResult();

    const exitCode = await runMatrixCommand(
      ["--source", "local-model", "--dataset-source", "tiny", "--steps", "1"],
      {
        stdout: (text) => stdout.push(text),
        stderr: (text) => stderr.push(text),
        runMatrix: async (options, progress) => {
          expect(options.sources).toEqual(["local-model"]);
          expect(options.passthrough).toEqual(["--dataset-source", "tiny", "--steps", "1"]);
          progress("child proof stdout");
          return result;
        },
      },
    );

    expect(exitCode).toBe(0);
    expect(stderr).toEqual(["child proof stdout"]);
    expect(stdout.join("\n")).toBe(formatMatrixSuccess(result));
    expect(stdout.join("\n")).toContain("training_proof_matrix:");
    expect(stdout.join("\n")).toContain("runs[1]");
  });

  test("runtime errors use structured stdout and stderr stack traces", async () => {
    const stdout: string[] = [];
    const stderr: string[] = [];

    const exitCode = await runMatrixCommand(["--source", "local-model"], {
      stdout: (text) => stdout.push(text),
      stderr: (text) => stderr.push(text),
      runMatrix: async () => {
        throw new Error("matrix failed");
      },
    });

    expect(exitCode).toBe(1);
    expect(stdout.join("\n")).toContain('code: "runtime"');
    expect(stdout.join("\n")).toContain("matrix failed");
    expect(stderr.join("\n")).toContain("Error: matrix failed");
  });

  test("matrix runner executes sources in order and keeps passthrough flags", async () => {
    const progress: string[] = [];
    const seen: string[] = [];

    const result = await runTrainingProofMatrix(
      { sources: ["one", "two"], passthrough: ["--dataset-source", "tiny"] },
      (line) => progress.push(line),
      async (source, passthrough, writeLine) => {
        seen.push(source);
        expect(passthrough).toEqual(["--dataset-source", "tiny"]);
        writeLine(`ran ${source}`);
        return {
          source,
          status: "passed",
          report: `${source}.json`,
          adapterOutput: `${source}-adapters`,
          quantizedOutput: `${source}-4bit`,
        };
      },
    );

    expect(seen).toEqual(["one", "two"]);
    expect(result.passthroughCount).toBe(2);
    expect(result.sources.map((entry) => entry.source)).toEqual(["one", "two"]);
    expect(progress).toEqual(["[training-proof-matrix] sources: one, two", "ran one", "ran two"]);
  });
});
