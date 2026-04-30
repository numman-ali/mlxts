import { describe, expect, test } from "bun:test";

import {
  formatQwenGemmaRegressionError,
  formatQwenGemmaRegressionSuccess,
  formatQwenGemmaRegressionUsage,
  parseQwenGemmaRegressionArgs,
  qwenGemmaRegressionStageSpecs,
  runQwenGemmaRegressionCommand,
} from "./regression-qwen-gemma";

function parsedOptions(argv: readonly string[]) {
  const command = parseQwenGemmaRegressionArgs(argv);
  if (command.kind !== "run") {
    throw new Error("expected run command");
  }
  return command.options;
}

describe("Qwen/Gemma regression options", () => {
  test("defaults to the cheap profile", () => {
    expect(parseQwenGemmaRegressionArgs([])).toMatchObject({
      kind: "run",
      options: {
        profile: "quick",
        qwenModel: "mlx-community/Qwen3.6-27B-4bit",
        gemma4Model: "google/gemma-4-E2B-it",
        reportDir: ".tmp/qwen-gemma-regression",
        requestTimeoutMs: 3_600_000,
        allowDownload: false,
      },
    });
  });

  test("parses substantial profile options", () => {
    expect(
      parseQwenGemmaRegressionArgs([
        "--profile",
        "substantial",
        "--qwen-model",
        "qwen",
        "--gemma4-model",
        "gemma",
        "--report-dir",
        ".tmp/reports",
        "--request-timeout-ms",
        "120000",
        "--allow-download",
      ]),
    ).toEqual({
      kind: "run",
      options: {
        profile: "substantial",
        qwenModel: "qwen",
        gemma4Model: "gemma",
        reportDir: ".tmp/reports",
        requestTimeoutMs: 120000,
        allowDownload: true,
      },
    });
  });

  test("parses help without exiting", () => {
    expect(parseQwenGemmaRegressionArgs(["--help"])).toEqual({ kind: "help" });
    expect(parseQwenGemmaRegressionArgs(["-h"])).toEqual({ kind: "help" });
  });

  test("rejects invalid profile and timeout values", () => {
    expect(() => parseQwenGemmaRegressionArgs(["--profile", "overnight"])).toThrow(
      '--profile must be "quick", "real", or "substantial"',
    );
    expect(() => parseQwenGemmaRegressionArgs(["--request-timeout-ms", "0"])).toThrow(
      "--request-timeout-ms must be a positive integer",
    );
    expect(() => parseQwenGemmaRegressionArgs(["--request-timeout-ms", "1.5"])).toThrow(
      "--request-timeout-ms must be a positive integer",
    );
    expect(() => parseQwenGemmaRegressionArgs(["--request-timeout-ms", "12abc"])).toThrow(
      "--request-timeout-ms must be a positive integer",
    );
    expect(() => parseQwenGemmaRegressionArgs(["--qwen-model", ""])).toThrow(
      "--qwen-model requires a value",
    );
    expect(() => parseQwenGemmaRegressionArgs(["--missing"])).toThrow('unknown option "--missing"');
  });

  test("keeps profile stage commands stable", () => {
    expect(qwenGemmaRegressionStageSpecs(parsedOptions([]))).toEqual([
      {
        label: "transformer focused regressions",
        args: ["bun", "run", "--filter", "@mlxts/transformers", "regression:models"],
      },
      {
        label: "serve focused regressions",
        args: ["bun", "run", "--filter", "@mlxts/serve", "regression:serve"],
      },
    ]);

    expect(
      qwenGemmaRegressionStageSpecs(
        parsedOptions([
          "--profile",
          "real",
          "--qwen-model",
          "qwen",
          "--gemma4-model",
          "gemma",
          "--report-dir",
          ".tmp/reports",
          "--request-timeout-ms",
          "120000",
          "--allow-download",
        ]),
      ),
    ).toEqual([
      {
        label: "transformer real decode smoke",
        args: [
          "bun",
          "run",
          "packages/transformers/scripts/regression-model-matrix.ts",
          "--decode-smoke",
          "--qwen-model",
          "qwen",
          "--gemma4-model",
          "gemma",
        ],
      },
      {
        label: "serve real endpoint smoke",
        args: [
          "bun",
          "run",
          "packages/serve/scripts/regression-serve-matrix.ts",
          "--real-models",
          "--qwen-model",
          "qwen",
          "--gemma4-model",
          "gemma",
          "--fairness-smoke",
          "--report-dir",
          ".tmp/reports/serve",
          "--request-timeout-ms",
          "120000",
          "--allow-download",
        ],
      },
    ]);

    expect(
      qwenGemmaRegressionStageSpecs(
        parsedOptions([
          "--profile",
          "substantial",
          "--qwen-model",
          "qwen",
          "--gemma4-model",
          "gemma",
          "--report-dir",
          ".tmp/reports",
          "--request-timeout-ms",
          "120000",
          "--allow-download",
        ]),
      ),
    ).toEqual([
      {
        label: "transformer real decode smoke",
        args: [
          "bun",
          "run",
          "packages/transformers/scripts/regression-model-matrix.ts",
          "--decode-smoke",
          "--qwen-model",
          "qwen",
          "--gemma4-model",
          "gemma",
        ],
      },
      {
        label: "serve capability smoke",
        args: [
          "bun",
          "run",
          "packages/serve/scripts/regression-serve-matrix.ts",
          "--capability-smoke",
          "--qwen-model",
          "qwen",
          "--gemma4-model",
          "gemma",
          "--report-dir",
          ".tmp/reports/serve",
          "--request-timeout-ms",
          "120000",
          "--allow-download",
        ],
      },
      {
        label: "Qwen long-context retrieval smoke",
        args: [
          "bun",
          "run",
          "bench:generation:context",
          "--model",
          "qwen",
          "--rungs",
          "32768",
          "--needle-placements",
          "all",
          "--generation-tokens",
          "24",
          "--fail-on-mismatch",
          "--max-active-slope-mb-per-token",
          "1",
          "--report-json",
          ".tmp/reports/qwen36-context-32k-all.json",
        ],
      },
    ]);
  });

  test("formats compact AXI success and error output", () => {
    expect(
      formatQwenGemmaRegressionSuccess({
        profile: "real",
        reportDir: ".tmp/qwen-gemma-regression",
        stages: [
          { label: "transformer real decode smoke", status: "passed" },
          { label: "serve real endpoint smoke", status: "passed" },
        ],
      }),
    ).toBe(
      [
        "qwen_gemma_regression:",
        "  status: passed",
        '  profile: "real"',
        '  report_dir: ".tmp/qwen-gemma-regression"',
        "  stages: 2",
        "stages[2]{label,status}:",
        '  "transformer real decode smoke","passed"',
        '  "serve real endpoint smoke","passed"',
      ].join("\n"),
    );
    expect(formatQwenGemmaRegressionError("bad flag", "rerun")).toBe(
      ["error:", '  message: "bad flag"', 'help: "rerun"'].join("\n"),
    );
    expect(formatQwenGemmaRegressionError("bad\nflag", "rerun")).toContain("  message: |");
  });

  test("runs help, success, usage error, and runtime error paths with AXI stdout", async () => {
    const helpStdout: string[] = [];
    expect(
      await runQwenGemmaRegressionCommand(["--help"], {
        stdout: (text) => helpStdout.push(text),
      }),
    ).toBe(0);
    expect(helpStdout.join("\n")).toBe(formatQwenGemmaRegressionUsage());

    const stdout: string[] = [];
    const stderr: string[] = [];
    expect(
      await runQwenGemmaRegressionCommand(["--profile", "real"], {
        stdout: (text) => stdout.push(text),
        stderr: (text) => stderr.push(text),
        runRegression: async (options, progress) => {
          expect(options.profile).toBe("real");
          progress("[qwen-gemma-regression] stage started");
          return {
            profile: options.profile,
            reportDir: options.reportDir,
            stages: [{ label: "transformer real decode smoke", status: "passed" }],
          };
        },
      }),
    ).toBe(0);
    expect(stderr).toEqual(["[qwen-gemma-regression] stage started"]);
    expect(stdout.join("\n")).toContain("qwen_gemma_regression:");
    expect(stdout.join("\n")).toContain('"transformer real decode smoke","passed"');

    const usageStdout: string[] = [];
    expect(
      await runQwenGemmaRegressionCommand(["--request-timeout-ms", "0"], {
        stdout: (text) => usageStdout.push(text),
      }),
    ).toBe(2);
    expect(usageStdout.join("\n")).toContain("--request-timeout-ms must be a positive integer");

    const runtimeStdout: string[] = [];
    expect(
      await runQwenGemmaRegressionCommand([], {
        stdout: (text) => runtimeStdout.push(text),
        runRegression: async () => {
          throw new Error("regression failed");
        },
      }),
    ).toBe(1);
    expect(runtimeStdout.join("\n")).toContain("regression failed");
  });
});
