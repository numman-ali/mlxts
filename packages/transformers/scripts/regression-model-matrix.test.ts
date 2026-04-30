import { describe, expect, test } from "bun:test";

import {
  formatTransformersModelRegressionError,
  formatTransformersModelRegressionSuccess,
  formatTransformersModelRegressionUsage,
  parseDecodeSmokeMetrics,
  parseTransformersModelRegressionArgs,
  runTransformersModelRegressionCommand,
  TRANSFORMERS_MODEL_REGRESSION_FOCUSED_TESTS,
} from "./regression-model-matrix";

function parsedOptions(argv: readonly string[]) {
  const command = parseTransformersModelRegressionArgs(argv);
  if (command.kind !== "run") {
    throw new Error("expected run command");
  }
  return command.options;
}

describe("transformers model regression options", () => {
  test("defaults to focused checks only", () => {
    expect(parseTransformersModelRegressionArgs([])).toEqual({
      kind: "run",
      options: {
        realModels: false,
        decodeSmoke: false,
        qwenModel: "mlx-community/Qwen3.6-27B-4bit",
        gemma4Model: "google/gemma-4-E2B-it",
        qwenMaxActiveGb: 16.5,
        gemma4MaxActiveGb: 10.5,
      },
    });
  });

  test("parses real and decode smoke profiles", () => {
    expect(parsedOptions(["--real-models"]).realModels).toBe(true);
    expect(
      parsedOptions([
        "--decode-smoke",
        "--qwen-model",
        "qwen",
        "--gemma4-model",
        "gemma",
        "--qwen-max-active-gb",
        "22.5",
        "--gemma4-max-active-gb",
        "12.25",
      ]),
    ).toEqual({
      realModels: true,
      decodeSmoke: true,
      qwenModel: "qwen",
      gemma4Model: "gemma",
      qwenMaxActiveGb: 22.5,
      gemma4MaxActiveGb: 12.25,
    });
  });

  test("parses help without exiting and rejects invalid args", () => {
    expect(parseTransformersModelRegressionArgs(["--help"])).toEqual({ kind: "help" });
    expect(parseTransformersModelRegressionArgs(["-h"])).toEqual({ kind: "help" });
    expect(() => parseTransformersModelRegressionArgs(["--qwen-model", ""])).toThrow(
      "--qwen-model requires a value",
    );
    expect(() => parseTransformersModelRegressionArgs(["--qwen-max-active-gb", "0"])).toThrow(
      "--qwen-max-active-gb must be a positive number",
    );
    expect(() => parseTransformersModelRegressionArgs(["--qwen-max-active-gb", "nan"])).toThrow(
      "--qwen-max-active-gb must be a positive number",
    );
    expect(() => parseTransformersModelRegressionArgs(["--missing"])).toThrow(
      'unknown option "--missing"',
    );
    expect(() => parseTransformersModelRegressionArgs(["extra"])).toThrow(
      'unexpected argument "extra"',
    );
  });

  test("keeps the focused guardrail file list pinned", () => {
    expect([...TRANSFORMERS_MODEL_REGRESSION_FOCUSED_TESTS]).toEqual([
      "packages/nn/src/quantized/quantized-embedding.test.ts",
      "packages/quantize/src/quantize-module.test.ts",
      "packages/quantize/src/setup-quantized-module.test.ts",
      "packages/transformers/scripts/regression-model-matrix.test.ts",
      "packages/transformers/src/families/qwen3_5/model.test.ts",
      "packages/transformers/src/families/qwen3_5/weights.test.ts",
      "packages/transformers/src/families/gemma4/model.test.ts",
      "packages/transformers/src/families/gemma4/weights.test.ts",
      "packages/transformers/src/load.test.ts",
    ]);
  });

  test("parses decode smoke averages and rejects missing benchmark keys", () => {
    expect(
      parseDecodeSmokeMetrics(
        [
          "Trial 1:",
          "Averages: prompt_tps=501.25 generation_tps=31 peak_memory=12.5 active_slope_mb_per_token=-0.125 evals_per_token=1",
        ].join("\n"),
        "decode smoke qwen",
      ),
    ).toEqual({
      promptTps: 501.25,
      generationTps: 31,
      peakMemoryGb: 12.5,
      activeSlopeMbPerToken: -0.125,
      explicitEvalCountPerToken: 1,
    });
    expect(() => parseDecodeSmokeMetrics("Trial 1", "decode smoke qwen")).toThrow(
      "[regression] decode smoke qwen did not print benchmark averages",
    );
    expect(() =>
      parseDecodeSmokeMetrics(
        "Averages: prompt_tps=501.25 generation_tps=31 peak_memory=12.5 evals_per_token=1",
        "decode smoke qwen",
      ),
    ).toThrow("[regression] decode smoke missing active_slope_mb_per_token");
  });

  test("formats compact AXI success and error output", () => {
    expect(
      formatTransformersModelRegressionSuccess({
        focusedChecks: "passed",
        realModels: true,
        decodeSmoke: true,
        stages: [{ label: "focused unit checks", status: "passed" }],
        memoryReports: [
          {
            model: "qwen",
            modelType: "qwen3_5_text",
            activeGb: 12.3456,
            cacheGb: 1.25,
            peakGb: 13,
            parameterCount: 42,
          },
        ],
        decodeReports: [
          {
            model: "qwen",
            metrics: {
              promptTps: 5000,
              generationTps: 70.1234,
              peakMemoryGb: 13,
              activeSlopeMbPerToken: 0.5,
              explicitEvalCountPerToken: 1,
            },
          },
        ],
      }),
    ).toBe(
      [
        "transformers_model_regression:",
        "  status: passed",
        "  real_models: true",
        "  decode_smoke: true",
        "  stages: 1",
        "  memory_reports: 1",
        "  decode_reports: 1",
        "stages[1]{label,status}:",
        '  "focused unit checks","passed"',
        "memory_reports[1]{model,model_type,active_gb,cache_gb,peak_gb,parameter_count}:",
        '"qwen","qwen3_5_text",12.346,1.250,13.000,42',
        "decode_reports[1]{model,prompt_tps,generation_tps,peak_memory_gb,active_slope_mb_per_token,evals_per_token}:",
        '"qwen",5000.000,70.123,13.000,0.500,1.000',
      ].join("\n"),
    );
    expect(formatTransformersModelRegressionError("bad flag", "rerun")).toBe(
      ["error:", '  message: "bad flag"', 'help: "rerun"'].join("\n"),
    );
    expect(formatTransformersModelRegressionError("bad\nflag", "rerun")).toContain("  message: |");
  });

  test("runs help, success, usage error, and runtime error paths with AXI stdout", async () => {
    const helpStdout: string[] = [];
    expect(
      await runTransformersModelRegressionCommand(["--help"], {
        stdout: (text) => helpStdout.push(text),
      }),
    ).toBe(0);
    expect(helpStdout.join("\n")).toBe(formatTransformersModelRegressionUsage());

    let cheapLockCalls = 0;
    expect(
      await runTransformersModelRegressionCommand([], {
        stdout: () => undefined,
        acquireLock: () => {
          cheapLockCalls += 1;
          return { [Symbol.dispose]: () => undefined };
        },
        runRegression: async (options) => ({
          focusedChecks: "passed",
          realModels: options.realModels,
          decodeSmoke: options.decodeSmoke,
          stages: [{ label: "focused unit checks", status: "passed" }],
          memoryReports: [],
          decodeReports: [],
        }),
      }),
    ).toBe(0);
    expect(cheapLockCalls).toBe(0);

    const stdout: string[] = [];
    const stderr: string[] = [];
    let lockDepth = 0;
    expect(
      await runTransformersModelRegressionCommand(["--decode-smoke"], {
        stdout: (text) => stdout.push(text),
        stderr: (text) => stderr.push(text),
        acquireLock: () => {
          lockDepth += 1;
          return {
            [Symbol.dispose]: () => {
              lockDepth -= 1;
            },
          };
        },
        runRegression: async (options, progress) => {
          expect(options.realModels).toBe(true);
          expect(options.decodeSmoke).toBe(true);
          expect(lockDepth).toBe(1);
          progress("[regression] focused unit checks");
          return {
            focusedChecks: "passed",
            realModels: options.realModels,
            decodeSmoke: options.decodeSmoke,
            stages: [{ label: "focused unit checks", status: "passed" }],
            memoryReports: [],
            decodeReports: [],
          };
        },
      }),
    ).toBe(0);
    expect(lockDepth).toBe(0);
    expect(stderr).toEqual(["[regression] focused unit checks"]);
    expect(stdout.join("\n")).toContain("transformers_model_regression:");

    const usageStdout: string[] = [];
    expect(
      await runTransformersModelRegressionCommand(["--qwen-max-active-gb", "0"], {
        stdout: (text) => usageStdout.push(text),
      }),
    ).toBe(2);
    expect(usageStdout.join("\n")).toContain("--qwen-max-active-gb must be a positive number");

    const runtimeStdout: string[] = [];
    expect(
      await runTransformersModelRegressionCommand([], {
        stdout: (text) => runtimeStdout.push(text),
        runRegression: async () => {
          throw new Error("regression failed");
        },
      }),
    ).toBe(1);
    expect(runtimeStdout.join("\n")).toContain("regression failed");
  });
});
