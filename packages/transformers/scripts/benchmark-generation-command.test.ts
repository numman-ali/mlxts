import { describe, expect, test } from "bun:test";
import {
  type BenchmarkCommandReport,
  formatBenchmarkError,
  formatBenchmarkSuccess,
  formatBenchmarkUsage,
} from "./benchmark-common";
import { runGenerationBenchmarkCommand } from "./benchmark-generation";
import { runGenerationParityBenchmarkCommand } from "./benchmark-generation-parity";

function benchmarkReport(): BenchmarkCommandReport {
  return {
    name: "gemma",
    model: "google/gemma-4-E2B-it",
    snapshotPath: "/tmp/gemma",
    promptTokens: 16,
    generationTokens: 2,
    prefillStepSize: 16,
    trials: 1,
    decodeSchedule: "async",
    materializeCacheEachToken: false,
    metrics: {
      promptTps: 128,
      generationTps: 8,
      peakMemoryGb: 1.25,
      activeMemoryStartGb: 1,
      activeMemoryEndGb: 1.1,
      activeMemoryDeltaGb: 0.1,
      activeMemoryMaxGb: 1.2,
      activeMemorySlopeMbPerToken: 50,
      explicitEvalCountPerToken: 1,
      totalTimeSeconds: 0.5,
    },
    mlxLmReference: {
      promptTps: 120,
      generationTps: 8.5,
      peakMemoryGb: 1.2,
      capturedAt: "2026-04-30",
      trialCount: 1,
    },
    warnings: ["generation_tps below mlx-lm"],
  };
}

describe("benchmark generation command boundary", () => {
  test("formats AXI help, success, and errors", () => {
    expect(formatBenchmarkUsage("synthetic")).toContain("exit_codes[3]");
    expect(formatBenchmarkUsage("parity")).toContain("--require-mlx-lm-reference");
    expect(formatBenchmarkSuccess("parity", [benchmarkReport()])).toBe(
      [
        "generation_benchmark:",
        "  status: passed",
        '  mode: "parity"',
        "  targets: 1",
        "targets[1]{name,model,snapshot_path,prompt_tokens,generation_tokens,prefill_step_size,trials,decode_schedule,materialize_cache_each_token,prompt_tps,generation_tps,peak_memory_gb,evals_per_token,total_time_s,mlx_lm_generation_tps,warning_count}:",
        '  "gemma","google/gemma-4-E2B-it","/tmp/gemma",16,2,16,1,"async",false,128.000,8.000,1.250,1.00,0.500,8.500,1',
        "warnings[1]{target,message}:",
        '  "gemma","generation_tps below mlx-lm"',
      ].join("\n"),
    );
    expect(formatBenchmarkError("bad flag", "rerun")).toBe(
      ["error:", '  message: "bad flag"', 'help: "rerun"'].join("\n"),
    );
    expect(formatBenchmarkError("bad\nflag", "rerun")).toContain("  message: |");
  });

  test("runs synthetic help, success, usage error, and runtime error paths", async () => {
    const helpStdout: string[] = [];
    expect(
      await runGenerationBenchmarkCommand(["--help"], {
        stdout: (text) => helpStdout.push(text),
      }),
    ).toBe(0);
    expect(helpStdout.join("\n")).toBe(formatBenchmarkUsage("synthetic"));

    const stdout: string[] = [];
    const stderr: string[] = [];
    let lockDepth = 0;
    expect(
      await runGenerationBenchmarkCommand(["--model", "model", "--trials", "1"], {
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
        runBenchmarks: async (parsed, progress) => {
          expect(parsed.model).toBe("model");
          expect(lockDepth).toBe(1);
          progress("benchmark-generation: probe");
          return [benchmarkReport()];
        },
      }),
    ).toBe(0);
    expect(lockDepth).toBe(0);
    expect(stderr).toEqual(["benchmark-generation: probe"]);
    expect(stdout.join("\n")).toContain("generation_benchmark:");
    expect(stdout.join("\n")).not.toContain("benchmark-generation: probe");

    const usageStdout: string[] = [];
    let usageLockCalls = 0;
    expect(
      await runGenerationBenchmarkCommand(["--trials", "0"], {
        stdout: (text) => usageStdout.push(text),
        acquireLock: () => {
          usageLockCalls += 1;
          return { [Symbol.dispose]: () => undefined };
        },
      }),
    ).toBe(2);
    expect(usageLockCalls).toBe(0);
    expect(usageStdout.join("\n")).toContain("trials must be a positive integer");

    const strictIntegerStdout: string[] = [];
    expect(
      await runGenerationBenchmarkCommand(["--trials", "12abc"], {
        stdout: (text) => strictIntegerStdout.push(text),
      }),
    ).toBe(2);
    expect(strictIntegerStdout.join("\n")).toContain("--trials expects an integer value");

    const missingValueStdout: string[] = [];
    expect(
      await runGenerationBenchmarkCommand(["--model", "--trials", "1"], {
        stdout: (text) => missingValueStdout.push(text),
      }),
    ).toBe(2);
    expect(missingValueStdout.join("\n")).toContain("--model requires a value");

    const runtimeStdout: string[] = [];
    let runtimeLockDepth = 0;
    expect(
      await runGenerationBenchmarkCommand(["--model", "model"], {
        stdout: (text) => runtimeStdout.push(text),
        acquireLock: () => {
          runtimeLockDepth += 1;
          return {
            [Symbol.dispose]: () => {
              runtimeLockDepth -= 1;
            },
          };
        },
        runBenchmarks: async () => {
          throw new Error("benchmark failed");
        },
      }),
    ).toBe(1);
    expect(runtimeLockDepth).toBe(0);
    expect(runtimeStdout.join("\n")).toContain("benchmark failed");
  });

  test("runs parity command paths with parity-specific help and progress", async () => {
    const helpStdout: string[] = [];
    expect(
      await runGenerationParityBenchmarkCommand(["--help"], {
        stdout: (text) => helpStdout.push(text),
      }),
    ).toBe(0);
    expect(helpStdout.join("\n")).toBe(formatBenchmarkUsage("parity"));

    const stdout: string[] = [];
    const stderr: string[] = [];
    expect(
      await runGenerationParityBenchmarkCommand(["--model", "model", "--skip-mlx-lm-reference"], {
        stdout: (text) => stdout.push(text),
        stderr: (text) => stderr.push(text),
        acquireLock: () => ({ [Symbol.dispose]: () => undefined }),
        runBenchmarks: async (parsed, progress) => {
          expect(parsed.reference.captureMlxLmReference).toBe(false);
          progress("benchmark-generation-parity: probe");
          return [benchmarkReport()];
        },
      }),
    ).toBe(0);
    expect(stderr).toEqual(["benchmark-generation-parity: probe"]);
    expect(stdout.join("\n")).toContain('mode: "parity"');
  });
});
