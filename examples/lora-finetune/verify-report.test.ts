import { describe, expect, test } from "bun:test";

import type { FinetuneReport } from "./args";
import { assertFinetuneReport, parseFinetuneReport } from "./verification";
import {
  formatLoRAFinetuneVerifySuccess,
  formatLoRAFinetuneVerifyUsage,
  parseLoRAFinetuneVerifyArgs,
  runLoRAFinetuneVerifyCommand,
} from "./verify-report";

function reportFixture(overrides: Partial<FinetuneReport> = {}): FinetuneReport {
  const base: FinetuneReport = {
    source: "local-model",
    mode: "lora",
    preset: "attention",
    adapterFormat: "mlxts",
    datasetSource: "tiny",
    trainLimit: 8,
    evalLimit: 4,
    batchSize: 2,
    steps: 2,
    maxSequenceLength: 128,
    outputDir: ".tmp/lora-finetune/local-model",
    adapterDir: ".tmp/lora-finetune/local-model/adapter",
    metrics: {
      evalLossBefore: 2.5,
      evalLossAfter: 2.25,
      averageTrainingLoss: 2.4,
      targetCount: 4,
    },
    targetPaths: [
      "model.layers.0.self_attn.q_proj",
      "model.layers.0.self_attn.k_proj",
      "model.layers.0.self_attn.v_proj",
      "model.layers.0.self_attn.o_proj",
    ],
    parameterCounts: {
      total: 32,
      trainable: 8,
    },
    memory: {
      peakBytes: 1024,
    },
    dataStats: {
      train: {
        kept: 8,
        skippedMalformed: 0,
        skippedLong: 0,
      },
      eval: {
        kept: 4,
        skippedMalformed: 0,
        skippedLong: 0,
      },
    },
    adapterCheck: {
      reloadedMatchesTrained: true,
      qloraQuantizedBasePreserved: null,
    },
    samplePrompt: [{ role: "user", content: "Say hi" }],
    sampleText: {
      trained: "hi from adapter",
      reloaded: "hi from adapter",
      merged: "hi from merged",
    },
  };
  return { ...base, ...overrides };
}

describe("lora finetune report verification", () => {
  test("accepts a complete machine-checkable report", () => {
    const report = parseFinetuneReport(reportFixture());
    const verification = assertFinetuneReport(report, {
      expectedMode: "lora",
      expectedAdapterFormat: "mlxts",
      requireLossNotWorse: true,
    });

    expect(verification.checks.length).toBeGreaterThan(8);
  });

  test("rejects reports without reload-equivalent sample text", () => {
    const report = parseFinetuneReport(
      reportFixture({
        sampleText: {
          trained: "adapter sample",
          reloaded: "different sample",
          merged: "merged sample",
        },
      }),
    );

    expect(() => assertFinetuneReport(report)).toThrow("saved adapters reload");
  });

  test("rejects malformed metric and expected option drift", () => {
    expect(() =>
      parseFinetuneReport({
        ...reportFixture(),
        metrics: { ...reportFixture().metrics, targetCount: 0 },
      }),
    ).toThrow("targetCount");

    expect(() =>
      assertFinetuneReport(parseFinetuneReport(reportFixture()), {
        expectedMode: "qlora",
      }),
    ).toThrow("mode is qlora");
  });
});

describe("lora finetune verify-report CLI", () => {
  test("parses help, report paths, and verification flags", () => {
    expect(parseLoRAFinetuneVerifyArgs(["--help"])).toEqual({ kind: "help" });
    expect(
      parseLoRAFinetuneVerifyArgs([
        ".tmp/report.json",
        "--mode",
        "qlora",
        "--adapter-format",
        "peft",
        "--require-loss-not-worse",
      ]),
    ).toEqual({
      kind: "verify",
      reportPath: ".tmp/report.json",
      options: {
        expectedMode: "qlora",
        expectedAdapterFormat: "peft",
        requireLossNotWorse: true,
      },
    });
  });

  test("formats compact AXI help and success", async () => {
    const stdout: string[] = [];
    const exitCode = await runLoRAFinetuneVerifyCommand(["--help"], {
      stdout: (text) => stdout.push(text),
    });

    expect(exitCode).toBe(0);
    expect(stdout.join("\n")).toBe(formatLoRAFinetuneVerifyUsage());
    expect(stdout.join("\n")).toContain("usage[3]");

    const report = parseFinetuneReport(reportFixture());
    const verification = assertFinetuneReport(report);
    expect(formatLoRAFinetuneVerifySuccess(".tmp/report.json", report, verification)).toContain(
      "lora_finetune_report:",
    );
  });

  test("runs success, usage error, and runtime error paths with AXI stdout", async () => {
    const successStdout: string[] = [];
    const successCode = await runLoRAFinetuneVerifyCommand(["report.json"], {
      stdout: (text) => successStdout.push(text),
      readText: async () => JSON.stringify(reportFixture()),
    });

    expect(successCode).toBe(0);
    expect(successStdout.join("\n")).toContain("status: passed");
    expect(successStdout.join("\n")).toContain("failed_checks: 0");

    const usageStdout: string[] = [];
    const usageCode = await runLoRAFinetuneVerifyCommand([], {
      stdout: (text) => usageStdout.push(text),
      readText: async () => {
        throw new Error("should not read");
      },
    });

    expect(usageCode).toBe(2);
    expect(usageStdout.join("\n")).toContain('code: "usage"');
    expect(usageStdout.join("\n")).toContain("report path is required");

    const runtimeStdout: string[] = [];
    const runtimeCode = await runLoRAFinetuneVerifyCommand(["report.json"], {
      stdout: (text) => runtimeStdout.push(text),
      readText: async () =>
        JSON.stringify(
          reportFixture({
            sampleText: {
              trained: "before",
              reloaded: "after",
              merged: "merged",
            },
          }),
        ),
    });

    expect(runtimeCode).toBe(1);
    expect(runtimeStdout.join("\n")).toContain('code: "runtime"');
    expect(runtimeStdout.join("\n")).toContain("saved adapters reload");
  });
});
