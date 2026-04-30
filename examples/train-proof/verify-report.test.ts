import { describe, expect, test } from "bun:test";

import type { TrainingProofReport } from "./types";
import { assertTrainingProofReport } from "./verification";
import {
  formatVerifyReportError,
  formatVerifyReportSuccess,
  parseVerifyReportArgs,
  runVerifyReportCommand,
} from "./verify-report";

function completeReport(): TrainingProofReport {
  return {
    source: "meta-llama/Llama-3.2-1B-Instruct",
    quantizedOutputDir: ".tmp/training-proof/model-4bit",
    adapterOutputDir: ".tmp/training-proof/adapters",
    datasetSource: "tiny",
    trainLimit: 8,
    evalLimit: 4,
    batchSize: 2,
    steps: 2,
    maxSequenceLength: 128,
    seed: 7,
    dataNotes: ["dataset_source=tiny"],
    stages: [
      {
        stage: "lora",
        evalLoss: { before: 4, after: 3.8, delta: -0.2 },
        averageTrainingLoss: 3.9,
        sampleText: "hello",
        targets: ["layers.0.self_attn.q_proj"],
        parameterCounts: { total: 32, trainable: 8 },
        memory: { peakBytes: 1024 },
        adapterCheck: {
          directory: ".tmp/training-proof/adapters/lora",
          reloadedMergeTargets: ["layers.0.self_attn.q_proj"],
          trainedSampleText: "hello",
          reloadedSampleText: "hello",
          reloadedMergedSampleText: "hello",
        },
        notes: [
          "preset=attention",
          "target_count=1",
          "merged_targets=1",
          "adapter_reloaded_targets=1",
          "trainable_parameters=8",
          "total_parameters=32",
          "peak_memory_bytes=1024",
          "train_examples=8",
          "eval_examples=4",
        ],
      },
    ],
  };
}

describe("training proof verify-report CLI", () => {
  test("parses help and report paths without prompting", () => {
    expect(parseVerifyReportArgs(["--help"])).toEqual({ kind: "help" });
    expect(parseVerifyReportArgs(["report.json"])).toEqual({
      kind: "verify",
      reportPath: "report.json",
    });
    expect(() => parseVerifyReportArgs([])).toThrow("report path is required");
    expect(() => parseVerifyReportArgs(["report.json", "extra.json"])).toThrow(
      "expected exactly one report path",
    );
    expect(() => parseVerifyReportArgs(["-x"])).toThrow('unknown option "-x"');
  });

  test("formats compact AXI success and multiline errors", () => {
    const verification = assertTrainingProofReport(completeReport());
    expect(formatVerifyReportSuccess("report.json", verification)).toBe(
      [
        "training_proof_report:",
        "  status: passed",
        "  report: report.json",
        "  failed_checks: 0",
        `  passed_checks: ${verification.checks.length}`,
      ].join("\n"),
    );
    expect(formatVerifyReportError("first\nsecond", "rerun command")).toBe(
      ["error:", "  message: |", "    first", "    second", "help: rerun command"].join("\n"),
    );
  });

  test("runs success and structured error paths with stable exit codes", async () => {
    const successOutput: string[] = [];
    const successCode = await runVerifyReportCommand(["report.json"], {
      stdout(text) {
        successOutput.push(text);
      },
      async readText() {
        return JSON.stringify(completeReport());
      },
    });

    expect(successCode).toBe(0);
    expect(successOutput.join("\n")).toContain("training_proof_report:");
    expect(successOutput.join("\n")).toContain("status: passed");

    const usageOutput: string[] = [];
    const usageCode = await runVerifyReportCommand([], {
      stdout(text) {
        usageOutput.push(text);
      },
    });

    expect(usageCode).toBe(2);
    expect(usageOutput.join("\n")).toContain("error:");
    expect(usageOutput.join("\n")).toContain("help: bun run examples/train-proof/verify-report.ts");

    const invalidOutput: string[] = [];
    const invalidCode = await runVerifyReportCommand(["report.json"], {
      stdout(text) {
        invalidOutput.push(text);
      },
      async readText() {
        return "{}";
      },
    });

    expect(invalidCode).toBe(1);
    expect(invalidOutput.join("\n")).toContain("error:");
    expect(invalidOutput.join("\n")).toContain("training proof report.stages");
  });
});
