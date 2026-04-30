import { describe, expect, test } from "bun:test";

import {
  formatTrainingProofSuccess,
  formatTrainingProofUsage,
  runTrainingProofCommand,
} from "./cli";
import type { TrainingProofReport } from "./types";

function proofReport(): TrainingProofReport {
  return {
    source: "local-model",
    quantizedOutputDir: ".tmp/proof-4bit",
    adapterOutputDir: ".tmp/proof-adapters",
    datasetSource: "tiny",
    trainLimit: 2,
    evalLimit: 1,
    batchSize: 1,
    steps: 1,
    maxSequenceLength: 128,
    seed: 7,
    dataNotes: ["tiny corpus"],
    stages: [
      {
        stage: "lora",
        evalLoss: { before: 2.5, after: 2.25, delta: -0.25 },
        averageTrainingLoss: 2.4,
        parameterCounts: { total: 100, trainable: 8 },
        memory: { peakBytes: 1024 },
        notes: ["target_count=2"],
      },
    ],
    verification: {
      passed: true,
      checks: [{ id: "stage:lora", passed: true, message: "ok" }],
    },
  };
}

describe("training proof command", () => {
  test("help is compact AXI stdout and does not acquire the runtime lock", async () => {
    const stdout: string[] = [];
    let lockCalls = 0;

    const exitCode = await runTrainingProofCommand(["--help"], {
      stdout: (text) => stdout.push(text),
      acquireLock: () => {
        lockCalls += 1;
        return { [Symbol.dispose]: () => undefined };
      },
    });

    expect(exitCode).toBe(0);
    expect(lockCalls).toBe(0);
    expect(stdout.join("\n")).toBe(formatTrainingProofUsage());
    expect(stdout.join("\n")).toContain("options[14]");
    expect(stdout.join("\n")).toContain("exit_codes[3]");
  });

  test("usage errors use structured stdout before the runtime lock", async () => {
    const stdout: string[] = [];
    let lockCalls = 0;

    const exitCode = await runTrainingProofCommand(["--train-limit", "--steps"], {
      stdout: (text) => stdout.push(text),
      acquireLock: () => {
        lockCalls += 1;
        return { [Symbol.dispose]: () => undefined };
      },
    });

    expect(exitCode).toBe(2);
    expect(lockCalls).toBe(0);
    expect(stdout.join("\n")).toContain('code: "usage"');
    expect(stdout.join("\n")).toContain("Missing value for --train-limit");
  });

  test("success output is structured and progress stays on stderr", async () => {
    const stdout: string[] = [];
    const stderr: string[] = [];
    let lockDepth = 0;
    const report = proofReport();

    const exitCode = await runTrainingProofCommand(["--dataset-source", "tiny"], {
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
      runProof: async (options, progress) => {
        expect(options.datasetSource).toBe("tiny");
        expect(lockDepth).toBe(1);
        progress("loading proof model");
        return report;
      },
    });

    expect(exitCode).toBe(0);
    expect(lockDepth).toBe(0);
    expect(stderr).toEqual(["loading proof model"]);
    expect(stdout.join("\n")).toBe(formatTrainingProofSuccess(reportPath(stdout), report));
    expect(stdout.join("\n")).toContain("training_proof:");
    expect(stdout.join("\n")).toContain("stages[1]");
  });

  test("runtime errors use structured stdout and stderr stack traces", async () => {
    const stdout: string[] = [];
    const stderr: string[] = [];

    const exitCode = await runTrainingProofCommand(["--dataset-source", "tiny"], {
      stdout: (text) => stdout.push(text),
      stderr: (text) => stderr.push(text),
      acquireLock: () => ({ [Symbol.dispose]: () => undefined }),
      runProof: async () => {
        throw new Error("proof failed");
      },
    });

    expect(exitCode).toBe(1);
    expect(stdout.join("\n")).toContain('code: "runtime"');
    expect(stdout.join("\n")).toContain("proof failed");
    expect(stderr.join("\n")).toContain("Error: proof failed");
  });
});

function reportPath(stdout: string[]): string {
  expect(stdout).toHaveLength(1);
  const reportLine = stdout[0]?.split("\n").find((line) => line.trim().startsWith("report:"));
  expect(reportLine).toBeDefined();
  return JSON.parse(reportLine?.split(": ").at(1) ?? '""');
}
