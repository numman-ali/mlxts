import { describe, expect, test } from "bun:test";
import type { FinetuneReport } from "./args";
import { formatLoRAFinetuneSuccess, formatLoRAFinetuneUsage, runLoRAFinetuneCommand } from "./cli";

function finetuneReport(): FinetuneReport {
  return {
    source: "local-model",
    mode: "lora",
    preset: "attention",
    adapterFormat: "mlxts",
    datasetSource: "tiny",
    trainLimit: 2,
    evalLimit: 1,
    batchSize: 1,
    steps: 1,
    maxSequenceLength: 128,
    outputDir: ".tmp/lora-finetune/local-model",
    adapterDir: ".tmp/lora-finetune/local-model/adapter",
    metrics: {
      evalLossBefore: 2.5,
      evalLossAfter: 2.25,
      averageTrainingLoss: 2.4,
      trainingStepLosses: [{ step: 1, loss: 2.4 }],
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
        kept: 2,
        skippedMalformed: 0,
        skippedLong: 0,
      },
      eval: {
        kept: 1,
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
      trained: "hi from trained",
      reloaded: "hi from trained",
      merged: "hi from merged",
    },
  };
}

describe("lora finetune command", () => {
  test("help is compact AXI stdout and does not acquire the runtime lock", async () => {
    const stdout: string[] = [];
    let lockCalls = 0;

    const exitCode = await runLoRAFinetuneCommand(["--help"], {
      stdout: (text) => stdout.push(text),
      acquireLock: () => {
        lockCalls += 1;
        return { [Symbol.dispose]: () => undefined };
      },
    });

    expect(exitCode).toBe(0);
    expect(lockCalls).toBe(0);
    expect(stdout.join("\n")).toBe(formatLoRAFinetuneUsage());
    expect(stdout.join("\n")).toContain("options[16]");
    expect(stdout.join("\n")).toContain("exit_codes[3]");
  });

  test("usage errors use structured stdout before the runtime lock", async () => {
    const stdout: string[] = [];
    let lockCalls = 0;

    const exitCode = await runLoRAFinetuneCommand(["--train-limit", "--steps"], {
      stdout: (text) => stdout.push(text),
      acquireLock: () => {
        lockCalls += 1;
        return { [Symbol.dispose]: () => undefined };
      },
    });

    expect(exitCode).toBe(2);
    expect(lockCalls).toBe(0);
    expect(stdout.join("\n")).toContain('code: "usage"');
    expect(stdout.join("\n")).toContain("missing value for --train-limit");
  });

  test("success output is structured and progress stays on stderr", async () => {
    const stdout: string[] = [];
    const stderr: string[] = [];
    let lockDepth = 0;
    const report = finetuneReport();

    const exitCode = await runLoRAFinetuneCommand(["--dataset-source", "tiny"], {
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
      runFinetune: async (options, progress) => {
        expect(options.datasetSource).toBe("tiny");
        expect(lockDepth).toBe(1);
        progress("loading lora model");
        return report;
      },
    });

    expect(exitCode).toBe(0);
    expect(lockDepth).toBe(0);
    expect(stderr).toEqual(["loading lora model"]);
    expect(stdout.join("\n")).toBe(formatLoRAFinetuneSuccess(reportPath(stdout), report));
    expect(stdout.join("\n")).toContain("lora_finetune:");
    expect(stdout.join("\n")).toContain("eval_loss_delta: -0.25");
  });

  test("runtime errors use structured stdout and stderr stack traces", async () => {
    const stdout: string[] = [];
    const stderr: string[] = [];

    const exitCode = await runLoRAFinetuneCommand(["--dataset-source", "tiny"], {
      stdout: (text) => stdout.push(text),
      stderr: (text) => stderr.push(text),
      acquireLock: () => ({ [Symbol.dispose]: () => undefined }),
      runFinetune: async () => {
        throw new Error("finetune failed");
      },
    });

    expect(exitCode).toBe(1);
    expect(stdout.join("\n")).toContain('code: "runtime"');
    expect(stdout.join("\n")).toContain("finetune failed");
    expect(stderr.join("\n")).toContain("Error: finetune failed");
  });
});

function reportPath(stdout: string[]): string {
  expect(stdout).toHaveLength(1);
  const reportLine = stdout[0]?.split("\n").find((line) => line.trim().startsWith("report:"));
  expect(reportLine).toBeDefined();
  return JSON.parse(reportLine?.split(": ").at(1) ?? '""');
}
