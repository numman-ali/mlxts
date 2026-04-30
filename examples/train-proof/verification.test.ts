import { describe, expect, test } from "bun:test";
import type { TrainingProofReport } from "./types";
import {
  assertTrainingProofReport,
  parseTrainingProofReport,
  verificationOptionsFromArgs,
  verifyTrainingProofReport,
} from "./verification";

function metric(before: number, after: number) {
  return {
    before,
    after,
    delta: after - before,
  };
}

function parameterCounts(trainable = 8) {
  return {
    total: 32,
    trainable,
  };
}

function memory() {
  return {
    peakBytes: 1024,
  };
}

function targetPaths(count: number): string[] {
  return Array.from({ length: count }, (_, index) => `layers.${index}.self_attn.q_proj`);
}

function adapterCheck(stage: string, count: number) {
  const sample = `${stage} sample`;
  return {
    directory: `.tmp/training-proof/adapters/${stage}`,
    reloadedMergeTargets: targetPaths(count),
    trainedSampleText: sample,
    reloadedSampleText: sample,
    reloadedMergedSampleText: sample,
  };
}

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
        evalLoss: metric(4, 3.8),
        averageTrainingLoss: 3.9,
        sampleText: "hello",
        targets: targetPaths(8),
        parameterCounts: parameterCounts(),
        memory: memory(),
        adapterCheck: adapterCheck("lora", 8),
        notes: [
          "preset=attention",
          "target_count=8",
          "merged_targets=8",
          "adapter_reloaded_targets=8",
          "trainable_parameters=8",
          "total_parameters=32",
          "peak_memory_bytes=1024",
          "train_examples=8",
          "eval_examples=4",
        ],
      },
      {
        stage: "qlora",
        evalLoss: metric(4.1, 4),
        averageTrainingLoss: 4.05,
        sampleText: "hello",
        targets: targetPaths(16),
        parameterCounts: parameterCounts(16),
        memory: memory(),
        adapterCheck: adapterCheck("qlora", 16),
        notes: [
          "preset=all-linear",
          "target_count=16",
          "merged_targets=16",
          "adapter_reloaded_targets=16",
          "quantized_base_preserved=true",
          "trainable_parameters=16",
          "total_parameters=32",
          "peak_memory_bytes=1024",
          "train_examples=8",
          "eval_examples=4",
        ],
      },
      {
        stage: "sft",
        evalLoss: metric(4.2, 4.1),
        averageTrainingLoss: 4.15,
        sampleText: "hello",
        parameterCounts: parameterCounts(32),
        memory: memory(),
        notes: [
          "dense_model=true",
          "trainable_parameters=32",
          "total_parameters=32",
          "peak_memory_bytes=1024",
          "train_examples=8",
          "eval_examples=4",
        ],
      },
      {
        stage: "dpo",
        evalLoss: metric(0.7, 0.6),
        rewardAccuracy: metric(0.5, 1),
        rewardMargin: metric(0.1, 0.2),
        chosenReward: metric(0.2, 0.3),
        rejectedReward: metric(0.1, 0.05),
        chosenLogProb: metric(-10, -9),
        rejectedLogProb: metric(-11, -12),
        rawPreferenceAccuracy: metric(0.5, 1),
        averageTrainingLoss: 0.65,
        sampleText: "hello",
        targets: targetPaths(8),
        parameterCounts: parameterCounts(),
        memory: memory(),
        adapterCheck: adapterCheck("dpo", 8),
        notes: [
          "reference_model=frozen_copy",
          "dpo_profile=canonical",
          "preset=attention",
          "target_count=8",
          "merged_targets=8",
          "adapter_reloaded_targets=8",
          "rank=8",
          "alpha=16",
          "dropout=0",
          "learning_rate=0.00005",
          "beta=0.1",
          "last_layers=2",
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

describe("training proof report verification", () => {
  test("accepts a complete machine-checkable report", () => {
    const report = completeReport();
    const verification = assertTrainingProofReport(report, {
      expectedSource: report.source,
      expectedDatasetSource: "tiny",
      expectedTrainLimit: 8,
      expectedEvalLimit: 4,
      expectedBatchSize: 2,
      expectedSteps: 2,
      expectedMaxSequenceLength: 128,
      expectedSeed: 7,
      requiredStages: ["lora", "qlora", "sft", "dpo"],
      expectedDPOProfile: "canonical",
      requireMetricImprovement: true,
    });

    expect(verification.passed).toBe(true);
    expect(verification.checks.every((entry) => entry.passed)).toBe(true);
  });

  test("rejects missing QLoRA quantized-base evidence", () => {
    const report = completeReport();
    const qlora = report.stages.find((stage) => stage.stage === "qlora");
    if (qlora === undefined) {
      throw new Error("test report must include qlora.");
    }
    qlora.notes = qlora.notes.filter((note) => note !== "quantized_base_preserved=true");

    const verification = verifyTrainingProofReport(report, {
      requiredStages: ["lora", "qlora", "sft", "dpo"],
    });

    expect(verification.passed).toBe(false);
    expect(verification.checks.some((entry) => entry.id === "qlora.quantized_base")).toBe(true);
  });

  test("rejects adapter stages without reload evidence", () => {
    const report = completeReport();
    const lora = report.stages.find((stage) => stage.stage === "lora");
    if (lora === undefined) {
      throw new Error("test report must include lora.");
    }
    delete lora.adapterCheck;

    const verification = verifyTrainingProofReport(report, {
      requiredStages: ["lora", "qlora", "sft", "dpo"],
    });

    expect(verification.passed).toBe(false);
    expect(verification.checks.some((entry) => entry.id === "lora.adapter_reload")).toBe(true);
  });

  test("rejects adapter reload evidence when greedy samples drift", () => {
    const report = completeReport();
    const dpo = report.stages.find((stage) => stage.stage === "dpo");
    if (dpo?.adapterCheck === undefined) {
      throw new Error("test report must include dpo adapter evidence.");
    }
    dpo.adapterCheck.reloadedMergedSampleText = "different sample";

    const verification = verifyTrainingProofReport(report, {
      requiredStages: ["lora", "qlora", "sft", "dpo"],
    });

    expect(verification.passed).toBe(false);
    expect(verification.checks.some((entry) => entry.id === "dpo.adapter_reload")).toBe(true);
  });

  test("rejects DPO reports with profile knob drift", () => {
    const report = completeReport();
    const dpo = report.stages.find((stage) => stage.stage === "dpo");
    if (dpo === undefined) {
      throw new Error("test report must include dpo.");
    }
    dpo.notes = dpo.notes.map((note) => (note === "beta=0.1" ? "beta=0.01" : note));

    const verification = verifyTrainingProofReport(report, {
      requiredStages: ["lora", "qlora", "sft", "dpo"],
      expectedDPOProfile: "canonical",
    });

    expect(verification.passed).toBe(false);
    expect(verification.checks.some((entry) => entry.id === "dpo.beta")).toBe(true);
  });

  test("rejects stages without parameter and memory evidence", () => {
    const report = completeReport();
    const sft = report.stages.find((stage) => stage.stage === "sft");
    if (sft === undefined) {
      throw new Error("test report must include sft.");
    }
    delete sft.parameterCounts;
    delete sft.memory;

    const verification = verifyTrainingProofReport(report, {
      requiredStages: ["lora", "qlora", "sft", "dpo"],
    });

    expect(verification.passed).toBe(false);
    expect(verification.checks.some((entry) => entry.id === "sft.parameter_counts")).toBe(true);
    expect(verification.checks.some((entry) => entry.id === "sft.memory_peak")).toBe(true);
  });

  test("parses JSON-compatible report data and rejects malformed metrics", () => {
    const payload = completeReport();
    const stage = payload.stages[0];
    if (stage === undefined) {
      throw new Error("test report must include at least one stage.");
    }
    stage.evalLoss = {
      before: 1,
      after: 0.9,
      delta: 2,
    };

    const parsed = parseTrainingProofReport(payload);
    expect(() => assertTrainingProofReport(parsed)).toThrow("Training proof verification failed");
  });

  test("builds strict verification options from parsed CLI args", () => {
    const options = verificationOptionsFromArgs({
      source: "source",
      quantizedOutputDir: "out",
      adapterOutputDir: "adapters",
      reportPath: "report.json",
      datasetSource: "tiny",
      trainLimit: 2,
      evalLimit: 1,
      batchSize: 1,
      steps: 1,
      maxSequenceLength: 64,
      seed: 3,
      stages: ["lora"],
      dpoProfile: "handbook",
    });

    expect(options).toEqual({
      expectedSource: "source",
      expectedDatasetSource: "tiny",
      expectedTrainLimit: 2,
      expectedEvalLimit: 1,
      expectedBatchSize: 1,
      expectedSteps: 1,
      expectedMaxSequenceLength: 64,
      expectedSeed: 3,
      requiredStages: ["lora"],
      expectedDPOProfile: "handbook",
    });
  });
});
