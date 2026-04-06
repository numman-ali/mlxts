import { describe, expect, test } from "bun:test";

import {
  createTrainingProofCorpus,
  DEFAULT_PROOF_BATCH_SIZE,
  DEFAULT_PROOF_DATASET_SOURCE,
  DEFAULT_PROOF_EVAL_LIMIT,
  DEFAULT_PROOF_MODEL,
  DEFAULT_PROOF_STEPS,
  DEFAULT_PROOF_TRAIN_LIMIT,
  defaultQuantizedOutputDir,
  defaultReportPath,
  parseTrainingProofArgs,
  parseUltrachatMessagesRow,
  parseUltrafeedbackPreferenceRow,
} from "./proof";

describe("training proof helpers", () => {
  test("derive stable default output paths", () => {
    const quantizedOutputDir = defaultQuantizedOutputDir(DEFAULT_PROOF_MODEL);
    const reportPath = defaultReportPath(DEFAULT_PROOF_MODEL);

    expect(quantizedOutputDir).toContain("meta-llama-Llama-3.2-1B-Instruct-4bit");
    expect(reportPath).toContain("meta-llama-Llama-3.2-1B-Instruct-report.json");
  });

  test("parses CLI overrides", () => {
    const parsed = parseTrainingProofArgs([
      "--source",
      "meta-llama/Llama-3.2-3B-Instruct",
      "--dataset-source",
      "tiny",
      "--train-limit",
      "8",
      "--eval-limit",
      "4",
      "--batch-size",
      "2",
      "--steps",
      "3",
      "--quantized-output",
      "/tmp/proof-4bit",
      "--report",
      "/tmp/proof.json",
    ]);

    expect(parsed.source).toBe("meta-llama/Llama-3.2-3B-Instruct");
    expect(parsed.datasetSource).toBe("tiny");
    expect(parsed.trainLimit).toBe(8);
    expect(parsed.evalLimit).toBe(4);
    expect(parsed.batchSize).toBe(2);
    expect(parsed.steps).toBe(3);
    expect(parsed.quantizedOutputDir).toBe("/tmp/proof-4bit");
    expect(parsed.reportPath).toBe("/tmp/proof.json");
  });

  test("uses realistic proof defaults", () => {
    const parsed = parseTrainingProofArgs([]);

    expect(parsed.datasetSource).toBe(DEFAULT_PROOF_DATASET_SOURCE);
    expect(parsed.trainLimit).toBe(DEFAULT_PROOF_TRAIN_LIMIT);
    expect(parsed.evalLimit).toBe(DEFAULT_PROOF_EVAL_LIMIT);
    expect(parsed.batchSize).toBe(DEFAULT_PROOF_BATCH_SIZE);
    expect(parsed.steps).toBe(DEFAULT_PROOF_STEPS);
  });

  test("builds a canonical small proof corpus", () => {
    const corpus = createTrainingProofCorpus();

    expect(corpus.supervisionExamples).toHaveLength(2);
    expect(corpus.supervisionExamples[0]?.at(-1)?.role).toBe("assistant");
    expect(corpus.promptMessages).toHaveLength(2);
    expect(corpus.chosen.role).toBe("assistant");
    expect(corpus.rejected.role).toBe("assistant");
  });

  test("parses ultrachat rows into chat messages", () => {
    const messages = parseUltrachatMessagesRow({
      messages: [
        { role: "user", content: "Hello" },
        { role: "assistant", content: "Hi there" },
      ],
    });

    expect(messages).toEqual([
      { role: "user", content: "Hello" },
      { role: "assistant", content: "Hi there" },
    ]);
  });

  test("parses ultrafeedback rows into prompt/chosen/rejected turns", () => {
    const parsed = parseUltrafeedbackPreferenceRow({
      chosen: [
        { role: "user", content: "Hello" },
        { role: "assistant", content: "Helpful answer" },
      ],
      rejected: [
        { role: "user", content: "Hello" },
        { role: "assistant", content: "Bad answer" },
      ],
    });

    expect(parsed.promptMessages).toEqual([{ role: "user", content: "Hello" }]);
    expect(parsed.chosen.content).toBe("Helpful answer");
    expect(parsed.rejected.content).toBe("Bad answer");
  });
});
