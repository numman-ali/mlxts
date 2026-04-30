import { describe, expect, test } from "bun:test";

import { defaultOutputDir, defaultQuantizedOutputDir, defaultReportPath, parseArgs } from "./args";

describe("lora finetune helpers", () => {
  test("derives stable default artifact paths", () => {
    const source = "meta-llama/Llama-3.2-1B-Instruct";

    expect(defaultOutputDir(source)).toContain("meta-llama-Llama-3.2-1B-Instruct");
    expect(defaultQuantizedOutputDir(source)).toContain("meta-llama-Llama-3.2-1B-Instruct-4bit");
    expect(defaultReportPath(source)).toContain("meta-llama-Llama-3.2-1B-Instruct-report.json");
  });

  test("parses explicit LoRA example options", () => {
    const parsed = parseArgs([
      "--source",
      "google/gemma-3-1b-it",
      "--mode",
      "qlora",
      "--preset",
      "all-linear",
      "--adapter-format",
      "peft",
      "--dataset-source",
      "jsonl",
      "--dataset-jsonl",
      "data.jsonl",
      "--train-limit",
      "8",
      "--eval-limit",
      "4",
      "--batch-size",
      "2",
      "--steps",
      "3",
      "--max-seq-len",
      "256",
      "--seed",
      "11",
      "--output-dir",
      ".tmp/out",
      "--quantized-output",
      ".tmp/4bit",
      "--report",
      ".tmp/report.json",
    ]);

    expect(parsed.source).toBe("google/gemma-3-1b-it");
    expect(parsed.mode).toBe("qlora");
    expect(parsed.preset).toBe("all-linear");
    expect(parsed.adapterFormat).toBe("peft");
    expect(parsed.datasetSource).toBe("jsonl");
    expect(parsed.datasetJsonlPath).toBe("data.jsonl");
    expect(parsed.trainLimit).toBe(8);
    expect(parsed.evalLimit).toBe(4);
    expect(parsed.batchSize).toBe(2);
    expect(parsed.steps).toBe(3);
    expect(parsed.maxSequenceLength).toBe(256);
    expect(parsed.seed).toBe(11);
    expect(parsed.outputDir).toBe(".tmp/out");
    expect(parsed.quantizedOutputDir).toBe(".tmp/4bit");
    expect(parsed.reportPath).toBe(".tmp/report.json");
  });

  test("defaults QLoRA to all-linear and JSONL requires a path", () => {
    expect(parseArgs(["--mode", "qlora"]).preset).toBe("all-linear");
    expect(() => parseArgs(["--dataset-source", "jsonl"])).toThrow("--dataset-jsonl");
  });
});
