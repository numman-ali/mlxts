import { describe, expect, test } from "bun:test";
import { loadSafetensors } from "@mlxts/core";
import { spawnSync } from "child_process";
import { mkdtempSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

import { saveCheckpoint } from "./checkpoint";
import { GPT_TINY, resolveConfig } from "./config";
import { prepareData } from "./data";
import { GPT } from "./model/gpt";
import { initializeGPT } from "./model/init";
import { createDefaultAdamW } from "./optimizer-defaults";
import { CharTokenizer } from "./tokenizer";
import { train } from "./train";

function runCli(args: string[]) {
  return spawnSync("bun", ["run", "src/cli.ts", ...args], {
    cwd: join(import.meta.dir, ".."),
    encoding: "utf-8",
  });
}

function parseJsonLine(line: string): Record<string, unknown> {
  const parsed: unknown = JSON.parse(line);
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    throw new Error("expected a JSON object line");
  }
  return Object.fromEntries(Object.entries(parsed));
}

function createCheckpointForCli() {
  const tokenizer = CharTokenizer.fromText("to be or not to be");
  const config = resolveConfig(
    { ...GPT_TINY, nLayer: 2, nHead: 2, nEmbd: 32, blockSize: 16, dropout: 0 },
    tokenizer.vocabSize,
  );
  const model = new GPT(config);
  initializeGPT(model, config);
  const directory = mkdtempSync(join(tmpdir(), "nanogpt-cli-"));
  const checkpointPath = join(directory, "sample-checkpoint");

  try {
    saveCheckpoint({ model, kind: "snapshot", config, step: 3, tokenizer, path: checkpointPath });
  } finally {
    model[Symbol.dispose]();
  }

  return checkpointPath;
}

function createCheckpointWithConfigOverride(
  override: Partial<ReturnType<typeof resolveConfig>>,
): string {
  const tokenizer = CharTokenizer.fromText("to be or not to be");
  const config = {
    ...resolveConfig(
      { ...GPT_TINY, nLayer: 2, nHead: 2, nEmbd: 32, blockSize: 16, dropout: 0 },
      tokenizer.vocabSize,
    ),
    ...override,
  };
  const model = new GPT(config);
  initializeGPT(model, config);
  const directory = mkdtempSync(join(tmpdir(), "nanogpt-cli-custom-"));
  const checkpointPath = join(directory, "custom-checkpoint");

  try {
    saveCheckpoint({ model, kind: "snapshot", config, step: 3, tokenizer, path: checkpointPath });
  } finally {
    model[Symbol.dispose]();
  }

  return checkpointPath;
}

function createResumableCheckpointForCli() {
  const text = "abcdefghijklmnopqrstuvwxyz ".repeat(120);
  const tokenizer = CharTokenizer.fromText(text);
  const config = resolveConfig(
    { ...GPT_TINY, nLayer: 2, nHead: 2, nEmbd: 32, blockSize: 16, dropout: 0 },
    tokenizer.vocabSize,
  );
  const model = new GPT(config);
  const optimizer = createDefaultAdamW(1e-3, 0.1);
  initializeGPT(model, config);
  const directory = mkdtempSync(join(tmpdir(), "nanogpt-cli-resume-"));
  const checkpointPath = join(directory, "resume-checkpoint");

  try {
    const { trainTokens, valTokens } = prepareData(tokenizer.encode(text), 0.9);
    train({
      model,
      optimizer,
      config,
      trainTokens,
      valTokens,
      trainConfig: {
        maxSteps: 1,
        batchSize: 1,
        learningRate: 1e-3,
        weightDecay: 0.1,
        warmupSteps: 0,
        minLearningRate: 1e-4,
        gradAccumSteps: 1,
        evalInterval: 1,
        evalSteps: 1,
        logInterval: 1,
        maxGradNorm: 1,
        seed: 42,
      },
    });
    saveCheckpoint({
      model,
      optimizer,
      kind: "resume",
      config,
      step: 1,
      tokenizer,
      path: checkpointPath,
    });
  } finally {
    optimizer[Symbol.dispose]();
    model[Symbol.dispose]();
  }

  return { checkpointPath, dataPath: join(directory, "resume.txt"), text };
}

describe("nanogpt CLI", () => {
  test("prints help", () => {
    const result = runCli(["help"]);
    expect(result.status).toBe(0);
    expect(result.stdout).toContain("nanogpt");
    expect(result.stdout).toContain("--json");
  });

  test("train --help shows training-specific flags", () => {
    const result = runCli(["train", "--help"]);
    expect(result.status).toBe(0);
    expect(result.stdout).toContain("--max-grad-norm");
    expect(result.stdout).toContain("--gradient-checkpointing");
    expect(result.stdout).toContain("--early-stop-patience");
    expect(result.stdout).toContain("--sample-interval");
    expect(result.stdout).not.toContain("--temperature");
  });

  test("generate --help shows generation-specific flags", () => {
    const result = runCli(["generate", "--help"]);
    expect(result.status).toBe(0);
    expect(result.stdout).toContain("--temperature");
    expect(result.stdout).not.toContain("--max-grad-norm");
  });

  test("train rejects unknown flags", () => {
    const result = runCli(["train", "--mystery"]);
    expect(result.status).toBe(1);
    expect(result.stderr).toContain("unknown flag");
  });

  test("generate rejects unknown flags", () => {
    const checkpoint = createCheckpointForCli();
    const result = runCli(["generate", "--checkpoint", checkpoint, "--mystery"]);
    expect(result.status).toBe(1);
    expect(result.stderr).toContain("unknown flag");
  });

  test("generate --json returns structured output", () => {
    const checkpoint = createCheckpointForCli();
    const result = runCli([
      "generate",
      "--checkpoint",
      checkpoint,
      "--prompt",
      "to ",
      "--max-tokens",
      "2",
      "--temperature",
      "0",
      "--json",
    ]);

    expect(result.status).toBe(0);
    const payload = parseJsonLine(result.stdout.trim());
    expect(payload.prompt).toBe("to ");
    expect(typeof payload.text).toBe("string");
  });

  test("generate without a checkpoint exits with user error", () => {
    const result = runCli(["generate"]);
    expect(result.status).toBe(1);
    expect(result.stderr).toContain("--checkpoint");
  });

  test("export writes model weights as safetensors", async () => {
    const checkpoint = createCheckpointForCli();
    const directory = mkdtempSync(join(tmpdir(), "nanogpt-cli-export-"));
    const outputPath = join(directory, "model.safetensors");

    const result = runCli(["export", "--checkpoint", checkpoint, "--output", outputPath]);

    expect(result.status).toBe(0);
    expect(result.stdout.trim()).toBe(outputPath);

    const loaded = await loadSafetensors(outputPath);
    try {
      expect(Object.keys(loaded.tensors).length).toBeGreaterThan(0);
      expect(loaded.metadata).toMatchObject({
        checkpoint,
        kind: "snapshot",
        step: "3",
      });
    } finally {
      for (const tensor of Object.values(loaded.tensors)) {
        tensor.free();
      }
    }
  });

  test("train --json emits structured events and saves checkpoints", () => {
    const directory = mkdtempSync(join(tmpdir(), "nanogpt-cli-train-"));
    const dataPath = join(directory, "tiny.txt");
    const checkpointDir = join(directory, "checkpoints");
    writeFileSync(dataPath, "abcdefghijklmnopqrstuvwxyz ".repeat(120), "utf-8");

    const result = runCli([
      "train",
      "--data",
      dataPath,
      "--max-steps",
      "1",
      "--batch-size",
      "1",
      "--grad-accum",
      "1",
      "--eval-interval",
      "1",
      "--eval-steps",
      "1",
      "--log-interval",
      "1",
      "--snapshot-interval",
      "1",
      "--resume-interval",
      "1",
      "--checkpoint-dir",
      checkpointDir,
      "--json",
    ]);

    expect(result.status).toBe(0);
    const lines = result.stdout
      .trim()
      .split("\n")
      .filter((line) => line.length > 0)
      .map(parseJsonLine);

    expect(lines.some((line) => line.type === "step")).toBe(true);
    expect(lines.some((line) => line.type === "eval")).toBe(true);
    expect(lines.some((line) => line.type === "checkpoint")).toBe(true);
    expect(lines.some((line) => line.type === "sample")).toBe(true);
    expect(lines.some((line) => line.type === "checkpoint" && line.kind === "resume")).toBe(true);
  });

  test("train --json tracks best checkpoints and stops early on validation plateau", () => {
    const directory = mkdtempSync(join(tmpdir(), "nanogpt-cli-early-stop-"));
    const dataPath = join(directory, "tiny.txt");
    const checkpointDir = join(directory, "checkpoints");
    writeFileSync(dataPath, "abcdefghijklmnopqrstuvwxyz ".repeat(120), "utf-8");

    const result = runCli([
      "train",
      "--data",
      dataPath,
      "--max-steps",
      "10",
      "--batch-size",
      "1",
      "--grad-accum",
      "1",
      "--eval-interval",
      "1",
      "--eval-steps",
      "1",
      "--log-interval",
      "1",
      "--snapshot-interval",
      "0",
      "--resume-interval",
      "0",
      "--early-stop-patience",
      "1",
      "--early-stop-min-delta",
      "10",
      "--checkpoint-dir",
      checkpointDir,
      "--json",
    ]);

    expect(result.status).toBe(0);
    const lines = result.stdout
      .trim()
      .split("\n")
      .filter((line) => line.length > 0)
      .map(parseJsonLine);

    expect(lines.some((line) => line.type === "best-checkpoint")).toBe(true);
    expect(lines.some((line) => line.type === "early-stop")).toBe(true);
    const stoppedEvent = lines.find((line) => line.type === "stopped");
    expect(stoppedEvent).toBeDefined();
    expect(stoppedEvent?.reason).toContain("validation loss did not improve");
    expect(typeof stoppedEvent?.bestCheckpointPath).toBe("string");
    const summary = stoppedEvent?.summary;
    if (typeof summary !== "object" || summary === null || Array.isArray(summary)) {
      throw new Error("expected stopped summary object");
    }
    expect("totalSteps" in summary ? summary.totalSteps : undefined).toBe(2);
  });

  test("train applies the gradient checkpointing override to the resolved config", () => {
    const directory = mkdtempSync(join(tmpdir(), "nanogpt-cli-gc-"));
    const dataPath = join(directory, "tiny.txt");
    writeFileSync(dataPath, "abcdefghijklmnopqrstuvwxyz ".repeat(120), "utf-8");

    const result = runCli([
      "train",
      "--preset",
      "gpt-tiny",
      "--gradient-checkpointing",
      "true",
      "--data",
      dataPath,
      "--max-steps",
      "1",
      "--batch-size",
      "1",
      "--grad-accum",
      "1",
      "--eval-interval",
      "1",
      "--eval-steps",
      "1",
      "--log-interval",
      "1",
      "--checkpoint-dir",
      join(directory, "checkpoints"),
      "--json",
    ]);

    expect(result.status).toBe(0);
    const startLine = result.stdout
      .trim()
      .split("\n")
      .map(parseJsonLine)
      .find((line) => line.type === "start");

    expect(startLine).toBeDefined();
    const config = startLine?.config;
    expect(typeof config).toBe("object");
    expect(config).not.toBeNull();
    if (typeof config !== "object" || config === null) {
      throw new Error("expected start event config object");
    }
    expect("gradientCheckpointing" in config ? config.gradientCheckpointing : undefined).toBe(true);
  });

  test("train --json emits samples when --sample-interval is set", () => {
    const directory = mkdtempSync(join(tmpdir(), "nanogpt-cli-sample-"));
    const dataPath = join(directory, "tiny.txt");
    writeFileSync(dataPath, "abcdefghijklmnopqrstuvwxyz ".repeat(120), "utf-8");

    const result = runCli([
      "train",
      "--data",
      dataPath,
      "--max-steps",
      "1",
      "--batch-size",
      "1",
      "--grad-accum",
      "1",
      "--eval-interval",
      "1",
      "--eval-steps",
      "1",
      "--log-interval",
      "1",
      "--sample-interval",
      "1",
      "--json",
    ]);

    expect(result.status).toBe(0);
    const lines = result.stdout
      .trim()
      .split("\n")
      .filter((line) => line.length > 0)
      .map(parseJsonLine);
    const sampleEvents = lines.filter((line) => line.type === "sample");
    expect(sampleEvents.length).toBeGreaterThanOrEqual(2);
  });

  test("train --resume continues from a checkpoint with optimizer state", () => {
    const { checkpointPath, dataPath, text } = createResumableCheckpointForCli();
    writeFileSync(dataPath, text, "utf-8");

    const result = runCli([
      "train",
      "--resume",
      checkpointPath,
      "--data",
      dataPath,
      "--max-steps",
      "2",
      "--batch-size",
      "1",
      "--grad-accum",
      "1",
      "--eval-interval",
      "1",
      "--eval-steps",
      "1",
      "--log-interval",
      "1",
      "--json",
    ]);

    expect(result.status).toBe(0);
    const lines = result.stdout
      .trim()
      .split("\n")
      .filter((line) => line.length > 0)
      .map(parseJsonLine);
    expect(lines.some((line) => line.type === "start" && line.startStep === 1)).toBe(true);
    expect(lines.some((line) => line.type === "step" && line.step === 2)).toBe(true);
    expect(lines.some((line) => line.type === "checkpoint" && typeof line.path === "string")).toBe(
      true,
    );
  });

  test("train --warm-start initializes from a model checkpoint with a fresh optimizer", () => {
    const checkpointPath = createCheckpointForCli();
    const directory = mkdtempSync(join(tmpdir(), "nanogpt-cli-warm-start-"));
    const dataPath = join(directory, "warm-start.txt");
    writeFileSync(dataPath, "to be or not to be ".repeat(40), "utf-8");

    const result = runCli([
      "train",
      "--warm-start",
      checkpointPath,
      "--data",
      dataPath,
      "--max-steps",
      "1",
      "--batch-size",
      "1",
      "--grad-accum",
      "1",
      "--eval-interval",
      "1",
      "--eval-steps",
      "1",
      "--log-interval",
      "1",
      "--json",
    ]);

    expect(result.status).toBe(0);
    const lines = result.stdout
      .trim()
      .split("\n")
      .filter((line) => line.length > 0)
      .map(parseJsonLine);
    expect(lines.some((line) => line.type === "start" && line.startStep === 0)).toBe(true);
    expect(lines.some((line) => line.type === "step" && line.step === 1)).toBe(true);
  });

  test("train --warm-start derives safe defaults from the checkpoint config", () => {
    const checkpointPath = createCheckpointWithConfigOverride({ gradientCheckpointing: true });
    const directory = mkdtempSync(join(tmpdir(), "nanogpt-cli-warm-start-config-"));
    const dataPath = join(directory, "warm-start.txt");
    writeFileSync(dataPath, "to be or not to be ".repeat(40), "utf-8");

    const result = runCli([
      "train",
      "--warm-start",
      checkpointPath,
      "--data",
      dataPath,
      "--max-steps",
      "1",
      "--json",
    ]);

    expect(result.status).toBe(0);
    const lines = result.stdout
      .trim()
      .split("\n")
      .filter((line) => line.length > 0)
      .map(parseJsonLine);
    const start = lines.find((line) => line.type === "start");
    expect(start?.gradAccumSteps).toBe(8);
    expect(start?.batchSize).toBe(1);
    expect(start?.config).toMatchObject({ gradientCheckpointing: true });
  });
});
