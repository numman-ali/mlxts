import { describe, expect, test } from "bun:test";

import {
  type AgentCacheProbeReport,
  agentCacheProbeFailures,
  parseAgentCacheRegressionArgs,
} from "./regression-agent-cache";

function probe(overrides: Partial<AgentCacheProbeReport> = {}): AgentCacheProbeReport {
  return {
    id: "probe",
    modelId: "model",
    cold: {
      hits: 0,
      misses: 2,
      writes: 2,
      readTokens: 0,
      writeTokens: 256,
    },
    warm: {
      hits: 2,
      misses: 0,
      writes: 2,
      readTokens: 256,
      writeTokens: 256,
    },
    warmClientReadTokens: 128,
    warmClientWriteTokens: 256,
    ...overrides,
  };
}

describe("agent cache regression", () => {
  test("parses default dense scenarios", () => {
    expect(parseAgentCacheRegressionArgs([])).toMatchObject({
      scenarios: ["qwen-dense", "gemma-dense", "multi-dense"],
      qwenModel: "mlx-community/Qwen3.6-27B-4bit",
      gemmaModel: "google/gemma-4-E2B-it",
      reportJson: ".tmp/agent-cache-regression/report.json",
      promptTokens: 128,
      maxTokens: 16,
      requestTimeoutMs: 3_600_000,
      gpuMemoryUtilization: 0.85,
      allowDownload: false,
    });
  });

  test("parses explicit scenario and MoE options", () => {
    expect(
      parseAgentCacheRegressionArgs([
        "--scenarios",
        "qwen-dense,gemma-dense",
        "--include-moe",
        "--include-moe-multi",
        "--qwen-model",
        "qwen",
        "--gemma-model",
        "gemma",
        "--qwen-moe-model",
        "qwen-moe",
        "--gemma-moe-model",
        "gemma-moe",
        "--prompt-tokens",
        "256",
        "--max-tokens",
        "8",
        "--request-timeout-ms",
        "120000",
        "--gpu-memory-utilization",
        "0.7",
        "--report-json",
        ".tmp/report.json",
        "--allow-download",
      ]),
    ).toMatchObject({
      scenarios: ["qwen-dense", "gemma-dense", "qwen-moe", "gemma-moe", "multi-moe"],
      qwenModel: "qwen",
      gemmaModel: "gemma",
      qwenMoeModel: "qwen-moe",
      gemmaMoeModel: "gemma-moe",
      promptTokens: 256,
      maxTokens: 8,
      requestTimeoutMs: 120000,
      gpuMemoryUtilization: 0.7,
      reportJson: ".tmp/report.json",
      allowDownload: true,
    });
  });

  test("rejects unknown scenarios", () => {
    expect(() => parseAgentCacheRegressionArgs(["--scenarios", "unknown"])).toThrow(
      'unknown scenario "unknown"',
    );
  });

  test("accepts a complete cold-plus-warm cache probe", () => {
    expect(agentCacheProbeFailures(probe())).toEqual([]);
  });

  test("requires cold writes and warm hits with client usage evidence", () => {
    expect(
      agentCacheProbeFailures(
        probe({
          cold: { hits: 0, misses: 2, writes: 1, readTokens: 0, writeTokens: 128 },
          warm: { hits: 1, misses: 1, writes: 2, readTokens: 128, writeTokens: 256 },
          warmClientReadTokens: 0,
        }),
      ),
    ).toEqual([
      "cold divergent sessions did not write two retained prompt boundaries",
      "warm divergent session replay did not hit both retained prompt boundaries",
      "OpenAI-compatible chat usage did not report cached prompt tokens",
    ]);
  });
});
