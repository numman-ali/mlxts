import { describe, expect, test } from "bun:test";

import {
  type AgentCacheProbeReport,
  type AgentCacheRegressionReport,
  agentCacheProbeFailures,
  formatAgentCacheRegressionError,
  formatAgentCacheRegressionSuccess,
  formatAgentCacheRegressionUsage,
  parseAgentCacheRegressionArgs,
  runAgentCacheRegressionCommand,
} from "./regression-agent-cache";

function parsedOptions(argv: readonly string[]) {
  const command = parseAgentCacheRegressionArgs(argv);
  if (command.kind !== "run") {
    throw new Error("expected run command");
  }
  return command.options;
}

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
    exactReplay: {
      hits: 1,
      misses: 0,
      writes: 1,
      readTokens: 128,
      writeTokens: 128,
    },
    warmClientUsage: [
      { session: "A", readTokens: 64, writeTokens: 128 },
      { session: "B", readTokens: 64, writeTokens: 128 },
    ],
    warmClientReadTokens: 128,
    warmClientWriteTokens: 256,
    exactReplayClientReadTokens: 128,
    exactReplayClientWriteTokens: 128,
    ...overrides,
  };
}

describe("agent cache regression", () => {
  test("parses default dense scenarios", () => {
    expect(parseAgentCacheRegressionArgs([])).toMatchObject({
      kind: "run",
      options: {
        scenarios: ["qwen-dense", "gemma-dense", "multi-dense"],
        qwenModel: "mlx-community/Qwen3.6-27B-4bit",
        gemmaModel: "google/gemma-4-E2B-it",
        reportJson: ".tmp/agent-cache-regression/report.json",
        promptTokens: 128,
        maxTokens: 16,
        requestTimeoutMs: 3_600_000,
        gpuMemoryUtilization: 0.85,
        allowDownload: false,
      },
    });
  });

  test("parses help without exiting", () => {
    expect(parseAgentCacheRegressionArgs(["--help"])).toEqual({ kind: "help" });
    expect(parseAgentCacheRegressionArgs(["-h"])).toEqual({ kind: "help" });
  });

  test("parses explicit scenario and MoE options", () => {
    expect(
      parsedOptions([
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

  test("rejects malformed usage", () => {
    expect(() => parseAgentCacheRegressionArgs(["--scenarios", "unknown"])).toThrow(
      'unknown scenario "unknown"',
    );
    expect(() => parseAgentCacheRegressionArgs(["--scenarios", ""])).toThrow(
      "--scenarios requires a value",
    );
    expect(() => parseAgentCacheRegressionArgs(["--prompt-tokens", "12abc"])).toThrow(
      "--prompt-tokens must be a positive integer",
    );
    expect(() => parseAgentCacheRegressionArgs(["--max-tokens", "0"])).toThrow(
      "--max-tokens must be a positive integer",
    );
    expect(() => parseAgentCacheRegressionArgs(["--gpu-memory-utilization", "1.1"])).toThrow(
      "--gpu-memory-utilization must be between 0 and 1",
    );
    expect(() => parseAgentCacheRegressionArgs(["extra"])).toThrow('unexpected argument "extra"');
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
    expect(
      agentCacheProbeFailures(
        probe({
          warmClientUsage: [
            { session: "A", readTokens: 64, writeTokens: 128 },
            { session: "B", readTokens: 0, writeTokens: 128 },
          ],
          exactReplay: { hits: 0, misses: 1, writes: 1, readTokens: 0, writeTokens: 128 },
          exactReplayClientReadTokens: 0,
        }),
      ),
    ).toEqual([
      "each warm replay session must report cached prompt tokens",
      "exact A replay after divergent A/B did not hit the retained prompt boundary",
      "exact A replay did not report cached prompt tokens",
    ]);
  });

  test("formats compact AXI success and error output", () => {
    const report: AgentCacheRegressionReport = {
      createdAt: "2026-04-30T00:00:00.000Z",
      promptTokens: 128,
      maxTokens: 16,
      scenarios: [
        {
          id: "multi-dense",
          models: ["qwen-dense-local", "gemma-dense-local"],
          targets: [
            { id: "qwen-dense-local", source: "qwen" },
            { id: "gemma-dense-local", source: "gemma" },
          ],
          probes: [
            probe({ modelId: "qwen-dense-local", warmClientReadTokens: 128 }),
            probe({ modelId: "gemma-dense-local", warmClientReadTokens: 256 }),
          ],
          status: "passed",
        },
      ],
    };
    expect(formatAgentCacheRegressionSuccess(report, ".tmp/report.json")).toBe(
      [
        "agent_cache_regression:",
        "  status: passed",
        "  scenarios: 1",
        '  report_json: ".tmp/report.json"',
        "scenarios[1]{id,status,models,warm_hits,warm_read_tokens,warm_client_read_tokens,exact_replay_hits,exact_replay_client_read_tokens}:",
        '  "multi-dense","passed","qwen-dense-local|gemma-dense-local",4,512,384,2,256',
      ].join("\n"),
    );
    expect(formatAgentCacheRegressionError("bad flag", "rerun")).toBe(
      ["error:", '  message: "bad flag"', 'help: "rerun"'].join("\n"),
    );
    expect(formatAgentCacheRegressionError("bad\nflag", "rerun")).toContain("  message: |");
  });

  test("runs help, success, usage error, and runtime error paths with AXI stdout", async () => {
    const helpStdout: string[] = [];
    expect(
      await runAgentCacheRegressionCommand(["--help"], {
        stdout: (text) => helpStdout.push(text),
      }),
    ).toBe(0);
    expect(helpStdout.join("\n")).toBe(formatAgentCacheRegressionUsage());

    const stdout: string[] = [];
    const stderr: string[] = [];
    let lockDepth = 0;
    const writtenReports: string[] = [];
    expect(
      await runAgentCacheRegressionCommand(["--scenarios", "qwen-dense"], {
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
        runRegression: async (options, progress) => {
          expect(options.scenarios).toEqual(["qwen-dense"]);
          expect(lockDepth).toBe(1);
          progress("agent-cache-regression: probe");
          return {
            createdAt: "2026-04-30T00:00:00.000Z",
            promptTokens: options.promptTokens,
            maxTokens: options.maxTokens,
            scenarios: [
              {
                id: "qwen-dense",
                models: ["qwen-dense-local"],
                targets: [{ id: "qwen-dense-local", source: options.qwenModel }],
                probes: [probe({ modelId: "qwen-dense-local" })],
                status: "passed",
              },
            ],
          };
        },
        writeReport: async (path) => {
          writtenReports.push(path);
        },
      }),
    ).toBe(0);
    expect(lockDepth).toBe(0);
    expect(stderr).toEqual(["agent-cache-regression: probe"]);
    expect(writtenReports).toEqual([".tmp/agent-cache-regression/report.json"]);
    expect(stdout.join("\n")).toContain("agent_cache_regression:");

    const usageStdout: string[] = [];
    let usageLockCalls = 0;
    expect(
      await runAgentCacheRegressionCommand(["--prompt-tokens", "0"], {
        stdout: (text) => usageStdout.push(text),
        acquireLock: () => {
          usageLockCalls += 1;
          return { [Symbol.dispose]: () => undefined };
        },
      }),
    ).toBe(2);
    expect(usageLockCalls).toBe(0);
    expect(usageStdout.join("\n")).toContain("--prompt-tokens must be a positive integer");

    const runtimeStdout: string[] = [];
    expect(
      await runAgentCacheRegressionCommand([], {
        stdout: (text) => runtimeStdout.push(text),
        runRegression: async () => {
          throw new Error("cache regression failed");
        },
      }),
    ).toBe(1);
    expect(runtimeStdout.join("\n")).toContain("cache regression failed");
  });
});
