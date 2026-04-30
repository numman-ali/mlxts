import { describe, expect, test } from "bun:test";

import {
  assertQwenImageRegressionReport,
  createQwenImageRegressionData,
  formatQwenImageRegressionError,
  formatQwenImageRegressionSuccess,
  formatQwenImageRegressionUsage,
  parseQwenImageRegressionArgs,
  type QwenImageProbeReport,
  type QwenImageRegressionReport,
  runQwenImageRegressionCommand,
} from "./regression-qwen-image";

function probeProtocol(label: string): string {
  if (label.startsWith("openai-chat")) {
    return "openai.chat_completions";
  }
  return label.startsWith("openai-responses") ? "openai.responses" : "anthropic.messages";
}

function probePath(label: string): string {
  if (label.startsWith("openai-chat")) {
    return "/v1/chat/completions";
  }
  return label.startsWith("openai-responses") ? "/v1/responses" : "/v1/messages";
}

function healthyProbe(label: string, overrides: Partial<QwenImageProbeReport> = {}) {
  const isRepeat = label.endsWith("-repeat");
  return {
    label,
    protocol: probeProtocol(label),
    path: probePath(label),
    status: 200,
    durationMs: 100,
    outputText: "A red, green, blue, and yellow quadrant image.",
    finishReason: "stop",
    promptTokens: 93,
    completionTokens: 16,
    totalTokens: 109,
    cacheReadTokens: isRepeat ? 92 : 0,
    cacheWriteTokens: isRepeat ? 0 : 92,
    promptPrepareEvents: 1,
    promptPrepareMs: 12,
    promptCache: {
      hits: isRepeat ? 1 : 0,
      misses: isRepeat ? 0 : 1,
      writes: isRepeat ? 0 : 1,
      readTokens: isRepeat ? 92 : 0,
      writeTokens: isRepeat ? 0 : 92,
    },
    routeDecisions: [
      {
        id: label,
        protocol: probeProtocol(label),
        route: "single",
        reason: "media_input",
        inputKind: "content",
      },
    ],
    continuousSchedulerPhases: 0,
    memoryPeakBytes: 1_000_000,
    ...overrides,
  } satisfies QwenImageProbeReport;
}

function healthyReport(overrides: Partial<QwenImageRegressionReport> = {}) {
  return {
    createdAt: "2026-04-30T00:00:00.000Z",
    model: "mlx-community/Qwen3.6-27B-4bit",
    reportVersion: 1,
    image: { mediaType: "image/bmp", width: 96, height: 96 },
    probes: [
      healthyProbe("openai-chat-cold"),
      healthyProbe("openai-chat-repeat"),
      healthyProbe("openai-responses-cold"),
      healthyProbe("openai-responses-repeat"),
      healthyProbe("anthropic-messages-cold"),
      healthyProbe("anthropic-messages-repeat"),
    ],
    ...overrides,
  } satisfies QwenImageRegressionReport;
}

describe("qwen image regression harness", () => {
  test("parses default and explicit options", () => {
    expect(parseQwenImageRegressionArgs([])).toEqual({
      kind: "run",
      options: {
        qwenModel: "mlx-community/Qwen3.6-27B-4bit",
        reportDir: ".tmp/qwen-image-regression",
        allowDownload: false,
        requestTimeoutMs: 600_000,
        port: 0,
      },
    });

    expect(
      parseQwenImageRegressionArgs([
        "--qwen-model",
        "local/qwen",
        "--report-dir",
        ".tmp/custom",
        "--request-timeout-ms",
        "1234",
        "--port",
        "8173",
        "--allow-download",
      ]),
    ).toEqual({
      kind: "run",
      options: {
        qwenModel: "local/qwen",
        reportDir: ".tmp/custom",
        allowDownload: true,
        requestTimeoutMs: 1234,
        port: 8173,
      },
    });
  });

  test("parses help and rejects usage errors without exiting", () => {
    expect(parseQwenImageRegressionArgs(["--help"])).toEqual({ kind: "help" });
    expect(parseQwenImageRegressionArgs(["-h"])).toEqual({ kind: "help" });
    expect(() => parseQwenImageRegressionArgs(["--qwen-model", ""])).toThrow(
      "--qwen-model requires a value.",
    );
    expect(() => parseQwenImageRegressionArgs(["--request-timeout-ms", "0"])).toThrow(
      "--request-timeout-ms must be a positive integer.",
    );
    expect(() => parseQwenImageRegressionArgs(["--request-timeout-ms", "1.5"])).toThrow(
      "--request-timeout-ms must be a positive integer.",
    );
    expect(() => parseQwenImageRegressionArgs(["--request-timeout-ms", "123abc"])).toThrow(
      "--request-timeout-ms must be a positive integer.",
    );
    expect(() => parseQwenImageRegressionArgs(["--port", "-1"])).toThrow(
      "--port must be a non-negative integer.",
    );
    expect(() => parseQwenImageRegressionArgs(["-x"])).toThrow('unknown option "-x".');
  });

  test("creates a deterministic BMP data URL", () => {
    const image = createQwenImageRegressionData();
    expect(image).toMatchObject({ width: 96, height: 96 });
    expect(image.dataUrl.startsWith("data:image/bmp;base64,")).toBe(true);
    expect(atob(image.base64).slice(0, 2)).toBe("BM");
  });

  test("accepts reports with media routing and repeat cache hits", () => {
    expect(() => assertQwenImageRegressionReport(healthyReport())).not.toThrow();
  });

  test("rejects missing visible output, scheduler use, and missing repeat cache", () => {
    const report = healthyReport({
      probes: [
        healthyProbe("openai-chat-cold", { outputText: "" }),
        healthyProbe("openai-chat-repeat", { cacheReadTokens: 0 }),
        healthyProbe("openai-responses-cold", { continuousSchedulerPhases: 1 }),
        healthyProbe("openai-responses-repeat"),
        healthyProbe("anthropic-messages-cold"),
        healthyProbe("anthropic-messages-repeat"),
      ],
    });

    expect(() => assertQwenImageRegressionReport(report)).toThrow("Qwen image regression failed");
  });

  test("formats compact AXI success and error output", () => {
    expect(formatQwenImageRegressionSuccess(".tmp/report.json", healthyReport())).toBe(
      [
        "qwen_image_regression:",
        "  status: passed",
        '  report: ".tmp/report.json"',
        '  model: "mlx-community/Qwen3.6-27B-4bit"',
        "  probes: 6",
        "  repeat_cache_hits: 3/3",
        "  cache_read_tokens: 276",
        "  cache_write_tokens: 276",
        "  max_memory_peak_bytes: 1000000",
        "probes[6]{label,protocol,route,cache_read_tokens,cache_write_tokens,output_chars,duration_ms}:",
        '  "openai-chat-cold","openai.chat_completions","single:media_input",0,92,46,100',
        '  "openai-chat-repeat","openai.chat_completions","single:media_input",92,0,46,100',
        '  "openai-responses-cold","openai.responses","single:media_input",0,92,46,100',
        '  "openai-responses-repeat","openai.responses","single:media_input",92,0,46,100',
        '  "anthropic-messages-cold","anthropic.messages","single:media_input",0,92,46,100',
        '  "anthropic-messages-repeat","anthropic.messages","single:media_input",92,0,46,100',
      ].join("\n"),
    );

    expect(formatQwenImageRegressionError("bad flag", "bun run regression:qwen-image")).toBe(
      ["error:", '  message: "bad flag"', 'help: "bun run regression:qwen-image"'].join("\n"),
    );
    expect(formatQwenImageRegressionError("bad\nflag", "rerun")).toContain("  message: |");
  });

  test("runs help, success, usage error, and runtime error paths with AXI stdout", async () => {
    const noOpLock = { [Symbol.dispose]() {} };
    let lockCalls = 0;
    const helpStdout: string[] = [];
    expect(
      await runQwenImageRegressionCommand(["--help"], {
        stdout: (text) => helpStdout.push(text),
        acquireLock: () => {
          lockCalls += 1;
          return noOpLock;
        },
      }),
    ).toBe(0);
    expect(lockCalls).toBe(0);
    expect(helpStdout.join("\n")).toBe(formatQwenImageRegressionUsage());

    const stdout: string[] = [];
    const stderr: string[] = [];
    expect(
      await runQwenImageRegressionCommand(["--report-dir", ".tmp/custom"], {
        stdout: (text) => stdout.push(text),
        stderr: (text) => stderr.push(text),
        acquireLock: () => noOpLock,
        runRegression: async (options, progress) => {
          expect(options.reportDir).toBe(".tmp/custom");
          progress("probe=openai-chat-cold duration_ms=100.0");
          return healthyReport();
        },
        writeReport: async (reportDir) => `${reportDir}/qwen-image-regression.json`,
      }),
    ).toBe(0);
    expect(stderr).toEqual(["probe=openai-chat-cold duration_ms=100.0"]);
    expect(stdout.join("\n")).toContain("qwen_image_regression:");
    expect(stdout.join("\n")).toContain('  report: ".tmp/custom/qwen-image-regression.json"');

    const usageStdout: string[] = [];
    expect(
      await runQwenImageRegressionCommand(["--port", "-1"], {
        stdout: (text) => usageStdout.push(text),
        acquireLock: () => noOpLock,
      }),
    ).toBe(2);
    expect(lockCalls).toBe(0);
    expect(usageStdout.join("\n")).toContain("error:");
    expect(usageStdout.join("\n")).toContain("--port must be a non-negative integer.");

    const runtimeStdout: string[] = [];
    expect(
      await runQwenImageRegressionCommand([], {
        stdout: (text) => runtimeStdout.push(text),
        acquireLock: () => noOpLock,
        runRegression: async () => {
          throw new Error("server failed");
        },
      }),
    ).toBe(1);
    expect(runtimeStdout.join("\n")).toContain("server failed");
  });
});
