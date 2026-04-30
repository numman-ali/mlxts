import { describe, expect, test } from "bun:test";

import {
  assertQwenImageRegressionReport,
  createQwenImageRegressionData,
  parseQwenImageRegressionArgs,
  type QwenImageProbeReport,
  type QwenImageRegressionReport,
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
      qwenModel: "mlx-community/Qwen3.6-27B-4bit",
      reportDir: ".tmp/qwen-image-regression",
      allowDownload: false,
      requestTimeoutMs: 600_000,
      port: 0,
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
      qwenModel: "local/qwen",
      reportDir: ".tmp/custom",
      allowDownload: true,
      requestTimeoutMs: 1234,
      port: 8173,
    });
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
});
