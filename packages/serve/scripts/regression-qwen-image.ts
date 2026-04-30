#!/usr/bin/env bun

import { mkdirSync } from "fs";
import { join } from "path";
import { acquireRuntimeCommandLock } from "../../../scripts/runtime-command-lock";
import { serveModel } from "../src/model-loading/server";
import type { ServeEvent } from "../src/types";

type CliOptions = {
  qwenModel: string;
  reportDir: string;
  allowDownload: boolean;
  requestTimeoutMs: number;
  port: number;
};

type QwenImageRegressionCommand = { kind: "help" } | { kind: "run"; options: CliOptions };

type RuntimeLock = {
  [Symbol.dispose](): void;
};

type QwenImageRegressionRuntime = {
  stdout?: (text: string) => void;
  stderr?: (text: string) => void;
  acquireLock?: () => RuntimeLock;
  runRegression?: (
    options: CliOptions,
    progress: (text: string) => void,
  ) => Promise<QwenImageRegressionReport>;
  writeReport?: (reportDir: string, report: QwenImageRegressionReport) => Promise<string>;
};

type RecordedServeEvent = ServeEvent & {
  observedAtMs: number;
};

type RouteDecisionReport = {
  id: string;
  protocol: string;
  route: string;
  reason: string;
  inputKind?: string;
};

type PromptCacheReport = {
  hits: number;
  misses: number;
  writes: number;
  readTokens: number;
  writeTokens: number;
};

export type QwenImageProbeReport = {
  label: string;
  protocol: string;
  path: string;
  status: number;
  durationMs: number;
  outputText: string;
  finishReason: string | null;
  promptTokens: number | null;
  completionTokens: number | null;
  totalTokens: number | null;
  cacheReadTokens: number;
  cacheWriteTokens: number;
  promptPrepareEvents: number;
  promptPrepareMs: number | null;
  promptCache: PromptCacheReport;
  routeDecisions: RouteDecisionReport[];
  continuousSchedulerPhases: number;
  memoryPeakBytes: number | null;
};

export type QwenImageRegressionReport = {
  createdAt: string;
  model: string;
  reportVersion: 1;
  image: {
    mediaType: "image/bmp";
    width: number;
    height: number;
  };
  probes: QwenImageProbeReport[];
};

type UsageReport = {
  promptTokens: number | null;
  completionTokens: number | null;
  totalTokens: number | null;
  cacheReadTokens: number;
  cacheWriteTokens: number;
};

type ProtocolProbe = {
  label: string;
  protocol: string;
  path: string;
  body: unknown;
  output: (body: unknown) => {
    outputText: string;
    finishReason: string | null;
    usage: UsageReport;
  };
};

const DEFAULT_MODEL = "mlx-community/Qwen3.6-27B-4bit";
const API_KEY = "mlxts-regression";
const IMAGE_WIDTH = 96;
const IMAGE_HEIGHT = 96;
const MAX_OUTPUT_TOKENS = 64;
const PROMPT = "Describe this image in one sentence. Name the quadrant colors and their positions.";

const PROBE_LABELS = [
  "openai-chat-cold",
  "openai-chat-repeat",
  "openai-responses-cold",
  "openai-responses-repeat",
  "anthropic-messages-cold",
  "anthropic-messages-repeat",
] as const;

class QwenImageRegressionUsageError extends Error {}

export function formatQwenImageRegressionUsage(): string {
  return [
    "Usage: bun run packages/serve/scripts/regression-qwen-image.ts [options]",
    "",
    "Runs the Qwen image serving regression across OpenAI Chat, OpenResponses, and Anthropic Messages.",
    "",
    "Options:",
    `  --qwen-model <id>         Qwen image-capable model id/path. Default: ${DEFAULT_MODEL}`,
    "  --report-dir <path>       Directory for qwen-image-regression.json.",
    "  --request-timeout-ms <n>  Client timeout per image request. Default: 600000.",
    "  --port <n>                Server port. Default: 0.",
    "  --allow-download          Allow Hub downloads; default is cached/local only.",
    "  --help                    Show this help.",
    "",
    "Exit codes:",
    "  0  regression passed",
    "  1  runtime or regression failure",
    "  2  usage error",
    "",
    "Examples:",
    "  bun run regression:qwen-image",
    "  bun run regression:qwen-image -- --qwen-model mlx-community/Qwen3.6-27B-4bit --report-dir .tmp/qwen-image",
  ].join("\n");
}

function defaultOptions(): CliOptions {
  return {
    qwenModel: DEFAULT_MODEL,
    reportDir: ".tmp/qwen-image-regression",
    allowDownload: false,
    requestTimeoutMs: 600_000,
    port: 0,
  };
}

function readStringFlag(args: readonly string[], index: number, flag: string): string {
  const value = args[index + 1];
  if (value === undefined || value.trim() === "" || value.startsWith("-")) {
    throw new QwenImageRegressionUsageError(`${flag} requires a value.`);
  }
  return value;
}

function readIntegerFlag(
  args: readonly string[],
  index: number,
  flag: string,
  predicate: (value: number) => boolean,
  description: string,
): number {
  const rawValue = args[index + 1]?.trim();
  if (rawValue === undefined || rawValue === "" || rawValue.startsWith("--")) {
    throw new QwenImageRegressionUsageError(`${flag} requires a value.`);
  }
  const value = /^-?\d+$/.test(rawValue) ? Number(rawValue) : Number.NaN;
  if (!Number.isInteger(value) || !predicate(value)) {
    throw new QwenImageRegressionUsageError(`${flag} must be ${description}.`);
  }
  return value;
}

export function parseQwenImageRegressionArgs(argv: readonly string[]): QwenImageRegressionCommand {
  if (argv.length === 1 && (argv[0] === "--help" || argv[0] === "-h")) {
    return { kind: "help" };
  }
  const options = defaultOptions();
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === undefined) {
      throw new QwenImageRegressionUsageError("argument parsing reached an empty slot.");
    }
    switch (arg) {
      case "--help":
      case "-h":
        return { kind: "help" };
      case "--qwen-model":
        options.qwenModel = readStringFlag(argv, index, arg);
        index += 1;
        break;
      case "--report-dir":
        options.reportDir = readStringFlag(argv, index, arg);
        index += 1;
        break;
      case "--request-timeout-ms":
        options.requestTimeoutMs = readIntegerFlag(
          argv,
          index,
          arg,
          (value) => value > 0,
          "a positive integer",
        );
        index += 1;
        break;
      case "--port":
        options.port = readIntegerFlag(
          argv,
          index,
          arg,
          (value) => value >= 0,
          "a non-negative integer",
        );
        index += 1;
        break;
      case "--allow-download":
        options.allowDownload = true;
        break;
      default:
        throw new QwenImageRegressionUsageError(
          arg.startsWith("-") ? `unknown option "${arg}".` : `unexpected argument "${arg}".`,
        );
    }
  }
  return { kind: "run", options };
}

function uint16le(value: number): number[] {
  return [value & 0xff, (value >> 8) & 0xff];
}

function int32le(value: number): number[] {
  return [value & 0xff, (value >> 8) & 0xff, (value >> 16) & 0xff, (value >> 24) & 0xff];
}

function uint32le(value: number): number[] {
  return int32le(value >>> 0);
}

function bmpBytes(width: number, height: number, pixels: readonly number[]): Uint8Array {
  const bytesPerPixel = 3;
  const rowStride = Math.ceil((width * bytesPerPixel) / 4) * 4;
  const pixelArraySize = rowStride * height;
  const fileSize = 54 + pixelArraySize;
  const header = [
    0x42,
    0x4d,
    ...uint32le(fileSize),
    0,
    0,
    0,
    0,
    ...uint32le(54),
    ...uint32le(40),
    ...int32le(width),
    ...int32le(-height),
    ...uint16le(1),
    ...uint16le(24),
    ...uint32le(0),
    ...uint32le(pixelArraySize),
    ...uint32le(0),
    ...uint32le(0),
    ...uint32le(0),
    ...uint32le(0),
  ];

  const pixelBytes: number[] = [];
  for (let row = 0; row < height; row += 1) {
    for (let column = 0; column < width; column += 1) {
      const index = (row * width + column) * 3;
      const red = pixels[index];
      const green = pixels[index + 1];
      const blue = pixels[index + 2];
      if (red === undefined || green === undefined || blue === undefined) {
        throw new Error("bmpBytes: missing RGB pixel data.");
      }
      pixelBytes.push(blue, green, red);
    }
    while (pixelBytes.length % rowStride !== 0) {
      pixelBytes.push(0);
    }
  }

  return new Uint8Array([...header, ...pixelBytes]);
}

function quadrantColor(row: number, column: number, width: number, height: number): number[] {
  const midX = Math.floor(width / 2);
  const midY = Math.floor(height / 2);
  const dividerRadius = Math.max(1, Math.floor(Math.min(width, height) / 48));
  const onDivider =
    Math.abs(column - midX) <= dividerRadius || Math.abs(row - midY) <= dividerRadius;
  if (onDivider) {
    return [0, 0, 0];
  }
  if (row < midY && column < midX) {
    return [255, 0, 0];
  }
  if (row < midY) {
    return [0, 255, 0];
  }
  return column < midX ? [0, 0, 255] : [255, 255, 0];
}

function quadrantPixels(width: number, height: number): number[] {
  const pixels: number[] = [];
  for (let row = 0; row < height; row += 1) {
    for (let column = 0; column < width; column += 1) {
      pixels.push(...quadrantColor(row, column, width, height));
    }
  }
  return pixels;
}

function base64FromBytes(bytes: Uint8Array): string {
  let binary = "";
  for (const byte of bytes) {
    binary += String.fromCharCode(byte);
  }
  return btoa(binary);
}

export function createQwenImageRegressionData(): {
  base64: string;
  dataUrl: string;
  width: number;
  height: number;
} {
  const base64 = base64FromBytes(
    bmpBytes(IMAGE_WIDTH, IMAGE_HEIGHT, quadrantPixels(IMAGE_WIDTH, IMAGE_HEIGHT)),
  );
  return {
    base64,
    dataUrl: `data:image/bmp;base64,${base64}`,
    width: IMAGE_WIDTH,
    height: IMAGE_HEIGHT,
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function numberValue(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function stringValue(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function usageFromOpenAICompletionLike(body: unknown): UsageReport {
  const usage = isRecord(body) && isRecord(body.usage) ? body.usage : {};
  const details = isRecord(usage.prompt_tokens_details) ? usage.prompt_tokens_details : {};
  return {
    promptTokens: numberValue(usage.prompt_tokens),
    completionTokens: numberValue(usage.completion_tokens),
    totalTokens: numberValue(usage.total_tokens),
    cacheReadTokens: numberValue(details.cached_tokens) ?? 0,
    cacheWriteTokens: numberValue(details.cache_write_tokens) ?? 0,
  };
}

function usageFromResponses(body: unknown): UsageReport {
  const usage = isRecord(body) && isRecord(body.usage) ? body.usage : {};
  const details = isRecord(usage.input_tokens_details) ? usage.input_tokens_details : {};
  return {
    promptTokens: numberValue(usage.input_tokens),
    completionTokens: numberValue(usage.output_tokens),
    totalTokens: numberValue(usage.total_tokens),
    cacheReadTokens: numberValue(details.cached_tokens) ?? 0,
    cacheWriteTokens: 0,
  };
}

function usageFromAnthropic(body: unknown): UsageReport {
  const usage = isRecord(body) && isRecord(body.usage) ? body.usage : {};
  return {
    promptTokens: numberValue(usage.input_tokens),
    completionTokens: numberValue(usage.output_tokens),
    totalTokens: null,
    cacheReadTokens: 0,
    cacheWriteTokens: 0,
  };
}

function chatOutput(body: unknown) {
  if (!isRecord(body) || !Array.isArray(body.choices) || !isRecord(body.choices[0])) {
    throw new Error("OpenAI chat response is missing choices.");
  }
  const choice = body.choices[0];
  if (!isRecord(choice.message)) {
    throw new Error("OpenAI chat response is missing message.");
  }
  return {
    outputText: stringValue(choice.message.content) ?? "",
    finishReason: stringValue(choice.finish_reason),
    usage: usageFromOpenAICompletionLike(body),
  };
}

function responsesOutput(body: unknown) {
  if (!isRecord(body)) {
    throw new Error("OpenAI responses body must be an object.");
  }
  const status = stringValue(body.status);
  return {
    outputText: stringValue(body.output_text) ?? "",
    finishReason:
      status === "incomplete" && isRecord(body.incomplete_details)
        ? stringValue(body.incomplete_details.reason)
        : status,
    usage: usageFromResponses(body),
  };
}

function anthropicOutput(body: unknown) {
  if (!isRecord(body) || !Array.isArray(body.content)) {
    throw new Error("Anthropic response is missing content.");
  }
  const text = body.content
    .map((part) => (isRecord(part) && part.type === "text" ? (stringValue(part.text) ?? "") : ""))
    .join("");
  return {
    outputText: text,
    finishReason: stringValue(body.stop_reason),
    usage: usageFromAnthropic(body),
  };
}

function chatBody(model: string, dataUrl: string): unknown {
  return {
    model,
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: PROMPT },
          { type: "image_url", image_url: { url: dataUrl } },
        ],
      },
    ],
    max_tokens: MAX_OUTPUT_TOKENS,
    temperature: 0,
    chat_template_kwargs: { enable_thinking: false },
  };
}

function responsesBody(model: string, dataUrl: string): unknown {
  return {
    model,
    input: [
      {
        role: "user",
        content: [
          { type: "input_text", text: PROMPT },
          { type: "input_image", image_url: dataUrl },
        ],
      },
    ],
    max_output_tokens: MAX_OUTPUT_TOKENS,
    temperature: 0,
    chat_template_kwargs: { enable_thinking: false },
  };
}

function anthropicBody(model: string, base64: string): unknown {
  return {
    model,
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: PROMPT },
          { type: "image", source: { type: "base64", media_type: "image/bmp", data: base64 } },
        ],
      },
    ],
    max_tokens: MAX_OUTPUT_TOKENS,
    temperature: 0,
    thinking: { type: "disabled" },
  };
}

function protocolProbes(model: string, image: ReturnType<typeof createQwenImageRegressionData>) {
  return [
    {
      label: "openai-chat-cold",
      protocol: "openai.chat_completions",
      path: "/v1/chat/completions",
      body: chatBody(model, image.dataUrl),
      output: chatOutput,
    },
    {
      label: "openai-chat-repeat",
      protocol: "openai.chat_completions",
      path: "/v1/chat/completions",
      body: chatBody(model, image.dataUrl),
      output: chatOutput,
    },
    {
      label: "openai-responses-cold",
      protocol: "openai.responses",
      path: "/v1/responses",
      body: responsesBody(model, image.dataUrl),
      output: responsesOutput,
    },
    {
      label: "openai-responses-repeat",
      protocol: "openai.responses",
      path: "/v1/responses",
      body: responsesBody(model, image.dataUrl),
      output: responsesOutput,
    },
    {
      label: "anthropic-messages-cold",
      protocol: "anthropic.messages",
      path: "/v1/messages",
      body: anthropicBody(model, image.base64),
      output: anthropicOutput,
    },
    {
      label: "anthropic-messages-repeat",
      protocol: "anthropic.messages",
      path: "/v1/messages",
      body: anthropicBody(model, image.base64),
      output: anthropicOutput,
    },
  ] satisfies ProtocolProbe[];
}

function isEventType<T extends ServeEvent["type"]>(
  event: RecordedServeEvent,
  type: T,
): event is Extract<ServeEvent, { type: T }> & { observedAtMs: number } {
  return event.type === type;
}

function promptCacheReport(events: readonly RecordedServeEvent[]): PromptCacheReport {
  const cacheEvents = events.filter((event) => isEventType(event, "generation_prompt_cache"));
  return {
    hits: cacheEvents.filter((event) => event.result === "hit").length,
    misses: cacheEvents.filter((event) => event.result === "miss").length,
    writes: cacheEvents.filter((event) => event.result === "write").length,
    readTokens: Math.max(0, ...cacheEvents.map((event) => event.cacheReadTokens)),
    writeTokens: Math.max(0, ...cacheEvents.map((event) => event.cacheWriteTokens)),
  };
}

function routeDecisionReports(events: readonly RecordedServeEvent[]): RouteDecisionReport[] {
  const starts = new Map<string, Extract<ServeEvent, { type: "generation_start" }>>();
  for (const event of events) {
    if (isEventType(event, "generation_start")) {
      starts.set(event.id, event);
    }
  }
  return events
    .filter((event) => isEventType(event, "generation_route_decision"))
    .map((event) => {
      const start = starts.get(event.id);
      return {
        id: event.id,
        protocol: event.protocol,
        route: event.route,
        reason: event.reason,
        ...(start === undefined ? {} : { inputKind: start.inputKind }),
      };
    });
}

function promptPrepareMs(events: readonly RecordedServeEvent[]): {
  promptPrepareEvents: number;
  promptPrepareMs: number | null;
} {
  const complete = events.filter(
    (event) => isEventType(event, "generation_prompt_prepare") && event.phase === "complete",
  );
  return {
    promptPrepareEvents: complete.length,
    promptPrepareMs:
      complete.length === 0
        ? null
        : complete.reduce((total, event) => total + event.durationMs, 0) / complete.length,
  };
}

function memoryPeakBytes(events: readonly RecordedServeEvent[]): number | null {
  const peaks = events
    .flatMap((event) =>
      "memory" in event && event.memory !== undefined ? [event.memory.peakBytes] : [],
    )
    .filter((value) => Number.isFinite(value));
  return peaks.length === 0 ? null : Math.max(...peaks);
}

async function fetchJsonProbe(
  endpoint: string,
  probe: ProtocolProbe,
  options: CliOptions,
  events: readonly RecordedServeEvent[],
): Promise<QwenImageProbeReport> {
  const eventStart = events.length;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), options.requestTimeoutMs);
  const started = performance.now();
  try {
    const response = await fetch(`${endpoint}${probe.path}`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${API_KEY}`,
      },
      body: JSON.stringify(probe.body),
      signal: controller.signal,
    });
    const responseText = await response.text();
    const durationMs = performance.now() - started;
    let body: unknown;
    try {
      body = JSON.parse(responseText);
    } catch (_error) {
      throw new Error(`${probe.label}: response was not JSON: ${responseText.slice(0, 300)}`);
    }
    if (response.status !== 200) {
      throw new Error(`${probe.label}: HTTP ${response.status}: ${responseText.slice(0, 500)}`);
    }
    const output = probe.output(body);
    const probeEvents = events.slice(eventStart);
    const cache = promptCacheReport(probeEvents);
    const prepare = promptPrepareMs(probeEvents);
    return {
      label: probe.label,
      protocol: probe.protocol,
      path: probe.path,
      status: response.status,
      durationMs,
      outputText: output.outputText,
      finishReason: output.finishReason,
      promptTokens: output.usage.promptTokens,
      completionTokens: output.usage.completionTokens,
      totalTokens: output.usage.totalTokens,
      cacheReadTokens: Math.max(output.usage.cacheReadTokens, cache.readTokens),
      cacheWriteTokens: Math.max(output.usage.cacheWriteTokens, cache.writeTokens),
      ...prepare,
      promptCache: cache,
      routeDecisions: routeDecisionReports(probeEvents),
      continuousSchedulerPhases: probeEvents.filter((event) =>
        isEventType(event, "generation_scheduler_phase"),
      ).length,
      memoryPeakBytes: memoryPeakBytes(probeEvents),
    };
  } finally {
    clearTimeout(timeout);
  }
}

function assertProbeHealthy(probe: QwenImageProbeReport): string[] {
  const failures: string[] = [];
  if (probe.status !== 200) {
    failures.push(`${probe.label}: status ${probe.status} !== 200`);
  }
  if (probe.outputText.trim() === "") {
    failures.push(`${probe.label}: visible output is empty`);
  }
  if (probe.promptPrepareEvents < 1) {
    failures.push(`${probe.label}: prompt preparation event missing`);
  }
  if (probe.continuousSchedulerPhases !== 0) {
    failures.push(`${probe.label}: continuous scheduler touched a media request`);
  }
  if (probe.routeDecisions.length === 0) {
    failures.push(`${probe.label}: route decision missing`);
  }
  for (const decision of probe.routeDecisions) {
    if (decision.route !== "single" || decision.reason !== "media_input") {
      failures.push(
        `${probe.label}: expected single:media_input, got ${decision.route}:${decision.reason}`,
      );
    }
    if (decision.inputKind !== undefined && decision.inputKind !== "content") {
      failures.push(`${probe.label}: expected content input, got ${decision.inputKind}`);
    }
  }
  if (probe.label.endsWith("-repeat") && probe.cacheReadTokens <= 0) {
    failures.push(`${probe.label}: repeat image request did not read from prompt cache`);
  }
  return failures;
}

export function assertQwenImageRegressionReport(report: QwenImageRegressionReport): void {
  const labels = new Set(report.probes.map((probe) => probe.label));
  const failures: string[] = [];
  for (const label of PROBE_LABELS) {
    if (!labels.has(label)) {
      failures.push(`missing probe: ${label}`);
    }
  }
  for (const probe of report.probes) {
    failures.push(...assertProbeHealthy(probe));
  }
  if (failures.length > 0) {
    throw new Error(
      `Qwen image regression failed:\n${failures.map((item) => `- ${item}`).join("\n")}`,
    );
  }
}

function formatMultilineField(name: string, value: string): string[] {
  const lines = value.split(/\r?\n/);
  if (lines.length === 1) {
    return [`  ${name}: ${toon(value)}`];
  }
  return [`  ${name}: |`, ...lines.map((line) => `    ${line}`)];
}

function toon(value: string | number | boolean | null): string {
  return typeof value === "string" ? JSON.stringify(value) : String(value);
}

function routeSummary(probe: QwenImageProbeReport): string {
  const routes = probe.routeDecisions.map((decision) => `${decision.route}:${decision.reason}`);
  return routes.length === 0 ? "none" : routes.join("|");
}

function probeDurationMs(probe: QwenImageProbeReport): number {
  return Math.round(probe.durationMs);
}

export function formatQwenImageRegressionSuccess(
  reportPath: string,
  report: QwenImageRegressionReport,
): string {
  const repeatProbes = report.probes.filter((probe) => probe.label.endsWith("-repeat"));
  const repeatCacheHits = repeatProbes.filter((probe) => probe.cacheReadTokens > 0).length;
  const cacheReadTokens = report.probes.reduce((total, probe) => total + probe.cacheReadTokens, 0);
  const cacheWriteTokens = report.probes.reduce(
    (total, probe) => total + probe.cacheWriteTokens,
    0,
  );
  const memoryPeaks = report.probes.flatMap((probe) =>
    probe.memoryPeakBytes === null ? [] : [probe.memoryPeakBytes],
  );
  const maxMemoryPeakBytes = memoryPeaks.length === 0 ? "null" : String(Math.max(...memoryPeaks));
  return [
    "qwen_image_regression:",
    "  status: passed",
    `  report: ${toon(reportPath)}`,
    `  model: ${toon(report.model)}`,
    `  probes: ${report.probes.length}`,
    `  repeat_cache_hits: ${repeatCacheHits}/${repeatProbes.length}`,
    `  cache_read_tokens: ${cacheReadTokens}`,
    `  cache_write_tokens: ${cacheWriteTokens}`,
    `  max_memory_peak_bytes: ${maxMemoryPeakBytes}`,
    `probes[${report.probes.length}]{label,protocol,route,cache_read_tokens,cache_write_tokens,output_chars,duration_ms}:`,
    ...report.probes.map((probe) =>
      [
        `  ${toon(probe.label)}`,
        toon(probe.protocol),
        toon(routeSummary(probe)),
        probe.cacheReadTokens,
        probe.cacheWriteTokens,
        probe.outputText.length,
        probeDurationMs(probe),
      ].join(","),
    ),
  ].join("\n");
}

export function formatQwenImageRegressionError(message: string, help: string): string {
  return ["error:", ...formatMultilineField("message", message), `help: ${toon(help)}`].join("\n");
}

export async function writeQwenImageRegressionReport(
  reportDir: string,
  report: QwenImageRegressionReport,
): Promise<string> {
  mkdirSync(reportDir, { recursive: true });
  const path = join(reportDir, "qwen-image-regression.json");
  await Bun.write(path, `${JSON.stringify(report, null, 2)}\n`);
  return path;
}

export async function runQwenImageRegression(
  options: CliOptions,
  progress: (text: string) => void = console.error,
): Promise<QwenImageRegressionReport> {
  const image = createQwenImageRegressionData();
  const events: RecordedServeEvent[] = [];
  progress(
    [
      `qwen_model=${options.qwenModel}`,
      `local_files_only=${String(!options.allowDownload)}`,
      `request_timeout_ms=${options.requestTimeoutMs}`,
    ].join(" "),
  );
  const server = await serveModel({
    source: options.qwenModel,
    modelId: options.qwenModel,
    port: options.port,
    localFilesOnly: !options.allowDownload,
    maxGeneratedTokens: MAX_OUTPUT_TOKENS,
    maxPromptTokens: 262_144,
    maxTotalTokens: 262_144,
    maxBatchSize: 2,
    maxConcurrentRequests: 1,
    promptPrefixCacheMaxEntries: 4,
    gpuMemoryUtilization: 0.85,
    apiKey: API_KEY,
    onEvent(event) {
      events.push({ ...event, observedAtMs: performance.now() });
    },
  });

  try {
    progress(`endpoint=${server.endpoint}`);
    const probes: QwenImageProbeReport[] = [];
    for (const probe of protocolProbes(options.qwenModel, image)) {
      const result = await fetchJsonProbe(server.endpoint, probe, options, events);
      probes.push(result);
      progress(
        [
          `probe=${result.label}`,
          `duration_ms=${result.durationMs.toFixed(1)}`,
          `route=${result.routeDecisions.map((route) => `${route.route}:${route.reason}`).join("|")}`,
          `cache_read_tokens=${result.cacheReadTokens}`,
          `cache_write_tokens=${result.cacheWriteTokens}`,
          `output_chars=${result.outputText.length}`,
        ].join(" "),
      );
    }
    return {
      createdAt: new Date().toISOString(),
      model: options.qwenModel,
      reportVersion: 1,
      image: { mediaType: "image/bmp", width: image.width, height: image.height },
      probes,
    };
  } finally {
    server.stop(true);
  }
}

export async function runQwenImageRegressionCommand(
  argv: readonly string[],
  runtime: QwenImageRegressionRuntime = {},
): Promise<number> {
  const stdout = runtime.stdout ?? console.log;
  const stderr = runtime.stderr ?? console.error;
  const runRegression = runtime.runRegression ?? runQwenImageRegression;
  const writeReport = runtime.writeReport ?? writeQwenImageRegressionReport;
  let command: QwenImageRegressionCommand;

  try {
    command = parseQwenImageRegressionArgs(argv);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    stdout(
      formatQwenImageRegressionError(
        message,
        "bun run regression:qwen-image -- --qwen-model <model-id>",
      ),
    );
    return error instanceof QwenImageRegressionUsageError ? 2 : 1;
  }

  if (command.kind === "help") {
    stdout(formatQwenImageRegressionUsage());
    return 0;
  }

  try {
    using _runtimeLock =
      runtime.acquireLock?.() ?? acquireRuntimeCommandLock("regression:qwen-image");
    const report = await runRegression(command.options, stderr);
    assertQwenImageRegressionReport(report);
    const reportPath = await writeReport(command.options.reportDir, report);
    stdout(formatQwenImageRegressionSuccess(reportPath, report));
    return 0;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    stdout(
      formatQwenImageRegressionError(
        message,
        "review stderr and rerun `bun run regression:qwen-image` after fixing the serving failure",
      ),
    );
    return 1;
  }
}

if (import.meta.main) {
  const exitCode = await runQwenImageRegressionCommand(Bun.argv.slice(2));
  process.exit(exitCode);
}
