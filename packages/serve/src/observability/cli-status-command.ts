import type { ServeInfoModel, ServeInfoResponse } from "../http/route-info";

export type ServeStatusCliOptions = {
  baseUrl: string;
  apiKey?: string;
  timeoutMs: number;
  full: boolean;
};

export type ServeStatusCliParseResult =
  | { kind: "status"; options: ServeStatusCliOptions }
  | { kind: "help"; exitCode: number; message?: string };

export type ServeStatusFetch = (
  input: string | URL | Request,
  init?: RequestInit,
) => Promise<Response>;

export type ServeStatusCliRuntime = {
  fetch?: ServeStatusFetch;
  env?: Readonly<Record<string, string | undefined>>;
  log?: (message: string) => void;
  exit?: (code: number) => void;
};

const DEFAULT_BASE_URL = "http://127.0.0.1:8000";
const DEFAULT_TIMEOUT_MS = 5_000;

function readStringFlag(flag: string, value: string | undefined): string {
  if (value === undefined || value.trim() === "") {
    throw new Error(`Missing value for ${flag}.`);
  }
  return value;
}

function readPositiveIntegerFlag(flag: string, value: string | undefined): number {
  const raw = readStringFlag(flag, value);
  const parsed = Number(raw);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`Expected ${flag} to be a positive integer, got "${raw}".`);
  }
  return parsed;
}

function normalizeBaseUrl(value: string): string {
  let url: URL;
  try {
    url = new URL(value);
  } catch {
    throw new Error(`Expected --base-url to be a valid URL, got "${value}".`);
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new Error(`Expected --base-url to use http or https, got "${url.protocol}".`);
  }
  if (url.search !== "" || url.hash !== "") {
    throw new Error("--base-url must not include query parameters or a fragment.");
  }
  url.pathname = url.pathname.replace(/\/v1\/?$/, "");
  return url.toString().replace(/\/$/, "");
}

function envValue(
  env: Readonly<Record<string, string | undefined>>,
  key: string,
): string | undefined {
  const value = env[key];
  return value === undefined || value.trim() === "" ? undefined : value;
}

function parseStatusState(
  argv: readonly string[],
  env: Readonly<Record<string, string | undefined>>,
): ServeStatusCliOptions {
  let baseUrl = envValue(env, "OPENAI_BASE_URL") ?? DEFAULT_BASE_URL;
  let apiKey = envValue(env, "OPENAI_API_KEY");
  let timeoutMs = DEFAULT_TIMEOUT_MS;
  let full = false;

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    switch (arg) {
      case "--base-url":
        baseUrl = readStringFlag(arg, argv[index + 1]);
        index += 1;
        break;
      case "--api-key":
        apiKey = readStringFlag(arg, argv[index + 1]);
        index += 1;
        break;
      case "--timeout-ms":
        timeoutMs = readPositiveIntegerFlag(arg, argv[index + 1]);
        index += 1;
        break;
      case "--full":
        full = true;
        break;
      default:
        if (arg?.startsWith("--")) {
          throw new Error(`Unknown argument: ${arg}`);
        }
        throw new Error(`Unexpected positional argument: ${arg ?? "<missing>"}`);
    }
  }

  return {
    baseUrl: normalizeBaseUrl(baseUrl),
    ...(apiKey === undefined ? {} : { apiKey }),
    timeoutMs,
    full,
  };
}

/** Parse `mlxts-serve status` arguments. */
export function parseServeStatusArgs(
  argv: readonly string[],
  env: Readonly<Record<string, string | undefined>> = process.env,
): ServeStatusCliParseResult {
  if (argv.includes("--help")) {
    return { kind: "help", exitCode: 0 };
  }

  try {
    return { kind: "status", options: parseStatusState(argv, env) };
  } catch (error) {
    return {
      kind: "help",
      exitCode: 2,
      message: error instanceof Error ? error.message : String(error),
    };
  }
}

function toon(value: string | number | boolean | null): string {
  return typeof value === "string" ? JSON.stringify(value) : String(value);
}

function formatHelp(commands: readonly string[]): string[] {
  if (commands.length === 0) {
    return [];
  }
  return [`help[${commands.length}]:`, ...commands.map((command) => `  ${command}`)];
}

/** Format `mlxts-serve status --help` as compact agent-readable stdout. */
export function formatServeStatusUsage(): string {
  return [
    "description: Inspect a running @mlxts/serve endpoint without sending generation work",
    "usage[2]:",
    "  mlxts-serve status",
    "  mlxts-serve status --base-url <url> [--api-key <key>] [--timeout-ms <n>] [--full]",
    "options[5]{flag,description}:",
    '  "--base-url <url>","Endpoint base URL; OPENAI_BASE_URL is used when present"',
    '  "--api-key <key>","Authorization bearer token for protected /info endpoints"',
    '  "--timeout-ms <n>","Request timeout in milliseconds"',
    '  "--full","Include limits, endpoint list, and runtime strategy"',
    '  "--help","Show this help"',
  ].join("\n");
}

/** Format a status command error for agent-readable stdout. */
export function formatServeStatusError(message: string, code = "usage"): string {
  return [
    "error:",
    `  code: ${toon(code)}`,
    `  message: ${toon(message)}`,
    ...formatHelp(["Run `mlxts-serve status --base-url <url> [--api-key <key>]`"]),
  ].join("\n");
}

function modelRows(models: readonly ServeInfoModel[]): string[] {
  if (models.length === 0) {
    return ["models: 0 served models reported"];
  }
  return [
    `models[${models.length}]{id,context_window,max_prompt_tokens,max_total_tokens}:`,
    ...models
      .map((model) =>
        [
          toon(model.id),
          toon(model.context_window),
          toon(model.max_prompt_tokens),
          toon(model.max_total_tokens),
        ].join(","),
      )
      .map((line) => `  ${line}`),
  ];
}

function formatLimits(info: ServeInfoResponse): string[] {
  const limits = info.limits;
  return [
    "limits:",
    `  max_generated_tokens: ${toon(limits.max_generated_tokens)}`,
    `  max_prompt_tokens: ${toon(limits.max_prompt_tokens)}`,
    `  max_total_tokens: ${toon(limits.max_total_tokens)}`,
    `  max_client_batch_size: ${toon(limits.max_client_batch_size)}`,
    `  batch_window_ms: ${toon(limits.batch_window_ms)}`,
    `  prefill_step_size: ${toon(limits.prefill_step_size)}`,
    `  active_prefill_step_size: ${toon(limits.active_prefill_step_size)}`,
    `  active_decode_steps_per_prefill_chunk: ${toon(
      limits.active_decode_steps_per_prefill_chunk,
    )}`,
    `  stream_decode_interval: ${toon(limits.stream_decode_interval)}`,
    `  max_concurrent_requests: ${toon(limits.max_concurrent_requests)}`,
    `  prompt_prefix_cache_max_entries: ${toon(limits.prompt_prefix_cache_max_entries)}`,
    `  prompt_prefix_cache_max_bytes: ${toon(limits.prompt_prefix_cache_max_bytes)}`,
    `  gpu_memory_utilization: ${toon(limits.gpu_memory_utilization)}`,
  ];
}

function formatDefaultLimits(info: ServeInfoResponse): string[] {
  const limits = info.limits;
  return [
    "limits:",
    `  max_generated_tokens: ${toon(limits.max_generated_tokens)}`,
    `  max_prompt_tokens: ${toon(limits.max_prompt_tokens)}`,
    `  max_total_tokens: ${toon(limits.max_total_tokens)}`,
    `  gpu_memory_utilization: ${toon(limits.gpu_memory_utilization)}`,
  ];
}

function formatRuntimeStrategy(info: ServeInfoResponse): string[] {
  const strategy = info.runtime_strategy;
  return [
    "runtime_strategy:",
    `  scheduler: ${toon(strategy.scheduler.mode)}`,
    `  max_batch_size: ${toon(strategy.scheduler.max_batch_size)}`,
    `  cache: ${toon(`${strategy.cache.backend}/${strategy.cache.precision}`)}`,
    `  attention: ${toon(strategy.attention.backend)}`,
    `  decoding: ${toon(strategy.decoding.backend)}`,
    `  stream_decode_interval: ${toon(strategy.streaming.stream_decode_interval)}`,
    `  memory_policy: ${toon(strategy.memory.policy)}`,
    `  gpu_memory_utilization: ${toon(strategy.memory.gpu_memory_utilization)}`,
  ];
}

/** Format a running serve endpoint status as compact agent-readable stdout. */
export function formatServeStatus(info: ServeInfoResponse, baseUrl: string, full: boolean): string {
  const lines = [
    "serve_status:",
    `  base_url: ${toon(baseUrl)}`,
    `  status: ${toon(info.status)}`,
    `  router: ${toon(info.router)}`,
    `  primary_model: ${toon(info.model_id)}`,
    `  model_count: ${toon(info.model_count)}`,
    `  streaming: ${toon(info.capabilities.sse_streaming)}`,
    `  batch_generation: ${toon(info.capabilities.batch_generation)}`,
    ...modelRows(info.models),
  ];
  if (full) {
    lines.push(
      ...formatLimits(info),
      ...formatRuntimeStrategy(info),
      `endpoints[${info.endpoints.length}]:`,
      ...info.endpoints.map((endpointPath) => `  ${toon(endpointPath)}`),
    );
  } else {
    lines.push(
      ...formatDefaultLimits(info),
      ...formatHelp([
        `Run \`mlxts-serve status --base-url ${toon(baseUrl)} --full\` for limits and runtime strategy`,
      ]),
    );
  }
  return lines.join("\n");
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isNumberOrNull(value: unknown): value is number | null {
  return typeof value === "number" || value === null;
}

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((item) => typeof item === "string");
}

function isServeInfoModel(value: unknown): value is ServeInfoModel {
  if (!isRecord(value)) {
    return false;
  }
  return (
    typeof value.id === "string" &&
    isNumberOrNull(value.context_window) &&
    isNumberOrNull(value.max_prompt_tokens) &&
    isNumberOrNull(value.max_total_tokens) &&
    isNumberOrNull(value.effective_total_tokens)
  );
}

function hasServeLimits(value: unknown): value is ServeInfoResponse["limits"] {
  if (!isRecord(value)) {
    return false;
  }
  return (
    isNumberOrNull(value.max_generated_tokens) &&
    isNumberOrNull(value.max_prompt_tokens) &&
    isNumberOrNull(value.max_total_tokens) &&
    isNumberOrNull(value.max_client_batch_size) &&
    isNumberOrNull(value.batch_window_ms) &&
    isNumberOrNull(value.prefill_step_size) &&
    isNumberOrNull(value.active_prefill_step_size) &&
    isNumberOrNull(value.active_decode_steps_per_prefill_chunk) &&
    isNumberOrNull(value.stream_decode_interval) &&
    isNumberOrNull(value.max_concurrent_requests) &&
    isNumberOrNull(value.prompt_prefix_cache_max_entries) &&
    isNumberOrNull(value.prompt_prefix_cache_max_bytes) &&
    isNumberOrNull(value.gpu_memory_utilization)
  );
}

function hasServeCapabilities(value: unknown): value is ServeInfoResponse["capabilities"] {
  if (!isRecord(value)) {
    return false;
  }
  return (
    value.completions === true &&
    value.chat_completions === "text_and_image_when_supported" &&
    value.responses === "text_and_image_when_supported" &&
    value.anthropic_messages === "text_and_image_when_supported" &&
    typeof value.sse_streaming === "boolean" &&
    typeof value.batch_generation === "boolean" &&
    value.reasoning_content === true &&
    value.tool_calls === true
  );
}

function hasRuntimeStrategy(value: unknown): value is ServeInfoResponse["runtime_strategy"] {
  if (!isRecord(value)) {
    return false;
  }
  const scheduler = value.scheduler;
  const cache = value.cache;
  const attention = value.attention;
  const decoding = value.decoding;
  const streaming = value.streaming;
  const memory = value.memory;
  return (
    isRecord(scheduler) &&
    scheduler.mode === "auto" &&
    typeof scheduler.max_batch_size === "number" &&
    typeof scheduler.batch_window_ms === "number" &&
    typeof scheduler.prefill_step_size === "number" &&
    typeof scheduler.active_prefill_step_size === "number" &&
    typeof scheduler.active_decode_steps_per_prefill_chunk === "number" &&
    typeof scheduler.max_concurrent_requests === "number" &&
    isRecord(cache) &&
    cache.backend === "managed" &&
    cache.precision === "model" &&
    typeof cache.prompt_prefix_max_entries === "number" &&
    isNumberOrNull(cache.prompt_prefix_max_bytes) &&
    isRecord(attention) &&
    attention.backend === "auto" &&
    isRecord(decoding) &&
    decoding.backend === "model" &&
    isRecord(streaming) &&
    typeof streaming.stream_decode_interval === "number" &&
    isRecord(memory) &&
    (memory.policy === "none" || memory.policy === "admit_only") &&
    isNumberOrNull(memory.gpu_memory_utilization)
  );
}

function isServeInfoResponse(value: unknown): value is ServeInfoResponse {
  if (!isRecord(value)) {
    return false;
  }
  return (
    value.status === "ok" &&
    value.router === "@mlxts/serve" &&
    value.version === null &&
    (typeof value.model_id === "string" || value.model_id === null) &&
    isStringArray(value.model_ids) &&
    typeof value.model_count === "number" &&
    Array.isArray(value.models) &&
    value.models.every(isServeInfoModel) &&
    isStringArray(value.endpoints) &&
    hasServeLimits(value.limits) &&
    hasServeCapabilities(value.capabilities) &&
    hasRuntimeStrategy(value.runtime_strategy)
  );
}

function endpointPath(endpoint: string, path: string): string {
  return `${endpoint}${path}`;
}

function authHeaders(options: ServeStatusCliOptions): Headers {
  const headers = new Headers();
  if (options.apiKey !== undefined) {
    headers.set("authorization", `Bearer ${options.apiKey}`);
  }
  return headers;
}

async function requestStatusPath(
  path: string,
  options: ServeStatusCliOptions,
  fetchImpl: ServeStatusFetch,
): Promise<Response> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), options.timeoutMs);
  try {
    return await fetchImpl(endpointPath(options.baseUrl, path), {
      headers: authHeaders(options),
      signal: controller.signal,
    });
  } catch (error) {
    if (controller.signal.aborted) {
      throw new Error(`Timed out after ${options.timeoutMs}ms while requesting ${path}.`);
    }
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Failed to reach ${options.baseUrl}${path}: ${message}`);
  } finally {
    clearTimeout(timeout);
  }
}

async function readErrorMessage(response: Response): Promise<string> {
  const text = await response.text();
  if (text.trim() === "") {
    return `Endpoint returned HTTP ${response.status}.`;
  }
  try {
    const parsed: unknown = JSON.parse(text);
    if (isRecord(parsed) && isRecord(parsed.error) && typeof parsed.error.message === "string") {
      return parsed.error.message;
    }
  } catch {
    return text;
  }
  return text;
}

/** Fetch and validate `/info` from a running serve endpoint. */
export async function fetchServeStatusInfo(
  options: ServeStatusCliOptions,
  fetchImpl: ServeStatusFetch = fetch,
): Promise<ServeInfoResponse> {
  const health = await requestStatusPath("/health", options, fetchImpl);
  if (!health.ok) {
    const message = await readErrorMessage(health);
    throw new Error(message);
  }

  const response = await requestStatusPath("/info", options, fetchImpl);
  if (!response.ok) {
    const message = await readErrorMessage(response);
    throw new Error(message);
  }
  let body: unknown;
  try {
    body = await response.json();
  } catch {
    throw new Error("Endpoint /info did not return JSON.");
  }
  if (!isServeInfoResponse(body)) {
    throw new Error("Endpoint /info did not return an @mlxts/serve status payload.");
  }
  return body;
}

/** Run the finite `mlxts-serve status` command. */
export async function runServeStatusCli(
  argv: readonly string[],
  runtime: ServeStatusCliRuntime = {},
): Promise<void> {
  const log = runtime.log ?? console.log;
  const exit = runtime.exit ?? process.exit;
  const parsed = parseServeStatusArgs(argv, runtime.env ?? process.env);
  if (parsed.kind === "help") {
    log(
      parsed.message === undefined
        ? formatServeStatusUsage()
        : formatServeStatusError(parsed.message),
    );
    exit(parsed.exitCode);
    return;
  }

  try {
    const info = await fetchServeStatusInfo(parsed.options, runtime.fetch ?? fetch);
    log(formatServeStatus(info, parsed.options.baseUrl, parsed.options.full));
    exit(0);
  } catch (error) {
    log(formatServeStatusError(error instanceof Error ? error.message : String(error), "status"));
    exit(1);
  }
}
