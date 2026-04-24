export type MatrixMode = "cartesian" | "zip";
export type SamplingMode = "model-defaults" | "greedy";
export type TransportMode = "non-streaming" | "streaming";
export type ProtocolMode = "completions" | "chat" | "responses";

export type ServeBenchmarkOptions = {
  model: string;
  modelId: string;
  promptTokens: number[];
  generationTokens: number[];
  concurrency: number[];
  requestStaggerMs: number;
  rungs?: ServeBenchmarkRung[];
  reportJson?: string;
  trials: number;
  warmup: boolean;
  matrix: MatrixMode;
  samplingMode: SamplingMode;
  transportMode: TransportMode;
  protocolMode: ProtocolMode;
  ignoreEos: boolean;
  localFilesOnly: boolean;
  port: number;
  maxBatchSize: number;
  batchWindowMs: number;
  maxConcurrentRequests: number;
  requestTimeoutMs: number;
  gpuMemoryUtilization: number;
  maxPromptTokens?: number;
  maxTotalTokens?: number;
};

export type ServeBenchmarkRung = {
  promptTokens: number;
  generationTokens: number;
  concurrency: number;
};

export function requestLaunchDelayMs(requestIndex: number, requestStaggerMs: number): number {
  return Math.max(0, requestIndex) * requestStaggerMs;
}

type PromptOutputPair = Pick<ServeBenchmarkRung, "promptTokens" | "generationTokens">;

const DEFAULT_PROMPT_TOKENS = [128];
const DEFAULT_GENERATION_TOKENS = [128];
const DEFAULT_CONCURRENCY = [1];

function usage(): never {
  console.error(
    [
      "Usage: bun run packages/serve/scripts/benchmark-serve.ts --model <repo-or-path> [options]",
      "",
      "Options:",
      "  --prompt-tokens <list>          Comma-separated prompt token targets, default 128",
      "  --generation-tokens <list>      Comma-separated max_tokens targets, default 128",
      "  --concurrency <list>            Comma-separated parallel request counts, default 1",
      "  --rungs <spec>                  Explicit rungs like 128x128@1,1024x512@2",
      "  --request-stagger-ms <n>        Delay each request launch by index*n ms, default 0",
      "  --matrix <cartesian|zip>        Pair prompt/output rungs, default cartesian",
      "  --trials <n>                    Trials per rung, default 1",
      "  --report-json <path>            Write a structured JSON report at the end",
      "  --protocol <name>               completions, chat, or responses; default completions",
      "  --stream                        Measure SSE streaming and time-to-first-token",
      "  --ignore-eos                    Request exact max_tokens for throughput ladders",
      "  --greedy                        Send temperature=0 for deterministic throughput",
      "  --no-warmup                     Skip the one-request warmup for each prompt/output pair",
      "  --max-concurrent-requests <n>   Server-side in-flight generation limit, default 1",
      "  --max-batch-size <n>            Admission micro-batch size, default 32",
      "  --batch-window-ms <n>           Admission micro-batch window, default 1",
      "  --gpu-memory-utilization <f>    Serving memory preflight budget, default 0.9",
      "  --request-timeout-ms <n>        Client fetch timeout per request, default 3600000",
      "  --max-prompt-tokens <n>         Override server prompt admission limit",
      "  --max-total-tokens <n>          Override server total-token admission limit",
      "  --allow-download                Allow Hub downloads; default is cached/local only",
    ].join("\n"),
  );
  process.exit(1);
}

function readRequiredValue(flag: string, value: string | undefined): string {
  if (value === undefined || value.startsWith("--")) {
    throw new Error(`benchmark-serve: missing value for ${flag}.`);
  }
  return value;
}

function readPositiveInteger(flag: string, value: string | undefined): number {
  const parsed = Number.parseInt(readRequiredValue(flag, value), 10);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`benchmark-serve: ${flag} expects a positive integer.`);
  }
  return parsed;
}

function readNonNegativeInteger(flag: string, value: string | undefined): number {
  const parsed = Number.parseInt(readRequiredValue(flag, value), 10);
  if (!Number.isInteger(parsed) || parsed < 0) {
    throw new Error(`benchmark-serve: ${flag} expects a non-negative integer.`);
  }
  return parsed;
}

function readPositiveFraction(flag: string, value: string | undefined): number {
  const parsed = Number.parseFloat(readRequiredValue(flag, value));
  if (!Number.isFinite(parsed) || parsed <= 0 || parsed > 1) {
    throw new Error(`benchmark-serve: ${flag} expects 0 < value <= 1.`);
  }
  return parsed;
}

export function parsePositiveIntegerList(flag: string, value: string | undefined): number[] {
  const requiredValue = readRequiredValue(flag, value);
  if (requiredValue.trim() === "") {
    throw new Error(`benchmark-serve: ${flag} expects a comma-separated integer list.`);
  }
  const parsed = requiredValue.split(",").map((entry) => readPositiveInteger(flag, entry.trim()));
  return [...new Set(parsed)];
}

function parseServeBenchmarkRung(flag: string, value: string): ServeBenchmarkRung {
  const match = /^(\d+)x(\d+)(?:@(\d+))?$/.exec(value);
  if (match === null) {
    throw new Error(`benchmark-serve: ${flag} entries must look like 128x128@1.`);
  }
  const [, promptTokensText, generationTokensText, concurrencyText] = match;
  return {
    promptTokens: readPositiveInteger(flag, promptTokensText),
    generationTokens: readPositiveInteger(flag, generationTokensText),
    concurrency: concurrencyText === undefined ? 1 : readPositiveInteger(flag, concurrencyText),
  };
}

export function parseServeBenchmarkRungs(
  flag: string,
  value: string | undefined,
): ServeBenchmarkRung[] {
  const requiredValue = readRequiredValue(flag, value);
  if (requiredValue.trim() === "") {
    throw new Error(`benchmark-serve: ${flag} expects comma-separated rungs.`);
  }
  return requiredValue.split(",").map((entry) => parseServeBenchmarkRung(flag, entry.trim()));
}

function parseMatrixMode(value: string | undefined): MatrixMode {
  const requiredValue = readRequiredValue("--matrix", value);
  if (requiredValue === "cartesian" || requiredValue === "zip") {
    return requiredValue;
  }
  throw new Error('benchmark-serve: --matrix must be "cartesian" or "zip".');
}

function defaultModelId(model: string | undefined): string {
  if (model === undefined || model.trim() === "") {
    return "local-model";
  }
  const pieces = model.split("/").filter((piece) => piece !== "");
  return pieces.at(-1) ?? "local-model";
}

type ParseState = {
  model?: string;
  modelId?: string;
  promptTokens: number[];
  generationTokens: number[];
  concurrency: number[];
  requestStaggerMs: number;
  rungs?: ServeBenchmarkRung[];
  reportJson?: string;
  trials: number;
  warmup: boolean;
  matrix: MatrixMode;
  samplingMode: SamplingMode;
  transportMode: TransportMode;
  protocolMode: ProtocolMode;
  ignoreEos: boolean;
  localFilesOnly: boolean;
  port: number;
  maxBatchSize: number;
  batchWindowMs: number;
  maxConcurrentRequests: number;
  requestTimeoutMs: number;
  gpuMemoryUtilization: number;
  maxPromptTokens?: number;
  maxTotalTokens?: number;
};

function defaultParseState(): ParseState {
  return {
    promptTokens: DEFAULT_PROMPT_TOKENS,
    generationTokens: DEFAULT_GENERATION_TOKENS,
    concurrency: DEFAULT_CONCURRENCY,
    requestStaggerMs: 0,
    trials: 1,
    warmup: true,
    matrix: "cartesian",
    samplingMode: "model-defaults",
    transportMode: "non-streaming",
    protocolMode: "completions",
    ignoreEos: false,
    localFilesOnly: true,
    port: 0,
    maxBatchSize: 32,
    batchWindowMs: 1,
    maxConcurrentRequests: 1,
    requestTimeoutMs: 3_600_000,
    gpuMemoryUtilization: 0.9,
  };
}

function parseProtocolMode(value: string | undefined): ProtocolMode {
  const requiredValue = readRequiredValue("--protocol", value);
  if (
    requiredValue === "completions" ||
    requiredValue === "chat" ||
    requiredValue === "responses"
  ) {
    return requiredValue;
  }
  throw new Error('benchmark-serve: --protocol must be "completions", "chat", or "responses".');
}

function readBenchmarkValueArg(state: ParseState, arg: string, value: string | undefined): boolean {
  switch (arg) {
    case "--model":
      state.model = readRequiredValue(arg, value);
      return true;
    case "--model-id":
      state.modelId = readRequiredValue(arg, value);
      return true;
    case "--prompt-tokens":
      state.promptTokens = parsePositiveIntegerList(arg, value);
      return true;
    case "--generation-tokens":
      state.generationTokens = parsePositiveIntegerList(arg, value);
      return true;
    case "--concurrency":
      state.concurrency = parsePositiveIntegerList(arg, value);
      return true;
    case "--request-stagger-ms":
      state.requestStaggerMs = readNonNegativeInteger(arg, value);
      return true;
    case "--rungs":
      state.rungs = parseServeBenchmarkRungs(arg, value);
      return true;
    case "--trials":
      state.trials = readPositiveInteger(arg, value);
      return true;
    case "--report-json":
      state.reportJson = readRequiredValue(arg, value);
      return true;
    case "--matrix":
      state.matrix = parseMatrixMode(value);
      return true;
    case "--protocol":
      state.protocolMode = parseProtocolMode(value);
      return true;
    default:
      return false;
  }
}

function readServerValueArg(state: ParseState, arg: string, value: string | undefined): boolean {
  switch (arg) {
    case "--port":
      state.port = readNonNegativeInteger(arg, value);
      return true;
    case "--max-batch-size":
      state.maxBatchSize = readPositiveInteger(arg, value);
      return true;
    case "--batch-window-ms":
      state.batchWindowMs = readNonNegativeInteger(arg, value);
      return true;
    case "--max-concurrent-requests":
      state.maxConcurrentRequests = readPositiveInteger(arg, value);
      return true;
    case "--gpu-memory-utilization":
      state.gpuMemoryUtilization = readPositiveFraction(arg, value);
      return true;
    case "--request-timeout-ms":
      state.requestTimeoutMs = readPositiveInteger(arg, value);
      return true;
    case "--max-prompt-tokens":
      state.maxPromptTokens = readPositiveInteger(arg, value);
      return true;
    case "--max-total-tokens":
      state.maxTotalTokens = readPositiveInteger(arg, value);
      return true;
    default:
      return false;
  }
}

function readBooleanArg(state: ParseState, arg: string): boolean {
  if (arg === "--help" || arg === "-h") {
    usage();
  }
  switch (arg) {
    case "--greedy":
      state.samplingMode = "greedy";
      return true;
    case "--stream":
      state.transportMode = "streaming";
      return true;
    case "--ignore-eos":
      state.ignoreEos = true;
      return true;
    case "--no-warmup":
      state.warmup = false;
      return true;
    case "--allow-download":
      state.localFilesOnly = false;
      return true;
    default:
      return false;
  }
}

function readValueArg(state: ParseState, arg: string, value: string | undefined): boolean {
  return readBenchmarkValueArg(state, arg, value) || readServerValueArg(state, arg, value);
}

function readBenchmarkArg(state: ParseState, argv: readonly string[], index: number): number {
  const arg = argv[index];
  if (arg === undefined) {
    return index;
  }
  if (readValueArg(state, arg, argv[index + 1])) {
    return index + 1;
  }
  if (readBooleanArg(state, arg)) {
    return index;
  }
  if (!arg.startsWith("--") && state.model === undefined) {
    state.model = arg;
    return index;
  }
  throw new Error(`benchmark-serve: unknown argument "${arg}".`);
}

export function parseServeBenchmarkArgs(argv: readonly string[]): ServeBenchmarkOptions {
  const state = defaultParseState();

  for (let index = 0; index < argv.length; index += 1) {
    index = readBenchmarkArg(state, argv, index);
  }

  if (state.model === undefined || state.model.trim() === "") {
    usage();
  }

  if (state.protocolMode === "responses" && state.ignoreEos) {
    throw new Error("benchmark-serve: --ignore-eos is not supported with --protocol responses.");
  }

  return {
    model: state.model,
    modelId: state.modelId ?? defaultModelId(state.model),
    promptTokens: state.promptTokens,
    generationTokens: state.generationTokens,
    concurrency: state.concurrency,
    requestStaggerMs: state.requestStaggerMs,
    ...(state.rungs === undefined ? {} : { rungs: state.rungs }),
    ...(state.reportJson === undefined ? {} : { reportJson: state.reportJson }),
    trials: state.trials,
    warmup: state.warmup,
    matrix: state.matrix,
    samplingMode: state.samplingMode,
    transportMode: state.transportMode,
    protocolMode: state.protocolMode,
    ignoreEos: state.ignoreEos,
    localFilesOnly: state.localFilesOnly,
    port: state.port,
    maxBatchSize: state.maxBatchSize,
    batchWindowMs: state.batchWindowMs,
    maxConcurrentRequests: state.maxConcurrentRequests,
    requestTimeoutMs: state.requestTimeoutMs,
    gpuMemoryUtilization: state.gpuMemoryUtilization,
    ...(state.maxPromptTokens === undefined ? {} : { maxPromptTokens: state.maxPromptTokens }),
    ...(state.maxTotalTokens === undefined ? {} : { maxTotalTokens: state.maxTotalTokens }),
  };
}

function buildZippedPromptOutputPairs(options: ServeBenchmarkOptions): PromptOutputPair[] {
  if (options.promptTokens.length !== options.generationTokens.length) {
    throw new Error("benchmark-serve: zip matrix requires equal prompt and generation counts.");
  }

  const pairs: PromptOutputPair[] = [];
  for (let index = 0; index < options.promptTokens.length; index += 1) {
    const promptTokens = options.promptTokens[index];
    const generationTokens = options.generationTokens[index];
    if (promptTokens === undefined || generationTokens === undefined) {
      throw new Error("benchmark-serve: invalid zip matrix entry.");
    }
    pairs.push({ promptTokens, generationTokens });
  }
  return pairs;
}

function buildCartesianPromptOutputPairs(options: ServeBenchmarkOptions): PromptOutputPair[] {
  return options.promptTokens.flatMap((promptTokens) =>
    options.generationTokens.map((generationTokens) => ({ promptTokens, generationTokens })),
  );
}

function buildPromptOutputPairs(options: ServeBenchmarkOptions): PromptOutputPair[] {
  return options.matrix === "zip"
    ? buildZippedPromptOutputPairs(options)
    : buildCartesianPromptOutputPairs(options);
}

export function buildServeBenchmarkRungs(options: ServeBenchmarkOptions): ServeBenchmarkRung[] {
  if (options.rungs !== undefined) {
    return options.rungs;
  }
  return buildPromptOutputPairs(options).flatMap((pair) =>
    options.concurrency.map((concurrency) => ({ ...pair, concurrency })),
  );
}
