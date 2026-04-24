export type MatrixMode = "cartesian" | "zip";
export type SamplingMode = "model-defaults" | "greedy";
export type TransportMode = "non-streaming" | "streaming";

export type ServeBenchmarkOptions = {
  model: string;
  modelId: string;
  promptTokens: number[];
  generationTokens: number[];
  concurrency: number[];
  trials: number;
  warmup: boolean;
  matrix: MatrixMode;
  samplingMode: SamplingMode;
  transportMode: TransportMode;
  localFilesOnly: boolean;
  port: number;
  maxBatchSize: number;
  batchWindowMs: number;
  maxConcurrentRequests: number;
  gpuMemoryUtilization: number;
  maxPromptTokens?: number;
  maxTotalTokens?: number;
};

export type ServeBenchmarkRung = {
  promptTokens: number;
  generationTokens: number;
  concurrency: number;
};

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
      "  --matrix <cartesian|zip>        Pair prompt/output rungs, default cartesian",
      "  --trials <n>                    Trials per rung, default 1",
      "  --stream                        Measure SSE streaming and time-to-first-token",
      "  --greedy                        Send temperature=0 for deterministic throughput",
      "  --no-warmup                     Skip the one-request warmup for each prompt/output pair",
      "  --max-concurrent-requests <n>   Server-side in-flight generation limit, default 1",
      "  --max-batch-size <n>            Admission micro-batch size, default 32",
      "  --batch-window-ms <n>           Admission micro-batch window, default 1",
      "  --gpu-memory-utilization <f>    Serving memory preflight budget, default 0.9",
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

export function parseServeBenchmarkArgs(argv: readonly string[]): ServeBenchmarkOptions {
  let model: string | undefined;
  let modelId: string | undefined;
  let promptTokens = DEFAULT_PROMPT_TOKENS;
  let generationTokens = DEFAULT_GENERATION_TOKENS;
  let concurrency = DEFAULT_CONCURRENCY;
  let trials = 1;
  let warmup = true;
  let matrix: MatrixMode = "cartesian";
  let samplingMode: SamplingMode = "model-defaults";
  let transportMode: TransportMode = "non-streaming";
  let localFilesOnly = true;
  let port = 0;
  let maxBatchSize = 32;
  let batchWindowMs = 1;
  let maxConcurrentRequests = 1;
  let gpuMemoryUtilization = 0.9;
  let maxPromptTokens: number | undefined;
  let maxTotalTokens: number | undefined;

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === undefined) {
      continue;
    }

    switch (arg) {
      case "--model":
        model = readRequiredValue(arg, argv[index + 1]);
        index += 1;
        break;
      case "--model-id":
        modelId = readRequiredValue(arg, argv[index + 1]);
        index += 1;
        break;
      case "--prompt-tokens":
        promptTokens = parsePositiveIntegerList(arg, argv[index + 1]);
        index += 1;
        break;
      case "--generation-tokens":
        generationTokens = parsePositiveIntegerList(arg, argv[index + 1]);
        index += 1;
        break;
      case "--concurrency":
        concurrency = parsePositiveIntegerList(arg, argv[index + 1]);
        index += 1;
        break;
      case "--trials":
        trials = readPositiveInteger(arg, argv[index + 1]);
        index += 1;
        break;
      case "--matrix":
        matrix = parseMatrixMode(argv[index + 1]);
        index += 1;
        break;
      case "--port":
        port = readNonNegativeInteger(arg, argv[index + 1]);
        index += 1;
        break;
      case "--max-batch-size":
        maxBatchSize = readPositiveInteger(arg, argv[index + 1]);
        index += 1;
        break;
      case "--batch-window-ms":
        batchWindowMs = readNonNegativeInteger(arg, argv[index + 1]);
        index += 1;
        break;
      case "--max-concurrent-requests":
        maxConcurrentRequests = readPositiveInteger(arg, argv[index + 1]);
        index += 1;
        break;
      case "--gpu-memory-utilization":
        gpuMemoryUtilization = readPositiveFraction(arg, argv[index + 1]);
        index += 1;
        break;
      case "--max-prompt-tokens":
        maxPromptTokens = readPositiveInteger(arg, argv[index + 1]);
        index += 1;
        break;
      case "--max-total-tokens":
        maxTotalTokens = readPositiveInteger(arg, argv[index + 1]);
        index += 1;
        break;
      case "--greedy":
        samplingMode = "greedy";
        break;
      case "--stream":
        transportMode = "streaming";
        break;
      case "--no-warmup":
        warmup = false;
        break;
      case "--allow-download":
        localFilesOnly = false;
        break;
      case "--help":
      case "-h":
        usage();
        break;
      default:
        if (!arg.startsWith("--") && model === undefined) {
          model = arg;
          break;
        }
        throw new Error(`benchmark-serve: unknown argument "${arg}".`);
    }
  }

  if (model === undefined || model.trim() === "") {
    usage();
  }

  return {
    model,
    modelId: modelId ?? defaultModelId(model),
    promptTokens,
    generationTokens,
    concurrency,
    trials,
    warmup,
    matrix,
    samplingMode,
    transportMode,
    localFilesOnly,
    port,
    maxBatchSize,
    batchWindowMs,
    maxConcurrentRequests,
    gpuMemoryUtilization,
    ...(maxPromptTokens === undefined ? {} : { maxPromptTokens }),
    ...(maxTotalTokens === undefined ? {} : { maxTotalTokens }),
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
  return buildPromptOutputPairs(options).flatMap((pair) =>
    options.concurrency.map((concurrency) => ({ ...pair, concurrency })),
  );
}
