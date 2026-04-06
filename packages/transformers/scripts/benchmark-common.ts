import {
  createStream,
  getRecommendedWorkingSetBytes,
  setWiredLimitBytes,
  startMetalCapture,
  stopMetalCapture,
  synchronize,
  withDefaultStream,
} from "@mlxts/core";
import type { Tokenizer } from "@mlxts/tokenizers";

export const BASELINE_PATH = "benchmarks/baselines.json";
const TRACE_DIR = "benchmarks/traces";
const MLX_LM_DECODE_TOLERANCE_RATIO = 0.98;

export type BenchmarkMode = "synthetic" | "parity";

export type BenchmarkTarget = {
  name: string;
  model: string;
  promptTokens: number;
  generationTokens: number;
  prefillStepSize?: number;
  promptTps?: number;
  generationTps?: number;
  peakMemoryGb?: number;
  explicitEvalCountPerToken?: number;
  mlxLmReference?: {
    promptTps: number;
    generationTps: number;
    peakMemoryGb: number;
    capturedAt: string;
  };
};

export type MlxLmReference = NonNullable<BenchmarkTarget["mlxLmReference"]>;

export type BenchmarkSection = {
  targets: BenchmarkTarget[];
};

export type BenchmarkBaselines = {
  synthetic: BenchmarkSection;
  parity: BenchmarkSection;
};

export type BenchmarkOptions = {
  promptTokens: number;
  generationTokens: number;
  trials: number;
  prefillStepSize: number;
  metalTrace: boolean;
};

export type ParsedBenchmarkArgs = {
  model?: string;
  options: BenchmarkOptions;
  reference: ReferenceBenchmarkOptions;
};

export type ReferenceBenchmarkOptions = {
  captureMlxLmReference: boolean;
  enforceMlxLmDecodeBar: boolean;
  mlxLmPython?: string;
};

export type TrialMetrics = {
  promptTps: number;
  generationTps: number;
  peakMemoryGb: number;
  explicitEvalCountPerToken: number;
  totalTimeSeconds: number;
};

type MutableBenchmarkOptions = {
  model?: string;
  promptTokens: number;
  generationTokens: number;
  trials: number;
  prefillStepSize: number;
  metalTrace: boolean;
  captureMlxLmReference: boolean;
  enforceMlxLmDecodeBar: boolean;
  mlxLmPython?: string;
};

function defaultOptions(): MutableBenchmarkOptions {
  return {
    model: undefined,
    promptTokens: 1024,
    generationTokens: 128,
    trials: 3,
    prefillStepSize: 2048,
    metalTrace: false,
    captureMlxLmReference: true,
    enforceMlxLmDecodeBar: false,
    mlxLmPython: undefined,
  };
}

function emptyBaselines(): BenchmarkBaselines {
  return {
    synthetic: { targets: [] },
    parity: { targets: [] },
  };
}

function parseRequiredNumber(value: string | undefined, flag: string): number {
  const parsed = Number.parseInt(value ?? "", 10);
  if (!Number.isInteger(parsed)) {
    throw new Error(`benchmark-generation: ${flag} expects an integer value.`);
  }
  return parsed;
}

function validateOptions(options: BenchmarkOptions): void {
  if (!Number.isInteger(options.promptTokens) || options.promptTokens <= 1) {
    throw new Error("benchmark-generation: promptTokens must be an integer greater than 1.");
  }
  if (!Number.isInteger(options.generationTokens) || options.generationTokens <= 0) {
    throw new Error("benchmark-generation: generationTokens must be a positive integer.");
  }
  if (!Number.isInteger(options.trials) || options.trials <= 0) {
    throw new Error("benchmark-generation: trials must be a positive integer.");
  }
  if (!Number.isInteger(options.prefillStepSize) || options.prefillStepSize <= 0) {
    throw new Error("benchmark-generation: prefillStepSize must be a positive integer.");
  }
}

function mergedOfflineEnv(): Record<string, string> {
  return {
    HF_HUB_OFFLINE: "1",
    HF_DATASETS_OFFLINE: "1",
    TRANSFORMERS_OFFLINE: "1",
    HF_HUB_DISABLE_TELEMETRY: "1",
    PYTHONNOUSERSITE: "1",
  };
}

function parseTarget(entry: unknown, context: string): BenchmarkTarget {
  if (typeof entry !== "object" || entry === null) {
    throw new Error(`${context}: targets must be objects.`);
  }

  if (
    typeof entry.name !== "string" ||
    typeof entry.model !== "string" ||
    typeof entry.promptTokens !== "number" ||
    typeof entry.generationTokens !== "number"
  ) {
    throw new Error(
      `${context}: targets must include name, model, promptTokens, and generationTokens.`,
    );
  }

  return {
    name: entry.name,
    model: entry.model,
    promptTokens: entry.promptTokens,
    generationTokens: entry.generationTokens,
    ...(typeof entry.prefillStepSize === "number"
      ? { prefillStepSize: entry.prefillStepSize }
      : {}),
    ...(typeof entry.promptTps === "number" ? { promptTps: entry.promptTps } : {}),
    ...(typeof entry.generationTps === "number" ? { generationTps: entry.generationTps } : {}),
    ...(typeof entry.peakMemoryGb === "number" ? { peakMemoryGb: entry.peakMemoryGb } : {}),
    ...(typeof entry.explicitEvalCountPerToken === "number"
      ? { explicitEvalCountPerToken: entry.explicitEvalCountPerToken }
      : {}),
    ...(typeof entry.mlxLmReference === "object" &&
    entry.mlxLmReference !== null &&
    typeof entry.mlxLmReference.promptTps === "number" &&
    typeof entry.mlxLmReference.generationTps === "number" &&
    typeof entry.mlxLmReference.peakMemoryGb === "number" &&
    typeof entry.mlxLmReference.capturedAt === "string"
      ? {
          mlxLmReference: {
            promptTps: entry.mlxLmReference.promptTps,
            generationTps: entry.mlxLmReference.generationTps,
            peakMemoryGb: entry.mlxLmReference.peakMemoryGb,
            capturedAt: entry.mlxLmReference.capturedAt,
          },
        }
      : {}),
  };
}

function parseSection(section: unknown, context: string): BenchmarkSection {
  if (typeof section !== "object" || section === null || !("targets" in section)) {
    throw new Error(`${context}: expected an object with a "targets" array.`);
  }
  if (!Array.isArray(section.targets)) {
    throw new Error(`${context}: "targets" must be an array.`);
  }
  return {
    targets: section.targets.map((entry) => parseTarget(entry, context)),
  };
}

function applyIntegerFlag(
  mutable: MutableBenchmarkOptions,
  flag: string,
  value: string | undefined,
): boolean {
  const parsed = parseRequiredNumber(value, flag);
  switch (flag) {
    case "--prompt-tokens":
      mutable.promptTokens = parsed;
      return true;
    case "--generation-tokens":
      mutable.generationTokens = parsed;
      return true;
    case "--trials":
      mutable.trials = parsed;
      return true;
    case "--prefill-step-size":
      mutable.prefillStepSize = parsed;
      return true;
    default:
      return false;
  }
}

function applyBooleanFlag(mutable: MutableBenchmarkOptions, flag: string): boolean {
  switch (flag) {
    case "--metal-trace":
      mutable.metalTrace = true;
      return true;
    case "--capture-mlx-lm-reference":
      mutable.captureMlxLmReference = true;
      return true;
    case "--enforce-mlx-lm-decode-bar":
      mutable.enforceMlxLmDecodeBar = true;
      return true;
    default:
      return false;
  }
}

function applyStringFlag(
  mutable: MutableBenchmarkOptions,
  flag: string,
  value: string | undefined,
): boolean {
  switch (flag) {
    case "--model":
      mutable.model = value;
      return true;
    case "--mlx-lm-python":
      mutable.mlxLmPython = value;
      return true;
    default:
      return false;
  }
}

function handlePositionalModel(mutable: MutableBenchmarkOptions, arg: string): void {
  if (mutable.model === undefined) {
    mutable.model = arg;
    return;
  }

  throw new Error(`benchmark-generation: unexpected positional argument "${arg}".`);
}

export function parseBenchmarkArgs(argv: readonly string[]): ParsedBenchmarkArgs {
  const mutable = defaultOptions();

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === undefined) {
      continue;
    }

    if (applyBooleanFlag(mutable, arg)) {
      continue;
    }

    if (applyStringFlag(mutable, arg, argv[index + 1])) {
      index += 1;
      continue;
    }

    if (applyIntegerFlag(mutable, arg, argv[index + 1])) {
      index += 1;
      continue;
    }

    if (arg.startsWith("--")) {
      throw new Error(`benchmark-generation: unknown flag "${arg}".`);
    }

    handlePositionalModel(mutable, arg);
  }

  const options: BenchmarkOptions = {
    promptTokens: mutable.promptTokens,
    generationTokens: mutable.generationTokens,
    trials: mutable.trials,
    prefillStepSize: mutable.prefillStepSize,
    metalTrace: mutable.metalTrace,
  };
  validateOptions(options);
  return {
    model: mutable.model,
    options,
    reference: {
      captureMlxLmReference: mutable.captureMlxLmReference,
      enforceMlxLmDecodeBar: mutable.enforceMlxLmDecodeBar,
      mlxLmPython: mutable.mlxLmPython,
    },
  };
}

export async function loadBaselines(): Promise<BenchmarkBaselines> {
  const file = Bun.file(BASELINE_PATH);
  if (!(await file.exists())) {
    return emptyBaselines();
  }

  return parseBaselineData(await file.json());
}

export function parseBaselineData(parsed: unknown): BenchmarkBaselines {
  if (typeof parsed !== "object" || parsed === null) {
    throw new Error(`${BASELINE_PATH} must be an object with synthetic/parity sections.`);
  }

  const synthetic = "synthetic" in parsed ? parsed.synthetic : undefined;
  const parity = "parity" in parsed ? parsed.parity : undefined;
  return {
    synthetic: parseSection(synthetic, `${BASELINE_PATH}.synthetic`),
    parity: parseSection(parity, `${BASELINE_PATH}.parity`),
  };
}

export function selectTargets(
  mode: BenchmarkMode,
  baselines: BenchmarkBaselines,
  parsed: ParsedBenchmarkArgs,
): BenchmarkTarget[] {
  const baselineTargets = baselines[mode].targets;
  if (parsed.model !== undefined) {
    const baselineTarget = baselineTargets.find((target) => target.model === parsed.model);
    if (baselineTarget !== undefined) {
      return [
        {
          ...baselineTarget,
          promptTokens: parsed.options.promptTokens,
          generationTokens: parsed.options.generationTokens,
          prefillStepSize: parsed.options.prefillStepSize,
        },
      ];
    }

    return [
      {
        name: sanitizePathSegment(parsed.model),
        model: parsed.model,
        promptTokens: parsed.options.promptTokens,
        generationTokens: parsed.options.generationTokens,
        prefillStepSize: parsed.options.prefillStepSize,
      },
    ];
  }

  if (baselineTargets.length === 0) {
    throw new Error(
      `benchmark-generation: no --model was provided and ${BASELINE_PATH}.${mode}.targets is empty.`,
    );
  }

  return baselineTargets;
}

export function formatMlxLmReference(target: BenchmarkTarget): string | null {
  const reference = target.mlxLmReference;
  if (reference === undefined) {
    return null;
  }

  return [
    "MLX-LM reference:",
    `prompt_tps=${reference.promptTps.toFixed(3)}`,
    `generation_tps=${reference.generationTps.toFixed(3)}`,
    `peak_memory=${reference.peakMemoryGb.toFixed(3)}`,
    `captured_at=${reference.capturedAt}`,
  ].join(" ");
}

/** Compare our benchmark metrics against mlx-lm's own averages. */
export function compareAgainstMlxLmReference(
  metrics: TrialMetrics,
  reference: MlxLmReference,
): string[] {
  return metrics.generationTps < reference.generationTps * MLX_LM_DECODE_TOLERANCE_RATIO
    ? [
        `generation_tps below mlx-lm: mlx_lm=${reference.generationTps.toFixed(1)}, current=${metrics.generationTps.toFixed(1)}`,
      ]
    : [];
}

export function compareAgainstBaseline(target: BenchmarkTarget, metrics: TrialMetrics): string[] {
  const warnings: string[] = [];

  if (target.promptTps !== undefined && metrics.promptTps < target.promptTps / 2) {
    warnings.push(
      `prompt_tps regressed >2x: baseline=${target.promptTps.toFixed(1)}, current=${metrics.promptTps.toFixed(1)}`,
    );
  }
  if (target.generationTps !== undefined && metrics.generationTps < target.generationTps / 2) {
    warnings.push(
      `generation_tps regressed >2x: baseline=${target.generationTps.toFixed(1)}, current=${metrics.generationTps.toFixed(1)}`,
    );
  }
  if (target.peakMemoryGb !== undefined && metrics.peakMemoryGb > target.peakMemoryGb * 2) {
    warnings.push(
      `peak_memory grew >2x: baseline=${target.peakMemoryGb.toFixed(3)}, current=${metrics.peakMemoryGb.toFixed(3)}`,
    );
  }
  if (
    target.explicitEvalCountPerToken !== undefined &&
    metrics.explicitEvalCountPerToken > target.explicitEvalCountPerToken + 0.25
  ) {
    warnings.push(
      `evals_per_token regressed: baseline=${target.explicitEvalCountPerToken.toFixed(2)}, current=${metrics.explicitEvalCountPerToken.toFixed(2)}`,
    );
  }

  return warnings;
}

export function mean(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0) / Math.max(values.length, 1);
}

export function printTrial(prefix: string, metrics: TrialMetrics): void {
  console.log(
    `${prefix}prompt_tps=${metrics.promptTps.toFixed(3)}, generation_tps=${metrics.generationTps.toFixed(3)}, peak_memory=${metrics.peakMemoryGb.toFixed(3)}, evals_per_token=${metrics.explicitEvalCountPerToken.toFixed(2)}, total_time=${metrics.totalTimeSeconds.toFixed(3)}`,
  );
}

export function safeDecodedTokenLength(tokenizer: Tokenizer, tokenId: number): number {
  try {
    return tokenizer.decode([tokenId], { skipSpecialTokens: false }).length;
  } catch (error) {
    if (error instanceof Error && error.message.includes("out of range")) {
      return 0;
    }
    throw error;
  }
}

function benchmarkHelperPath(): string {
  return `${import.meta.dir}/benchmark-mlx-lm.py`;
}

function inheritedStringEnv(): Record<string, string> {
  return Object.fromEntries(
    Object.entries(Bun.env).filter((entry): entry is [string, string] => {
      const value = entry[1];
      return typeof value === "string";
    }),
  );
}

function parseMlxLmReferencePayload(output: string): MlxLmReference {
  const parsed = JSON.parse(output) as Record<string, unknown>;
  if (
    typeof parsed.prompt_tps !== "number" ||
    typeof parsed.generation_tps !== "number" ||
    typeof parsed.peak_memory_gb !== "number" ||
    typeof parsed.captured_at !== "string"
  ) {
    throw new Error("benchmark-generation: MLX-LM helper returned malformed benchmark JSON.");
  }

  return {
    promptTps: parsed.prompt_tps,
    generationTps: parsed.generation_tps,
    peakMemoryGb: parsed.peak_memory_gb,
    capturedAt: parsed.captured_at,
  };
}

/** Run the local MLX-LM helper on the exact prompt-token sequence used by mlxts. */
export async function captureMlxLmReference(
  modelPath: string,
  promptTokenIds: readonly number[],
  generationTokens: number,
  referenceOptions: ReferenceBenchmarkOptions,
): Promise<MlxLmReference | null> {
  if (!referenceOptions.captureMlxLmReference) {
    return null;
  }

  const pythonExecutable =
    referenceOptions.mlxLmPython ??
    Bun.env.MLX_LM_BENCH_PYTHON ??
    Bun.env.MLX_LM_PYTHON ??
    "python3";
  const process = Bun.spawn(
    [
      pythonExecutable,
      benchmarkHelperPath(),
      "--model",
      modelPath,
      "--prompt-token-ids-json",
      JSON.stringify(promptTokenIds),
      "--max-tokens",
      String(generationTokens),
    ],
    {
      stdout: "pipe",
      stderr: "pipe",
      env: {
        ...inheritedStringEnv(),
        ...mergedOfflineEnv(),
      },
    },
  );

  const [stdout, stderr, exitCode] = await Promise.all([
    new Response(process.stdout).text(),
    new Response(process.stderr).text(),
    process.exited,
  ]);

  if (exitCode !== 0) {
    const message = stderr.trim() || stdout.trim();
    if (
      message.includes("No module named mlx_lm") ||
      message.includes("No such file or directory") ||
      message.includes("command not found")
    ) {
      return null;
    }
    throw new Error(
      `benchmark-generation: MLX-LM reference benchmark failed${message === "" ? "" : `: ${message}`}`,
    );
  }

  return parseMlxLmReferencePayload(stdout);
}

export function sanitizePathSegment(value: string): string {
  return value.replaceAll(/[^a-zA-Z0-9._-]+/g, "-");
}

function directoryExists(path: string): boolean {
  const result = Bun.spawnSync(["/bin/test", "-d", path]);
  return result.exitCode === 0;
}

export async function resolveCachedSnapshotPath(modelSource: string): Promise<string> {
  if (modelSource.startsWith("/") || modelSource.startsWith(".")) {
    return modelSource;
  }

  const [owner, name] = modelSource.split("/");
  if (owner === undefined || name === undefined) {
    return modelSource;
  }

  const homeDir = Bun.env.HOME;
  if (homeDir === undefined) {
    return modelSource;
  }

  const cacheRoot =
    Bun.env.HF_HUB_CACHE ??
    Bun.env.HUGGINGFACE_HUB_CACHE ??
    Bun.env.HF_HOME?.concat("/hub") ??
    `${homeDir}/.cache/huggingface/hub`;
  const repoCacheDir = `${cacheRoot}/models--${owner}--${name}`;
  const mainRefPath = `${repoCacheDir}/refs/main`;
  const snapshotsDir = `${repoCacheDir}/snapshots`;
  const mainRef = Bun.file(mainRefPath);
  if (await mainRef.exists()) {
    const revision = (await mainRef.text()).trim();
    const snapshotPath = `${snapshotsDir}/${revision}`;
    if (directoryExists(snapshotPath)) {
      return snapshotPath;
    }
  }

  throw new Error(
    `benchmark-generation: no cached snapshot for ${modelSource}. Benchmark commands must run against local cached checkpoints only.`,
  );
}

function createTracePath(targetName: string): string {
  Bun.spawnSync(["mkdir", "-p", TRACE_DIR]);
  const timestamp = new Date().toISOString().replaceAll(/[:.]/g, "-");
  return `${TRACE_DIR}/${sanitizePathSegment(targetName)}-${timestamp}.gputrace`;
}

export function withBenchmarkRuntimeScope<T>(
  targetName: string,
  metalTrace: boolean,
  fn: () => T,
): T {
  const tracePath = metalTrace ? createTracePath(targetName) : null;
  if (tracePath !== null) {
    console.log(`Metal trace: ${tracePath}`);
    startMetalCapture(tracePath);
  }

  using generationStream = createStream();
  const previousWiredLimit = setWiredLimitBytes(getRecommendedWorkingSetBytes());

  try {
    return withDefaultStream(generationStream, fn);
  } finally {
    synchronize(generationStream);
    setWiredLimitBytes(previousWiredLimit);
    if (tracePath !== null) {
      stopMetalCapture();
    }
  }
}

export function createPromptTokenIds(length: number, vocabSize: number): number[] {
  const tokenIds: number[] = [];
  let state = 0x12345678;
  const usableVocab = Math.max(2, vocabSize - 1);

  for (let index = 0; index < length; index += 1) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    tokenIds.push((state % usableVocab) + 1);
  }

  return tokenIds;
}

/** Throw when the current decode throughput is below the chosen MLX-LM reference bar. */
export function enforceMlxLmDecodeBar(
  modelName: string,
  metrics: TrialMetrics,
  reference: MlxLmReference | null,
  options: ReferenceBenchmarkOptions,
): void {
  if (!options.enforceMlxLmDecodeBar || reference === null) {
    return;
  }

  if (metrics.generationTps >= reference.generationTps * MLX_LM_DECODE_TOLERANCE_RATIO) {
    return;
  }

  throw new Error(
    `benchmark-generation: decode throughput ${metrics.generationTps.toFixed(3)} tokens/s is below MLX-LM reference ${reference.generationTps.toFixed(3)} tokens/s for ${modelName}.`,
  );
}
