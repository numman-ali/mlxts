import {
  createStream,
  getActiveMemoryBytes,
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
const MLX_LM_MEMORY_WARNING_RATIO = 1.15;

export type BenchmarkMode = "synthetic" | "parity";
export type BenchmarkDecodeSchedule = "async" | "sync";

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
    trialCount?: number;
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
  memorySampleInterval: number;
  decodeSchedule: BenchmarkDecodeSchedule;
  materializeCacheEachToken: boolean;
};

export type ParsedBenchmarkArgs = {
  model?: string;
  options: BenchmarkOptions;
  reference: ReferenceBenchmarkOptions;
};

export type BenchmarkCommand = { kind: "help" } | { kind: "run"; parsed: ParsedBenchmarkArgs };

export type BenchmarkProgress = (text: string) => void;

export type ReferenceBenchmarkOptions = {
  captureMlxLmReference: boolean;
  enforceMlxLmDecodeBar: boolean;
  requireMlxLmReference: boolean;
  allowMlxLmExtraWeights: boolean;
  mlxLmPython?: string;
};

export type MlxLmCaptureOptions = {
  generationTokens: number;
  prefillStepSize: number;
  trials: number;
};

export type TrialMetrics = {
  promptTps: number;
  generationTps: number;
  peakMemoryGb: number;
  activeMemoryStartGb: number;
  activeMemoryEndGb: number;
  activeMemoryDeltaGb: number;
  activeMemoryMaxGb: number;
  activeMemorySlopeMbPerToken: number;
  explicitEvalCountPerToken: number;
  totalTimeSeconds: number;
};

export type DecodeMemoryMetrics = Pick<
  TrialMetrics,
  | "activeMemoryStartGb"
  | "activeMemoryEndGb"
  | "activeMemoryDeltaGb"
  | "activeMemoryMaxGb"
  | "activeMemorySlopeMbPerToken"
>;

export type DecodeMemoryTracker = {
  sample: () => void;
  finish: (generationTokens: number) => DecodeMemoryMetrics;
};

export type BenchmarkCommandReport = {
  name: string;
  model: string;
  snapshotPath: string;
  promptTokens: number;
  generationTokens: number;
  prefillStepSize: number;
  trials: number;
  decodeSchedule: BenchmarkDecodeSchedule;
  materializeCacheEachToken: boolean;
  metrics: TrialMetrics;
  mlxLmReference?: MlxLmReference;
  warnings: string[];
};

type MutableBenchmarkOptions = {
  model?: string;
  promptTokens: number;
  generationTokens: number;
  trials: number;
  prefillStepSize: number;
  metalTrace: boolean;
  memorySampleInterval: number;
  decodeSchedule: BenchmarkDecodeSchedule;
  materializeCacheEachToken: boolean;
  captureMlxLmReference: boolean;
  enforceMlxLmDecodeBar: boolean;
  requireMlxLmReference: boolean;
  allowMlxLmExtraWeights: boolean;
  mlxLmPython?: string;
};

export class BenchmarkUsageError extends Error {}

function defaultOptions(): MutableBenchmarkOptions {
  return {
    model: undefined,
    promptTokens: 1024,
    generationTokens: 128,
    trials: 3,
    prefillStepSize: 2048,
    metalTrace: false,
    memorySampleInterval: 64,
    decodeSchedule: "async",
    materializeCacheEachToken: false,
    captureMlxLmReference: true,
    enforceMlxLmDecodeBar: false,
    requireMlxLmReference: false,
    allowMlxLmExtraWeights: false,
    mlxLmPython: undefined,
  };
}

function emptyBaselines(): BenchmarkBaselines {
  return {
    synthetic: { targets: [] },
    parity: { targets: [] },
  };
}

function readRequiredValue(flag: string, value: string | undefined): string {
  if (value === undefined || value.trim() === "" || value.startsWith("--")) {
    throw new BenchmarkUsageError(`benchmark-generation: ${flag} requires a value.`);
  }
  return value;
}

function parseRequiredNumber(value: string | undefined, flag: string): number {
  const raw = readRequiredValue(flag, value);
  const parsed = /^\d+$/.test(raw) ? Number(raw) : Number.NaN;
  if (!Number.isInteger(parsed)) {
    throw new BenchmarkUsageError(`benchmark-generation: ${flag} expects an integer value.`);
  }
  return parsed;
}

function validateOptions(options: BenchmarkOptions): void {
  if (!Number.isInteger(options.promptTokens) || options.promptTokens <= 1) {
    throw new BenchmarkUsageError(
      "benchmark-generation: promptTokens must be an integer greater than 1.",
    );
  }
  if (!Number.isInteger(options.generationTokens) || options.generationTokens <= 0) {
    throw new BenchmarkUsageError(
      "benchmark-generation: generationTokens must be a positive integer.",
    );
  }
  if (!Number.isInteger(options.trials) || options.trials <= 0) {
    throw new BenchmarkUsageError("benchmark-generation: trials must be a positive integer.");
  }
  if (!Number.isInteger(options.prefillStepSize) || options.prefillStepSize <= 0) {
    throw new BenchmarkUsageError(
      "benchmark-generation: prefillStepSize must be a positive integer.",
    );
  }
  if (!Number.isInteger(options.memorySampleInterval) || options.memorySampleInterval <= 0) {
    throw new BenchmarkUsageError(
      "benchmark-generation: memorySampleInterval must be a positive integer.",
    );
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
            ...(typeof entry.mlxLmReference.trialCount === "number"
              ? { trialCount: entry.mlxLmReference.trialCount }
              : {}),
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
    case "--memory-sample-interval":
      mutable.memorySampleInterval = parsed;
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
    case "--sync-decode":
      mutable.decodeSchedule = "sync";
      return true;
    case "--materialize-cache-each-token":
      mutable.materializeCacheEachToken = true;
      return true;
    case "--capture-mlx-lm-reference":
      mutable.captureMlxLmReference = true;
      return true;
    case "--skip-mlx-lm-reference":
    case "--no-capture-mlx-lm-reference":
      mutable.captureMlxLmReference = false;
      return true;
    case "--enforce-mlx-lm-decode-bar":
      mutable.enforceMlxLmDecodeBar = true;
      return true;
    case "--require-mlx-lm-reference":
      mutable.requireMlxLmReference = true;
      return true;
    case "--mlx-lm-allow-extra-weights":
      mutable.allowMlxLmExtraWeights = true;
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
      mutable.model = readRequiredValue(flag, value);
      return true;
    case "--mlx-lm-python":
      mutable.mlxLmPython = readRequiredValue(flag, value);
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

  throw new BenchmarkUsageError(`benchmark-generation: unexpected positional argument "${arg}".`);
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
      throw new BenchmarkUsageError(`benchmark-generation: unknown flag "${arg}".`);
    }

    if (arg.startsWith("-")) {
      throw new BenchmarkUsageError(`benchmark-generation: unknown argument "${arg}".`);
    }

    handlePositionalModel(mutable, arg);
  }

  const options: BenchmarkOptions = {
    promptTokens: mutable.promptTokens,
    generationTokens: mutable.generationTokens,
    trials: mutable.trials,
    prefillStepSize: mutable.prefillStepSize,
    metalTrace: mutable.metalTrace,
    memorySampleInterval: mutable.memorySampleInterval,
    decodeSchedule: mutable.decodeSchedule,
    materializeCacheEachToken: mutable.materializeCacheEachToken,
  };
  validateOptions(options);
  if (mutable.requireMlxLmReference && !mutable.captureMlxLmReference) {
    throw new BenchmarkUsageError(
      "benchmark-generation: --require-mlx-lm-reference cannot be combined with --skip-mlx-lm-reference.",
    );
  }
  return {
    model: mutable.model,
    options,
    reference: {
      captureMlxLmReference: mutable.captureMlxLmReference,
      enforceMlxLmDecodeBar: mutable.enforceMlxLmDecodeBar,
      requireMlxLmReference: mutable.requireMlxLmReference,
      allowMlxLmExtraWeights: mutable.allowMlxLmExtraWeights,
      mlxLmPython: mutable.mlxLmPython,
    },
  };
}

export function parseBenchmarkCommand(argv: readonly string[]): BenchmarkCommand {
  if (argv.some((arg) => arg === "--help" || arg === "-h")) {
    return { kind: "help" };
  }
  return { kind: "run", parsed: parseBenchmarkArgs(argv) };
}

export type BenchmarkCommandMode = "synthetic" | "parity";

function toon(value: string | number | boolean | null): string {
  if (typeof value === "number") {
    return Number.isFinite(value) ? value.toString() : "null";
  }
  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }
  if (value === null) {
    return "null";
  }
  return JSON.stringify(value);
}

function formatMultilineField(name: string, value: string): string[] {
  if (!value.includes("\n")) {
    return [`  ${name}: ${toon(value)}`];
  }
  return [`  ${name}: |`, ...value.split("\n").map((line) => `    ${line}`)];
}

function nullableMetric(value: number | undefined, fractionDigits: number): string {
  return value === undefined ? "null" : value.toFixed(fractionDigits);
}

function commandName(mode: BenchmarkCommandMode): string {
  return mode === "synthetic" ? "bench:generation" : "bench:generation:parity";
}

function modeDescription(mode: BenchmarkCommandMode): string {
  return mode === "synthetic"
    ? "Benchmark @mlxts/transformers synthetic generation throughput"
    : "Benchmark @mlxts/transformers generation parity against MLX-LM";
}

export function formatBenchmarkUsage(mode: BenchmarkCommandMode): string {
  const command = commandName(mode);
  const referenceOptions =
    mode === "parity"
      ? [
          '  "--skip-mlx-lm-reference","Use stored baseline references only"',
          '  "--capture-mlx-lm-reference","Run the MLX-LM helper before mlxts; default true"',
          '  "--require-mlx-lm-reference","Fail when no live or stored MLX-LM reference exists"',
          '  "--enforce-mlx-lm-decode-bar","Fail when decode throughput trails MLX-LM beyond tolerance"',
          '  "--mlx-lm-allow-extra-weights","Pass allow-extra-weights to the MLX-LM helper"',
          '  "--mlx-lm-python <path>","Python executable for the MLX-LM helper"',
        ]
      : [];
  return [
    `description: ${modeDescription(mode)}`,
    "usage[3]:",
    `  bun run ${command} -- --model <repo-or-path>`,
    `  bun run ${command} -- --model <repo-or-path> --prompt-tokens 1024 --generation-tokens 128 --trials 3`,
    `  bun run ${command} -- <repo-or-path> --sync-decode`,
    `options[${10 + referenceOptions.length}]{flag,description}:`,
    '  "--model <repo-or-path>","Model id/path; may also be the first positional argument"',
    '  "--prompt-tokens <n>","Synthetic prompt token count; default 1024"',
    '  "--generation-tokens <n>","Decode token count; default 128"',
    '  "--trials <n>","Measured trial count after warmup; default 3"',
    '  "--prefill-step-size <n>","Prompt prefill chunk size; default 2048"',
    '  "--memory-sample-interval <n>","Decode memory sample cadence; default 64"',
    '  "--sync-decode","Use scalar-synchronized decode instead of async scheduled decode"',
    '  "--materialize-cache-each-token","Force cache materialization after each decode token"',
    '  "--metal-trace","Capture a Metal trace under benchmarks/traces"',
    ...referenceOptions,
    '  "--help","Show this help"',
    "exit_codes[3]{code,meaning}:",
    '  0,"benchmark completed"',
    '  1,"runtime or benchmark failure"',
    '  2,"usage error"',
  ].join("\n");
}

export function formatBenchmarkError(message: string, help: string): string {
  return ["error:", ...formatMultilineField("message", message), `help: ${toon(help)}`].join("\n");
}

export function formatBenchmarkSuccess(
  mode: BenchmarkCommandMode,
  reports: readonly BenchmarkCommandReport[],
): string {
  const rows = reports.map((report) =>
    [
      toon(report.name),
      toon(report.model),
      toon(report.snapshotPath),
      report.promptTokens.toString(),
      report.generationTokens.toString(),
      report.prefillStepSize.toString(),
      report.trials.toString(),
      toon(report.decodeSchedule),
      toon(report.materializeCacheEachToken),
      report.metrics.promptTps.toFixed(3),
      report.metrics.generationTps.toFixed(3),
      report.metrics.peakMemoryGb.toFixed(3),
      report.metrics.explicitEvalCountPerToken.toFixed(2),
      report.metrics.totalTimeSeconds.toFixed(3),
      nullableMetric(report.mlxLmReference?.generationTps, 3),
      report.warnings.length.toString(),
    ].join(","),
  );
  const warningRows = reports.flatMap((report) =>
    report.warnings.map((warning) => [toon(report.name), toon(warning)].join(",")),
  );

  return [
    "generation_benchmark:",
    "  status: passed",
    `  mode: ${toon(mode)}`,
    `  targets: ${reports.length}`,
    `targets[${rows.length}]{name,model,snapshot_path,prompt_tokens,generation_tokens,prefill_step_size,trials,decode_schedule,materialize_cache_each_token,prompt_tps,generation_tps,peak_memory_gb,evals_per_token,total_time_s,mlx_lm_generation_tps,warning_count}:`,
    ...rows.map((row) => `  ${row}`),
    ...(warningRows.length === 0
      ? []
      : [
          `warnings[${warningRows.length}]{target,message}:`,
          ...warningRows.map((row) => `  ${row}`),
        ]),
  ].join("\n");
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
    throw new BenchmarkUsageError(
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
    ...(reference.trialCount === undefined ? [] : [`trials=${reference.trialCount}`]),
    `captured_at=${reference.capturedAt}`,
  ].join(" ");
}

/** Compare our benchmark metrics against mlx-lm's own averages. */
export function compareAgainstMlxLmReference(
  metrics: TrialMetrics,
  reference: MlxLmReference,
): string[] {
  const warnings: string[] = [];
  if (metrics.generationTps < reference.generationTps * MLX_LM_DECODE_TOLERANCE_RATIO) {
    warnings.push(
      `generation_tps below mlx-lm: mlx_lm=${reference.generationTps.toFixed(1)}, current=${metrics.generationTps.toFixed(1)}`,
    );
  }
  if (metrics.peakMemoryGb > reference.peakMemoryGb * MLX_LM_MEMORY_WARNING_RATIO) {
    warnings.push(
      `peak_memory above mlx-lm: mlx_lm=${reference.peakMemoryGb.toFixed(3)}, current=${metrics.peakMemoryGb.toFixed(3)}`,
    );
  }
  return warnings;
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

export function printTrial(
  prefix: string,
  metrics: TrialMetrics,
  progress: BenchmarkProgress = console.log,
): void {
  progress(
    `${prefix}prompt_tps=${metrics.promptTps.toFixed(3)}, generation_tps=${metrics.generationTps.toFixed(3)}, peak_memory=${metrics.peakMemoryGb.toFixed(3)}, active_start=${metrics.activeMemoryStartGb.toFixed(3)}, active_end=${metrics.activeMemoryEndGb.toFixed(3)}, active_delta=${metrics.activeMemoryDeltaGb.toFixed(3)}, active_max=${metrics.activeMemoryMaxGb.toFixed(3)}, active_slope_mb_per_token=${metrics.activeMemorySlopeMbPerToken.toFixed(2)}, evals_per_token=${metrics.explicitEvalCountPerToken.toFixed(2)}, total_time=${metrics.totalTimeSeconds.toFixed(3)}`,
  );
}

/** Track live allocator growth during steady-state decode without adding a new harness. */
export function createDecodeMemoryTracker(): DecodeMemoryTracker {
  const startBytes = getActiveMemoryBytes();
  let maxBytes = startBytes;

  return {
    sample() {
      maxBytes = Math.max(maxBytes, getActiveMemoryBytes());
    },
    finish(generationTokens: number) {
      const endBytes = getActiveMemoryBytes();
      maxBytes = Math.max(maxBytes, endBytes);
      const deltaBytes = endBytes - startBytes;
      return {
        activeMemoryStartGb: startBytes / 1e9,
        activeMemoryEndGb: endBytes / 1e9,
        activeMemoryDeltaGb: deltaBytes / 1e9,
        activeMemoryMaxGb: maxBytes / 1e9,
        activeMemorySlopeMbPerToken: deltaBytes / 1e6 / Math.max(generationTokens, 1),
      };
    },
  };
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

type ParsedMlxLmReferencePayload = MlxLmReference & {
  generationTokens: number;
  finishReason: string;
};

function parseMlxLmReferencePayload(output: string): ParsedMlxLmReferencePayload {
  const parsed = JSON.parse(output) as Record<string, unknown>;
  if (
    typeof parsed.prompt_tps !== "number" ||
    typeof parsed.generation_tps !== "number" ||
    typeof parsed.peak_memory_gb !== "number" ||
    typeof parsed.captured_at !== "string" ||
    typeof parsed.generation_tokens !== "number" ||
    typeof parsed.finish_reason !== "string" ||
    (typeof parsed.trial_count !== "number" && parsed.trial_count !== undefined)
  ) {
    throw new Error("benchmark-generation: MLX-LM helper returned malformed benchmark JSON.");
  }

  return {
    promptTps: parsed.prompt_tps,
    generationTps: parsed.generation_tps,
    peakMemoryGb: parsed.peak_memory_gb,
    capturedAt: parsed.captured_at,
    ...(typeof parsed.trial_count === "number" ? { trialCount: parsed.trial_count } : {}),
    generationTokens: parsed.generation_tokens,
    finishReason: parsed.finish_reason,
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function extractPositiveInteger(value: unknown): number | null {
  return Number.isInteger(value) && typeof value === "number" && value > 0 ? value : null;
}

function configVocabSize(config: Record<string, unknown>): number | null {
  const directVocabSize = extractPositiveInteger(config.vocab_size);
  if (directVocabSize !== null) {
    return directVocabSize;
  }

  const textConfig = config.text_config;
  if (!isRecord(textConfig)) {
    return null;
  }
  return extractPositiveInteger(textConfig.vocab_size);
}

/** Read enough checkpoint config metadata to create deterministic benchmark prompts. */
export async function readBenchmarkVocabSize(snapshotPath: string): Promise<number> {
  const configPath = `${snapshotPath}/config.json`;
  const payload = await Bun.file(configPath).json();
  if (!isRecord(payload)) {
    throw new Error(`benchmark-generation: ${configPath} must contain a JSON object.`);
  }

  const vocabSize = configVocabSize(payload);
  if (vocabSize === null) {
    throw new Error(
      `benchmark-generation: ${configPath} must include a positive vocab_size, directly or under text_config.`,
    );
  }
  return vocabSize;
}

/** Run the local MLX-LM helper on the exact prompt-token sequence used by mlxts. */
export async function captureMlxLmReference(
  modelPath: string,
  promptTokenIds: readonly number[],
  referenceOptions: ReferenceBenchmarkOptions,
  captureOptions: MlxLmCaptureOptions,
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
      String(captureOptions.generationTokens),
      "--prefill-step-size",
      String(captureOptions.prefillStepSize),
      "--trials",
      String(captureOptions.trials),
      "--warmup-trials",
      "1",
      ...(referenceOptions.allowMlxLmExtraWeights ? ["--allow-extra-weights"] : []),
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

  const parsed = parseMlxLmReferencePayload(stdout);
  if (
    parsed.generationTokens !== captureOptions.generationTokens ||
    parsed.finishReason !== "length"
  ) {
    throw new Error(
      `benchmark-generation: MLX-LM helper did not complete the requested fixed-length decode (tokens=${parsed.generationTokens}, finish_reason=${parsed.finishReason}).`,
    );
  }

  return {
    promptTps: parsed.promptTps,
    generationTps: parsed.generationTps,
    peakMemoryGb: parsed.peakMemoryGb,
    ...(parsed.trialCount === undefined ? {} : { trialCount: parsed.trialCount }),
    capturedAt: parsed.capturedAt,
  };
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
  progress: BenchmarkProgress = console.log,
): T {
  const tracePath = metalTrace ? createTracePath(targetName) : null;
  if (tracePath !== null) {
    progress(`Metal trace: ${tracePath}`);
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
