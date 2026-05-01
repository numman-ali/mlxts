#!/usr/bin/env bun

import {
  generateWhisperGreedyTranscription,
  loadPretrainedTokenizer,
  loadWhisperModel,
  type PretrainedLoadProgressEvent,
  parseWhisperFeatureExtractorConfig,
  prepareWhisperAudioFeatures,
  resolveWhisperModelSource,
  resolveWhisperSpecialTokens,
  type WhisperTask,
} from "@mlxts/transformers";
import { existsSync, readFileSync } from "fs";
import { join } from "path";

import { acquireRuntimeCommandLock } from "../../scripts/runtime-command-lock";
import { loadWhisperWavAudio } from "./wav";

type CliOptions = {
  source: string;
  audioPath: string;
  revision?: string;
  cacheDir?: string;
  hfToken?: string;
  localFilesOnly: boolean;
  task: WhisperTask;
  language?: string | null;
  withoutTimestamps: boolean;
  maxTokens: number;
  json: boolean;
};

type CliCommand = { kind: "help" } | { kind: "run"; options: CliOptions };

type CliOptionsDraft = {
  audioPath?: string;
  revision?: string;
  cacheDir?: string;
  hfToken?: string;
  localFilesOnly: boolean;
  task: WhisperTask;
  language?: string | null;
  withoutTimestamps: boolean;
  maxTokens: number;
  json: boolean;
};

type WhisperExampleResult = {
  source: string;
  snapshotPath: string;
  audioPath: string;
  sampleRate: number;
  channels: number;
  frames: number;
  durationSeconds: number;
  task: WhisperTask;
  language: string | null;
  withoutTimestamps: boolean;
  maxTokens: number;
  promptTokens: number;
  generatedTokens: number;
  stoppedReason: "eos" | "max_tokens";
  text: string;
  elapsedMs: number;
};

type RuntimeLock = {
  [Symbol.dispose](): void;
};

type WhisperExampleRuntime = {
  stdout?: (text: string) => void;
  stderr?: (text: string) => void;
  acquireLock?: () => RuntimeLock;
  runExample?: (cli: CliOptions, progress: (line: string) => void) => Promise<WhisperExampleResult>;
};

class WhisperExampleUsageError extends Error {}

function quoteScalar(value: string | number | boolean | null): string {
  return typeof value === "string" ? JSON.stringify(value) : String(value);
}

export function formatUsage(): string {
  return [
    "description: Run one Whisper speech-to-text proof over a 16 kHz WAV file",
    "usage[2]:",
    "  bun run examples/whisper/index.ts <snapshot-source> --audio ./speech.wav",
    "  bun run examples/whisper/index.ts openai/whisper-tiny --local-files-only --audio ./speech.wav --max-tokens 64 --json",
    "arguments[1]{name,description}:",
    '  "snapshot-source","Local Whisper snapshot directory or Hugging Face model id"',
    "options[11]{flag,description}:",
    '  "--audio <path>","Required 16 kHz PCM or float WAV file"',
    '  "--revision <rev>","Hub revision; default main"',
    '  "--cache-dir <path>","Hub cache directory; default Hugging Face cache"',
    '  "--hf-token <token>","Hub access token; defaults to HF token environment or cache file"',
    '  "--local-files-only","Use only an already-cached Hub snapshot"',
    '  "--task <transcribe|translate>","Whisper task token; default transcribe"',
    '  "--language <code>","Language token such as en; default en when the tokenizer has it"',
    '  "--no-language","Omit the language token for English-only checkpoints"',
    '  "--timestamps","Allow timestamp tokens; default text-only no-timestamps prompt"',
    '  "--max-tokens <n>","Maximum generated text tokens; default 64"',
    '  "--json","Emit final result as JSON"',
    '  "--help","Show this help"',
    "exit_codes[3]{code,meaning}:",
    '  0,"transcription proof passed or help"',
    '  1,"runtime or generation failure"',
    '  2,"usage error"',
  ].join("\n");
}

function readStringFlag(flag: string, value: string | undefined): string {
  if (value === undefined || value.trim() === "" || value.startsWith("--")) {
    throw new WhisperExampleUsageError(`Missing value for ${flag}.`);
  }
  return value;
}

function readPositiveIntegerFlag(flag: string, value: string | undefined): number {
  const raw = readStringFlag(flag, value);
  const parsed = Number(raw);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new WhisperExampleUsageError(`Expected ${flag} to be a positive integer.`);
  }
  return parsed;
}

function readTaskFlag(value: string | undefined): WhisperTask {
  const raw = readStringFlag("--task", value);
  if (raw === "transcribe" || raw === "translate") {
    return raw;
  }
  throw new WhisperExampleUsageError("Expected --task to be transcribe or translate.");
}

function applyFlag(argv: readonly string[], index: number, draft: CliOptionsDraft): number {
  const arg = argv[index];
  switch (arg) {
    case "--audio":
      draft.audioPath = readStringFlag(arg, argv[index + 1]);
      return index + 1;
    case "--revision":
      draft.revision = readStringFlag(arg, argv[index + 1]);
      return index + 1;
    case "--cache-dir":
      draft.cacheDir = readStringFlag(arg, argv[index + 1]);
      return index + 1;
    case "--hf-token":
      draft.hfToken = readStringFlag(arg, argv[index + 1]);
      return index + 1;
    case "--local-files-only":
      draft.localFilesOnly = true;
      return index;
    case "--task":
      draft.task = readTaskFlag(argv[index + 1]);
      return index + 1;
    case "--language":
      draft.language = readStringFlag(arg, argv[index + 1]);
      return index + 1;
    case "--no-language":
      draft.language = null;
      return index;
    case "--timestamps":
      draft.withoutTimestamps = false;
      return index;
    case "--max-tokens":
      draft.maxTokens = readPositiveIntegerFlag(arg, argv[index + 1]);
      return index + 1;
    case "--json":
      draft.json = true;
      return index;
    default:
      throw new WhisperExampleUsageError(
        arg === undefined ? "Missing argument." : `Unknown argument: ${arg}`,
      );
  }
}

function applyOptionalCliOptions(options: CliOptions, draft: CliOptionsDraft): void {
  if (draft.revision !== undefined) {
    options.revision = draft.revision;
  }
  if (draft.cacheDir !== undefined) {
    options.cacheDir = draft.cacheDir;
  }
  if (draft.hfToken !== undefined) {
    options.hfToken = draft.hfToken;
  }
  if (draft.language !== undefined) {
    options.language = draft.language;
  }
}

export function parseCommand(argv: readonly string[]): CliCommand {
  if (argv.some((arg) => arg === "--help" || arg === "-h")) {
    return { kind: "help" };
  }
  const source = argv[0];
  if (source === undefined || source.trim() === "" || source.startsWith("--")) {
    throw new WhisperExampleUsageError("Missing snapshot source.");
  }

  const draft: CliOptionsDraft = {
    localFilesOnly: false,
    task: "transcribe",
    withoutTimestamps: true,
    maxTokens: 64,
    json: false,
  };
  for (let index = 1; index < argv.length; index += 1) {
    index = applyFlag(argv, index, draft);
  }
  if (draft.audioPath === undefined) {
    throw new WhisperExampleUsageError("Missing required --audio <path>.");
  }

  const options: CliOptions = {
    source,
    audioPath: draft.audioPath,
    localFilesOnly: draft.localFilesOnly,
    task: draft.task,
    withoutTimestamps: draft.withoutTimestamps,
    maxTokens: draft.maxTokens,
    json: draft.json,
  };
  applyOptionalCliOptions(options, draft);
  return { kind: "run", options };
}

export function parseArgs(argv: readonly string[]): CliOptions {
  const command = parseCommand(argv);
  if (command.kind === "help") {
    throw new WhisperExampleUsageError("Help is not a Whisper proof command.");
  }
  return command.options;
}

function readJsonIfExists(path: string): unknown {
  return existsSync(path) ? JSON.parse(readFileSync(path, "utf8")) : {};
}

function readFeatureExtractorConfig(snapshotPath: string): unknown {
  const preprocessorConfigPath = join(snapshotPath, "preprocessor_config.json");
  if (existsSync(preprocessorConfigPath)) {
    return readJsonIfExists(preprocessorConfigPath);
  }
  return readJsonIfExists(join(snapshotPath, "feature_extractor_config.json"));
}

function formatResolveProgress(event: PretrainedLoadProgressEvent): string {
  if (event.stage === "resolve" && event.status === "start") {
    return `Resolving snapshot source: ${event.source}`;
  }
  if (event.stage === "resolve") {
    return `Resolved ${event.sourceKind} snapshot: ${event.directory} (${event.fileCount} files, ${event.totalBytes} bytes)`;
  }
  if (event.stage === "download") {
    return `Snapshot ${event.status} ${event.index}/${event.totalFiles}: ${event.relativePath}`;
  }
  if (event.stage === "model") {
    return `Model ${event.status}: ${event.shardCount} shard(s)`;
  }
  return `Tokenizer ${event.status}: ${event.directory}`;
}

function printRunIntro(cli: CliOptions, writeLine: (line: string) => void): void {
  writeLine(`Snapshot source: ${cli.source}`);
  if (cli.revision !== undefined) {
    writeLine(`Snapshot revision: ${cli.revision}`);
  }
  if (cli.localFilesOnly) {
    writeLine("Local files only: true");
  }
  writeLine(`Audio: ${cli.audioPath}`);
  writeLine(`Task: ${cli.task}`);
  writeLine(`Max tokens: ${cli.maxTokens}`);
  writeLine("");
}

export async function runWhisperExample(
  cli: CliOptions,
  progress: (line: string) => void,
): Promise<WhisperExampleResult> {
  const startedAt = performance.now();
  printRunIntro(cli, progress);
  const snapshotPath = await resolveWhisperModelSource(cli.source, {
    ...(cli.revision === undefined ? {} : { revision: cli.revision }),
    ...(cli.cacheDir === undefined ? {} : { cacheDir: cli.cacheDir }),
    ...(cli.hfToken === undefined ? {} : { accessToken: cli.hfToken }),
    localFilesOnly: cli.localFilesOnly,
    onProgress: (event) => {
      progress(formatResolveProgress(event));
    },
  });

  progress("Loading Whisper model...");
  using model = await loadWhisperModel(snapshotPath, { strictUnexpectedWeights: true });
  progress("Loading Whisper tokenizer...");
  const tokenizer = await loadPretrainedTokenizer(snapshotPath);
  const tokenizerConfig = readJsonIfExists(join(snapshotPath, "tokenizer_config.json"));
  const featureConfig = parseWhisperFeatureExtractorConfig(
    readFeatureExtractorConfig(snapshotPath),
  );
  const specialTokens = resolveWhisperSpecialTokens(tokenizerConfig, {
    tokenizer,
    config: model.config,
  });

  progress("Loading WAV audio...");
  const wav = loadWhisperWavAudio(cli.audioPath, {
    expectedSampleRate: featureConfig.samplingRate,
  });
  try {
    progress("Preparing Whisper audio features...");
    const prepared = prepareWhisperAudioFeatures(wav.audio, featureConfig);
    try {
      progress("Running greedy Whisper decode...");
      const result = generateWhisperGreedyTranscription(
        model,
        prepared.inputFeatures,
        tokenizer,
        specialTokens,
        {
          task: cli.task,
          ...(cli.language === undefined ? {} : { language: cli.language }),
          withoutTimestamps: cli.withoutTimestamps,
          maxNewTokens: cli.maxTokens,
          onToken: (event) => {
            if (event.step === 1 || event.step % 8 === 0) {
              progress(`Decoded token ${event.step}: ${event.tokenId}`);
            }
          },
        },
      );

      return {
        source: cli.source,
        snapshotPath,
        audioPath: wav.path,
        sampleRate: wav.sampleRate,
        channels: wav.channels,
        frames: wav.frames,
        durationSeconds: Math.round(wav.durationSeconds * 1000) / 1000,
        task: cli.task,
        language: cli.language ?? (specialTokens.languageTokenIds.has("en") ? "en" : null),
        withoutTimestamps: cli.withoutTimestamps,
        maxTokens: cli.maxTokens,
        promptTokens: result.promptTokenIds.length,
        generatedTokens: result.generatedTokens,
        stoppedReason: result.stoppedReason,
        text: result.text,
        elapsedMs: Math.round((performance.now() - startedAt) * 100) / 100,
      };
    } finally {
      prepared.inputFeatures.free();
    }
  } finally {
    wav.audio.free();
  }
}

export function formatSuccess(report: WhisperExampleResult): string {
  return [
    "whisper_example:",
    "  status: passed",
    `  source: ${quoteScalar(report.source)}`,
    `  snapshot_path: ${quoteScalar(report.snapshotPath)}`,
    `  audio_path: ${quoteScalar(report.audioPath)}`,
    `  sample_rate: ${report.sampleRate}`,
    `  channels: ${report.channels}`,
    `  duration_seconds: ${report.durationSeconds}`,
    `  task: ${quoteScalar(report.task)}`,
    `  language: ${quoteScalar(report.language)}`,
    `  without_timestamps: ${report.withoutTimestamps}`,
    `  max_tokens: ${report.maxTokens}`,
    `  prompt_tokens: ${report.promptTokens}`,
    `  generated_tokens: ${report.generatedTokens}`,
    `  stopped_reason: ${quoteScalar(report.stoppedReason)}`,
    `  elapsed_ms: ${report.elapsedMs}`,
    `  text: ${quoteScalar(report.text)}`,
  ].join("\n");
}

function formatError(message: string, code: "usage" | "runtime"): string {
  return [
    "error:",
    `  code: ${quoteScalar(code)}`,
    `  message: ${quoteScalar(message)}`,
    "help[1]:",
    '  "Run `bun run examples/whisper/index.ts --help` for options"',
  ].join("\n");
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export async function runWhisperExampleCommand(
  argv: readonly string[],
  runtime: WhisperExampleRuntime = {},
): Promise<number> {
  const stdout = runtime.stdout ?? console.log;
  const stderr = runtime.stderr ?? console.error;
  let command: CliCommand;
  try {
    command = parseCommand(argv);
  } catch (error) {
    stdout(formatError(errorMessage(error), "usage"));
    return error instanceof WhisperExampleUsageError ? 2 : 1;
  }

  if (command.kind === "help") {
    stdout(formatUsage());
    return 0;
  }

  const acquireLock = runtime.acquireLock ?? (() => acquireRuntimeCommandLock("example:whisper"));
  const runExample = runtime.runExample ?? runWhisperExample;
  let lock: RuntimeLock | undefined;
  try {
    lock = acquireLock();
    const report = await runExample(command.options, stderr);
    stdout(command.options.json ? JSON.stringify(report, null, 2) : formatSuccess(report));
    return 0;
  } catch (error) {
    stdout(formatError(errorMessage(error), "runtime"));
    if (error instanceof Error && error.stack !== undefined) {
      stderr(error.stack);
    }
    return 1;
  } finally {
    lock?.[Symbol.dispose]();
  }
}

if (import.meta.main) {
  const exitCode = await runWhisperExampleCommand(Bun.argv.slice(2));
  process.exit(exitCode);
}
