import { acquireRuntimeCommandLock } from "../../scripts/runtime-command-lock";
import { parseTrainingProofCommand, type TrainingProofArgs, TrainingProofUsageError } from "./args";
import type { TrainingProofReport } from "./types";
import { runTrainingProof } from "./workflow";

type RuntimeLock = {
  [Symbol.dispose](): void;
};

type TrainingProofRuntime = {
  stdout?: (text: string) => void;
  stderr?: (text: string) => void;
  acquireLock?: () => RuntimeLock;
  runProof?: (
    options: TrainingProofArgs,
    progress: (line: string) => void,
  ) => Promise<TrainingProofReport>;
};

function quoteScalar(value: string | number | boolean | null): string {
  return typeof value === "string" ? JSON.stringify(value) : String(value);
}

function formatOptionalNumber(value: number | undefined): string {
  return value === undefined ? "null" : String(Math.round(value * 10000) / 10000);
}

export function formatTrainingProofUsage(): string {
  return [
    "description: Run the canonical Phase 8 LoRA, QLoRA, SFT, and DPO proof",
    "usage[2]:",
    "  bun run proof:training",
    "  bun run proof:training --dataset-source tiny --train-limit 8 --eval-limit 4 --steps 2",
    "options[14]{flag,description}:",
    '  "--source <id>","Dense checkpoint source; default meta-llama/Llama-3.2-1B-Instruct"',
    '  "--quantized-output <dir>","4-bit snapshot output directory"',
    '  "--adapter-output <dir>","Proof adapter output directory"',
    '  "--report <path>","Training proof JSON report path"',
    '  "--dataset-source tiny|huggingface","Dataset source; default huggingface"',
    '  "--train-limit <n>","Training examples per task; default 128"',
    '  "--eval-limit <n>","Held-out examples per task; default 32"',
    '  "--batch-size <n>","Batch size; default 4"',
    '  "--steps <n>","Optimizer steps per stage; default 16"',
    '  "--max-seq-len <n>","Token cap per example; default 1024"',
    '  "--seed <n>","Deterministic batching seed; default 7"',
    '  "--stages <list>","Comma-separated subset of lora,qlora,sft,dpo"',
    '  "--dpo-profile canonical|handbook","DPO recipe profile; default canonical"',
    '  "--help","Show this help"',
    "exit_codes[3]{code,meaning}:",
    '  0,"proof passed or help"',
    '  1,"runtime or proof failure"',
    '  2,"usage error"',
  ].join("\n");
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function formatBlockField(name: string, value: string): string[] {
  const lines = value.split(/\r?\n/);
  return [`  ${name}: |`, ...lines.map((line) => `    ${line}`)];
}

export function formatTrainingProofError(message: string, code: "usage" | "runtime"): string {
  return [
    "error:",
    `  code: ${quoteScalar(code)}`,
    ...formatBlockField("message", message),
    "help[1]:",
    '  "Run `bun run proof:training --help` for options"',
  ].join("\n");
}

export function formatTrainingProofSuccess(
  reportPath: string,
  report: TrainingProofReport,
): string {
  const verificationChecks = report.verification?.checks.length ?? 0;
  const rows = report.stages.map((stage) =>
    [
      quoteScalar(stage.stage),
      formatOptionalNumber(stage.evalLoss?.delta),
      formatOptionalNumber(stage.rewardAccuracy?.delta),
      formatOptionalNumber(stage.averageTrainingLoss),
      stage.parameterCounts?.trainable ?? "null",
      stage.memory?.peakBytes ?? "null",
    ].join(","),
  );

  return [
    "training_proof:",
    "  status: passed",
    `  source: ${quoteScalar(report.source)}`,
    `  dataset_source: ${quoteScalar(report.datasetSource)}`,
    `  report: ${quoteScalar(reportPath)}`,
    `  stages: ${report.stages.length}`,
    `  verification_checks: ${verificationChecks}`,
    `  data_notes: ${report.dataNotes.length}`,
    `stages[${rows.length}]{stage,eval_loss_delta,reward_accuracy_delta,train_loss,trainable_parameters,peak_memory_bytes}:`,
    ...rows.map((row) => `  ${row}`),
  ].join("\n");
}

export async function runTrainingProofCommand(
  argv: readonly string[],
  runtime: TrainingProofRuntime = {},
): Promise<number> {
  const stdout = runtime.stdout ?? console.log;
  const stderr = runtime.stderr ?? console.error;
  let command: ReturnType<typeof parseTrainingProofCommand>;

  try {
    command = parseTrainingProofCommand(argv);
  } catch (error) {
    stdout(formatTrainingProofError(errorMessage(error), "usage"));
    return error instanceof TrainingProofUsageError ? 2 : 1;
  }

  if (command.kind === "help") {
    stdout(formatTrainingProofUsage());
    return 0;
  }

  const acquireLock = runtime.acquireLock ?? (() => acquireRuntimeCommandLock("proof:training"));
  const runProof = runtime.runProof ?? runTrainingProof;
  let lock: RuntimeLock | undefined;
  try {
    lock = acquireLock();
    const report = await runProof(command.options, stderr);
    stdout(formatTrainingProofSuccess(command.options.reportPath, report));
    return 0;
  } catch (error) {
    stdout(formatTrainingProofError(errorMessage(error), "runtime"));
    if (error instanceof Error && error.stack !== undefined) {
      stderr(error.stack);
    }
    return 1;
  } finally {
    lock?.[Symbol.dispose]();
  }
}
