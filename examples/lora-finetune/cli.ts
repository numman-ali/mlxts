import { acquireRuntimeCommandLock } from "../../scripts/runtime-command-lock";
import {
  type FinetuneArgs,
  type FinetuneReport,
  FinetuneUsageError,
  parseFinetuneCommand,
} from "./args";
import { runLoRAFinetune } from "./workflow";

type RuntimeLock = {
  [Symbol.dispose](): void;
};

type FinetuneRuntime = {
  stdout?: (text: string) => void;
  stderr?: (text: string) => void;
  acquireLock?: () => RuntimeLock;
  runFinetune?: (
    options: FinetuneArgs,
    progress: (line: string) => void,
  ) => Promise<FinetuneReport>;
};

function quoteScalar(value: string | number | boolean | null): string {
  return typeof value === "string" ? JSON.stringify(value) : String(value);
}

function formatNumber(value: number): string {
  return String(Math.round(value * 10000) / 10000);
}

export function formatLoRAFinetuneUsage(): string {
  return [
    "description: Run the readable Phase 8 LoRA and QLoRA fine-tuning example",
    "usage[3]:",
    "  bun run examples/lora-finetune/index.ts",
    "  bun run examples/lora-finetune/index.ts --dataset-source tiny --train-limit 8 --eval-limit 4 --steps 2",
    "  bun run examples/lora-finetune/index.ts --mode qlora --preset all-linear",
    "options[16]{flag,description}:",
    '  "--source <id>","Dense checkpoint source; default meta-llama/Llama-3.2-1B-Instruct"',
    '  "--mode lora|qlora","Adapter training mode; default lora"',
    '  "--preset attention|attention+mlp|all-linear","LoRA target preset; default attention or all-linear for qlora"',
    '  "--adapter-format mlxts|peft","Adapter output format; default mlxts"',
    '  "--dataset-source tiny|huggingface|jsonl","Dataset source; default huggingface"',
    '  "--dataset-jsonl <path>","JSONL dataset path required for --dataset-source jsonl"',
    '  "--train-limit <n>","Training examples; default 64"',
    '  "--eval-limit <n>","Held-out examples; default 16"',
    '  "--batch-size <n>","Batch size; default 4"',
    '  "--steps <n>","Optimizer steps; default 8"',
    '  "--max-seq-len <n>","Token cap per example; default 1024"',
    '  "--seed <n>","Deterministic batching seed; default 7"',
    '  "--output-dir <dir>","Adapter and example artifact directory"',
    '  "--quantized-output <dir>","4-bit snapshot output directory for qlora"',
    '  "--report <path>","LoRA finetune JSON report path"',
    '  "--help","Show this help"',
    "exit_codes[3]{code,meaning}:",
    '  0,"finetune completed or help"',
    '  1,"runtime or training failure"',
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

export function formatLoRAFinetuneError(message: string, code: "usage" | "runtime"): string {
  return [
    "error:",
    `  code: ${quoteScalar(code)}`,
    ...formatBlockField("message", message),
    "help[1]:",
    '  "Run `bun run examples/lora-finetune/index.ts --help` for options"',
  ].join("\n");
}

export function formatLoRAFinetuneSuccess(reportPath: string, report: FinetuneReport): string {
  const lossDelta = report.metrics.evalLossAfter - report.metrics.evalLossBefore;
  return [
    "lora_finetune:",
    "  status: passed",
    `  source: ${quoteScalar(report.source)}`,
    `  mode: ${quoteScalar(report.mode)}`,
    `  preset: ${quoteScalar(report.preset)}`,
    `  adapter_format: ${quoteScalar(report.adapterFormat)}`,
    `  dataset_source: ${quoteScalar(report.datasetSource)}`,
    `  report: ${quoteScalar(reportPath)}`,
    `  adapter_dir: ${quoteScalar(report.adapterDir)}`,
    `  target_count: ${report.metrics.targetCount}`,
    `  trainable_parameters: ${report.parameterCounts.trainable}`,
    `  total_parameters: ${report.parameterCounts.total}`,
    `  peak_memory_bytes: ${report.memory.peakBytes}`,
    `  eval_loss_before: ${formatNumber(report.metrics.evalLossBefore)}`,
    `  eval_loss_after: ${formatNumber(report.metrics.evalLossAfter)}`,
    `  eval_loss_delta: ${formatNumber(lossDelta)}`,
    `  average_training_loss: ${formatNumber(report.metrics.averageTrainingLoss)}`,
  ].join("\n");
}

export async function runLoRAFinetuneCommand(
  argv: readonly string[],
  runtime: FinetuneRuntime = {},
): Promise<number> {
  const stdout = runtime.stdout ?? console.log;
  const stderr = runtime.stderr ?? console.error;
  let command: ReturnType<typeof parseFinetuneCommand>;

  try {
    command = parseFinetuneCommand(argv);
  } catch (error) {
    stdout(formatLoRAFinetuneError(errorMessage(error), "usage"));
    return error instanceof FinetuneUsageError ? 2 : 1;
  }

  if (command.kind === "help") {
    stdout(formatLoRAFinetuneUsage());
    return 0;
  }

  const acquireLock =
    runtime.acquireLock ?? (() => acquireRuntimeCommandLock("example:lora-finetune"));
  const runFinetune = runtime.runFinetune ?? runLoRAFinetune;
  let lock: RuntimeLock | undefined;
  try {
    lock = acquireLock();
    const report = await runFinetune(command.options, stderr);
    stdout(formatLoRAFinetuneSuccess(command.options.reportPath, report));
    return 0;
  } catch (error) {
    stdout(formatLoRAFinetuneError(errorMessage(error), "runtime"));
    if (error instanceof Error && error.stack !== undefined) {
      stderr(error.stack);
    }
    return 1;
  } finally {
    lock?.[Symbol.dispose]();
  }
}
