#!/usr/bin/env bun

import type { CausalLMAdapterFormat } from "@mlxts/transformers";

import type { FinetuneMode } from "./args";
import {
  assertFinetuneReport,
  type FinetuneReportVerification,
  type FinetuneReportVerificationOptions,
  parseFinetuneReport,
} from "./verification";

type VerifyCommand = {
  kind: "verify";
  reportPath: string;
  options: FinetuneReportVerificationOptions;
};

type VerifyReportCommand = { kind: "help" } | VerifyCommand;

type VerifyReportRuntime = {
  stdout?: (text: string) => void;
  readText?: (path: string) => Promise<string>;
};

class VerifyUsageError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "VerifyUsageError";
  }
}

function quoteScalar(value: string | number | boolean | null): string {
  return typeof value === "string" ? JSON.stringify(value) : String(value);
}

function formatNumber(value: number): string {
  return String(Math.round(value * 10000) / 10000);
}

function readValue(flag: string, value: string | undefined): string {
  if (value === undefined || value.trim() === "" || value.startsWith("--")) {
    throw new VerifyUsageError(`lora-finetune verify-report: ${flag} expects a value.`);
  }
  return value;
}

function readMode(value: string): FinetuneMode {
  if (value === "lora" || value === "qlora") {
    return value;
  }
  throw new VerifyUsageError(`lora-finetune verify-report: unknown mode "${value}".`);
}

function readAdapterFormat(value: string): CausalLMAdapterFormat {
  if (value === "mlxts" || value === "peft") {
    return value;
  }
  throw new VerifyUsageError(`lora-finetune verify-report: unknown adapter format "${value}".`);
}

function readReportPathArgument(arg: string, current: string | null): string {
  if (arg.startsWith("--")) {
    throw new VerifyUsageError(`lora-finetune verify-report: unknown option "${arg}".`);
  }
  if (current !== null) {
    throw new VerifyUsageError("lora-finetune verify-report: expected one report path.");
  }
  return arg;
}

export function formatLoRAFinetuneVerifyUsage(): string {
  return [
    "description: Verify a LoRA finetune JSON report without rerunning training",
    "usage[3]:",
    "  bun run examples/lora-finetune/verify-report.ts <report.json>",
    "  bun run examples/lora-finetune/verify-report.ts <report.json> --mode qlora",
    "  bun run examples/lora-finetune/verify-report.ts <report.json> --adapter-format peft",
    "options[5]{flag,description}:",
    '  "--mode lora|qlora","Require the report mode"',
    '  "--adapter-format mlxts|peft","Require the saved adapter format"',
    '  "--require-loss-not-worse","Require held-out loss after training to be no worse"',
    '  "--help","Show this help"',
    '  "<report.json>","Report emitted by examples/lora-finetune/index.ts"',
    "exit_codes[3]{code,meaning}:",
    '  0,"report verified or help"',
    '  1,"report read, parse, or verification failure"',
    '  2,"usage error"',
  ].join("\n");
}

export function parseLoRAFinetuneVerifyArgs(argv: readonly string[]): VerifyReportCommand {
  if (argv.length === 1 && (argv[0] === "--help" || argv[0] === "-h")) {
    return { kind: "help" };
  }

  let reportPath: string | null = null;
  let expectedMode: FinetuneMode | undefined;
  let expectedAdapterFormat: CausalLMAdapterFormat | undefined;
  let requireLossNotWorse = false;

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === undefined) {
      continue;
    }

    switch (arg) {
      case "--mode":
        expectedMode = readMode(readValue(arg, argv[index + 1]));
        index += 1;
        break;
      case "--adapter-format":
        expectedAdapterFormat = readAdapterFormat(readValue(arg, argv[index + 1]));
        index += 1;
        break;
      case "--require-loss-not-worse":
        requireLossNotWorse = true;
        break;
      case "--help":
      case "-h":
        throw new VerifyUsageError("help must be the only argument.");
      default:
        reportPath = readReportPathArgument(arg, reportPath);
        break;
    }
  }

  if (reportPath === null || reportPath.trim() === "") {
    throw new VerifyUsageError("lora-finetune verify-report: report path is required.");
  }

  const options: FinetuneReportVerificationOptions = { requireLossNotWorse };
  if (expectedMode !== undefined) {
    options.expectedMode = expectedMode;
  }
  if (expectedAdapterFormat !== undefined) {
    options.expectedAdapterFormat = expectedAdapterFormat;
  }

  return {
    kind: "verify",
    reportPath,
    options,
  };
}

export function formatLoRAFinetuneVerifySuccess(
  reportPath: string,
  report: ReturnType<typeof parseFinetuneReport>,
  verification: FinetuneReportVerification,
): string {
  const lossDelta = report.metrics.evalLossAfter - report.metrics.evalLossBefore;
  return [
    "lora_finetune_report:",
    "  status: passed",
    `  report: ${quoteScalar(reportPath)}`,
    "  failed_checks: 0",
    `  passed_checks: ${verification.checks.length}`,
    `  source: ${quoteScalar(report.source)}`,
    `  mode: ${quoteScalar(report.mode)}`,
    `  adapter_format: ${quoteScalar(report.adapterFormat)}`,
    `  target_count: ${report.metrics.targetCount}`,
    `  trainable_parameters: ${report.parameterCounts.trainable}`,
    `  total_parameters: ${report.parameterCounts.total}`,
    `  peak_memory_bytes: ${report.memory.peakBytes}`,
    `  eval_loss_delta: ${formatNumber(lossDelta)}`,
  ].join("\n");
}

function formatBlockField(name: string, value: string): string[] {
  const lines = value.split(/\r?\n/);
  return [`  ${name}: |`, ...lines.map((line) => `    ${line}`)];
}

export function formatLoRAFinetuneVerifyError(message: string, code: "usage" | "runtime"): string {
  return [
    "error:",
    `  code: ${quoteScalar(code)}`,
    ...formatBlockField("message", message),
    "help[1]:",
    '  "Run `bun run examples/lora-finetune/verify-report.ts --help` for options"',
  ].join("\n");
}

async function defaultReadText(path: string): Promise<string> {
  return await Bun.file(path).text();
}

export async function runLoRAFinetuneVerifyCommand(
  argv: readonly string[],
  runtime: VerifyReportRuntime = {},
): Promise<number> {
  const stdout = runtime.stdout ?? console.log;
  const readText = runtime.readText ?? defaultReadText;
  let command: VerifyReportCommand;

  try {
    command = parseLoRAFinetuneVerifyArgs(argv);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    stdout(formatLoRAFinetuneVerifyError(message, "usage"));
    return error instanceof VerifyUsageError ? 2 : 1;
  }

  if (command.kind === "help") {
    stdout(formatLoRAFinetuneVerifyUsage());
    return 0;
  }

  try {
    const parsed: unknown = JSON.parse(await readText(command.reportPath));
    const report = parseFinetuneReport(parsed);
    const verification = assertFinetuneReport(report, command.options);
    stdout(formatLoRAFinetuneVerifySuccess(command.reportPath, report, verification));
    return 0;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    stdout(formatLoRAFinetuneVerifyError(message, "runtime"));
    return 1;
  }
}

if (import.meta.main) {
  const exitCode = await runLoRAFinetuneVerifyCommand(Bun.argv.slice(2));
  process.exit(exitCode);
}
