#!/usr/bin/env bun

import type { TrainingProofVerification } from "./types";
import { assertTrainingProofReport, parseTrainingProofReport } from "./verification";

type VerifyReportCommand = { kind: "help" } | { kind: "verify"; reportPath: string };

type VerifyReportRuntime = {
  stdout?: (text: string) => void;
  readText?: (path: string) => Promise<string>;
};

class UsageError extends Error {}

export function formatVerifyReportUsage(): string {
  return [
    "Usage: bun run examples/train-proof/verify-report.ts <report.json>",
    "",
    "Verifies a training proof JSON report without rerunning training.",
    "",
    "Exit codes:",
    "  0  report is valid",
    "  1  report cannot be read, parsed, or verified",
    "  2  usage error",
    "",
    "Examples:",
    "  bun run examples/train-proof/verify-report.ts .tmp/training-proof/meta-llama-Llama-3.2-1B-Instruct-report.json",
  ].join("\n");
}

export function parseVerifyReportArgs(argv: readonly string[]): VerifyReportCommand {
  if (argv.length === 1 && (argv[0] === "--help" || argv[0] === "-h")) {
    return { kind: "help" };
  }
  if (argv.length === 0) {
    throw new UsageError("report path is required.");
  }
  if (argv.length > 1) {
    throw new UsageError("expected exactly one report path.");
  }
  const reportPath = argv[0];
  if (reportPath === undefined || reportPath.trim() === "") {
    throw new UsageError("report path is required.");
  }
  if (reportPath.startsWith("-")) {
    throw new UsageError(`unknown option "${reportPath}".`);
  }
  return { kind: "verify", reportPath };
}

export function formatVerifyReportSuccess(
  reportPath: string,
  verification: TrainingProofVerification,
): string {
  return [
    "training_proof_report:",
    "  status: passed",
    `  report: ${reportPath}`,
    "  failed_checks: 0",
    `  passed_checks: ${verification.checks.length}`,
  ].join("\n");
}

function formatMultilineField(name: string, value: string): string[] {
  const lines = value.split(/\r?\n/);
  if (lines.length === 1) {
    return [`  ${name}: ${value}`];
  }
  return [`  ${name}: |`, ...lines.map((line) => `    ${line}`)];
}

export function formatVerifyReportError(message: string, help: string): string {
  return ["error:", ...formatMultilineField("message", message), `help: ${help}`].join("\n");
}

async function defaultReadText(path: string): Promise<string> {
  return await Bun.file(path).text();
}

export async function runVerifyReportCommand(
  argv: readonly string[],
  runtime: VerifyReportRuntime = {},
): Promise<number> {
  const stdout = runtime.stdout ?? console.log;
  const readText = runtime.readText ?? defaultReadText;
  let command: VerifyReportCommand;

  try {
    command = parseVerifyReportArgs(argv);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    stdout(
      formatVerifyReportError(
        message,
        "bun run examples/train-proof/verify-report.ts <report.json>",
      ),
    );
    return error instanceof UsageError ? 2 : 1;
  }

  if (command.kind === "help") {
    stdout(formatVerifyReportUsage());
    return 0;
  }

  try {
    const parsed: unknown = JSON.parse(await readText(command.reportPath));
    const report = parseTrainingProofReport(parsed);
    const verification = assertTrainingProofReport(report);
    stdout(formatVerifyReportSuccess(command.reportPath, verification));
    return 0;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    stdout(
      formatVerifyReportError(
        message,
        "inspect the report JSON or rerun `bun run proof:training` to regenerate it",
      ),
    );
    return 1;
  }
}

if (import.meta.main) {
  const exitCode = await runVerifyReportCommand(Bun.argv.slice(2));
  process.exit(exitCode);
}
