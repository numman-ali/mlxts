#!/usr/bin/env bun

import { assertTrainingProofReport, parseTrainingProofReport } from "./verification";

function readReportPath(argv: readonly string[]): string {
  const path = argv[0];
  if (path === undefined || path.trim() === "") {
    throw new Error("Usage: bun run examples/train-proof/verify-report.ts <report.json>");
  }
  return path;
}

const reportPath = readReportPath(Bun.argv.slice(2));
const parsed: unknown = JSON.parse(await Bun.file(reportPath).text());
const report = parseTrainingProofReport(parsed);
const verification = assertTrainingProofReport(report);

console.log(`Training proof report verified: ${reportPath}`);
console.log(`Checks passed: ${verification.checks.length}`);
