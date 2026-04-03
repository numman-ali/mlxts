/**
 * Runs package coverage and enforces the mlx-ts quality gate.
 *
 * Bun can emit LCOV reports but does not provide a built-in threshold gate,
 * so this script runs coverage for each workspace package and verifies that
 * mlx-ts stays above the required minimums.
 */

import { spawnSync } from "node:child_process";
import { existsSync, readFileSync, rmSync } from "node:fs";
import { join } from "node:path";

type CoverageTotals = {
  functionsFound: number;
  functionsHit: number;
  linesFound: number;
  linesHit: number;
};

type FileCoverage = {
  path: string;
  functionsFound: number;
  functionsHit: number;
  linesFound: number;
  linesHit: number;
};

type PackageConfig = {
  name: string;
  cwd: string;
  coverageDir: string;
  thresholds?: {
    functions: number;
    lines: number;
  };
};

const PROJECT_ROOT = join(import.meta.dirname, "..");

const PACKAGES: PackageConfig[] = [
  {
    name: "mlx-ts",
    cwd: join(PROJECT_ROOT, "packages", "mlx-ts"),
    coverageDir: join(PROJECT_ROOT, "coverage", "mlx-ts"),
    thresholds: { lines: 95, functions: 90 },
  },
  {
    name: "nanogpt",
    cwd: join(PROJECT_ROOT, "packages", "nanogpt"),
    coverageDir: join(PROJECT_ROOT, "coverage", "nanogpt"),
  },
];

function percent(hit: number, found: number): number {
  if (found === 0) {
    return 100;
  }
  return (hit / found) * 100;
}

function emptyFileCoverage(): FileCoverage {
  return {
    path: "",
    functionsFound: 0,
    functionsHit: 0,
    linesFound: 0,
    linesHit: 0,
  };
}

function parseRecordLine(fileCoverage: FileCoverage, line: string): void {
  if (line.startsWith("SF:")) {
    fileCoverage.path = line.slice(3);
  } else if (line.startsWith("FNF:")) {
    fileCoverage.functionsFound = Number(line.slice(4));
  } else if (line.startsWith("FNH:")) {
    fileCoverage.functionsHit = Number(line.slice(4));
  } else if (line.startsWith("LF:")) {
    fileCoverage.linesFound = Number(line.slice(3));
  } else if (line.startsWith("LH:")) {
    fileCoverage.linesHit = Number(line.slice(3));
  }
}

function parseRecord(record: string): FileCoverage | null {
  const lines = record
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);

  if (lines.length === 0) {
    return null;
  }

  const fileCoverage = emptyFileCoverage();
  for (const line of lines) {
    parseRecordLine(fileCoverage, line);
  }

  return fileCoverage.path === "" ? null : fileCoverage;
}

function parseLcov(lcovPath: string): { files: FileCoverage[]; totals: CoverageTotals } {
  const content = readFileSync(lcovPath, "utf8");
  const files = content
    .split("end_of_record")
    .map((record) => parseRecord(record))
    .filter((record): record is FileCoverage => record !== null);

  const totals = files.reduce<CoverageTotals>(
    (acc, file) => ({
      functionsFound: acc.functionsFound + file.functionsFound,
      functionsHit: acc.functionsHit + file.functionsHit,
      linesFound: acc.linesFound + file.linesFound,
      linesHit: acc.linesHit + file.linesHit,
    }),
    { functionsFound: 0, functionsHit: 0, linesFound: 0, linesHit: 0 },
  );

  return { files, totals };
}

function runCoverage(pkg: PackageConfig): { files: FileCoverage[]; totals: CoverageTotals } {
  rmSync(pkg.coverageDir, { recursive: true, force: true });

  const result = spawnSync(
    "bun",
    ["test", "--coverage", "--coverage-reporter=lcov", `--coverage-dir=${pkg.coverageDir}`],
    {
      cwd: pkg.cwd,
      stdio: "inherit",
    },
  );

  if (result.status !== 0) {
    process.exit(result.status ?? 1);
  }

  const lcovPath = join(pkg.coverageDir, "lcov.info");
  if (!existsSync(lcovPath)) {
    throw new Error(`Coverage report not found for ${pkg.name}: ${lcovPath}`);
  }

  return parseLcov(lcovPath);
}

function printSummary(pkg: PackageConfig, totals: CoverageTotals): void {
  const linePercent = percent(totals.linesHit, totals.linesFound);
  const functionPercent = percent(totals.functionsHit, totals.functionsFound);
  const gateLabel =
    pkg.thresholds === undefined
      ? "report-only"
      : `gate ${pkg.thresholds.lines}% lines / ${pkg.thresholds.functions}% funcs`;

  console.log("");
  console.log(
    `${pkg.name} coverage: ${linePercent.toFixed(2)}% lines, ${functionPercent.toFixed(2)}% funcs (${gateLabel})`,
  );
}

function printWeakestFiles(files: FileCoverage[]): void {
  const weakest = [...files]
    .sort(
      (left, right) =>
        percent(left.linesHit, left.linesFound) - percent(right.linesHit, right.linesFound),
    )
    .slice(0, 5);

  if (weakest.length === 0) {
    return;
  }

  console.log("Lowest line coverage files:");
  for (const file of weakest) {
    console.log(`  ${file.path}: ${percent(file.linesHit, file.linesFound).toFixed(2)}%`);
  }
}

let failed = false;

for (const pkg of PACKAGES) {
  const { files, totals } = runCoverage(pkg);
  printSummary(pkg, totals);

  if (pkg.thresholds === undefined) {
    continue;
  }

  const linePercent = percent(totals.linesHit, totals.linesFound);
  const functionPercent = percent(totals.functionsHit, totals.functionsFound);

  if (linePercent < pkg.thresholds.lines || functionPercent < pkg.thresholds.functions) {
    failed = true;
    console.error(
      `${pkg.name} coverage gate failed: expected at least ${pkg.thresholds.lines}% lines and ${pkg.thresholds.functions}% funcs.`,
    );
    printWeakestFiles(files);
  }
}

if (failed) {
  process.exit(1);
}

console.log("");
console.log("Coverage thresholds satisfied.");
