/**
 * Runs package coverage and enforces the repo quality gates.
 *
 * Bun now supports built-in coverage thresholds, but this repo keeps a custom
 * package-aware gate so we can enforce different thresholds per workspace and
 * print the weakest source files directly. Long soak and acceptance runs remain
 * separate from this fast gate.
 */

import { existsSync, readFileSync, rmSync } from "fs";
import { join } from "path";

type CoverageTotals = {
  functionsFound: number;
  functionsHit: number;
  branchesFound: number;
  branchesHit: number;
  linesFound: number;
  linesHit: number;
};

type FileCoverage = {
  path: string;
  functionsFound: number;
  functionsHit: number;
  branchesFound: number;
  branchesHit: number;
  linesFound: number;
  linesHit: number;
};

type PackageConfig = {
  name: string;
  cwd: string;
  coverageDir: string;
  thresholds?: {
    functions: number;
    branches?: number;
    lines: number;
  };
};

const PROJECT_ROOT = join(import.meta.dirname, "..");

const PACKAGES: PackageConfig[] = [
  {
    name: "mlx-ts",
    cwd: join(PROJECT_ROOT, "packages", "mlx-ts"),
    coverageDir: join(PROJECT_ROOT, "coverage", "mlx-ts"),
    thresholds: { lines: 95, functions: 90, branches: 85 },
  },
  {
    name: "nanogpt",
    cwd: join(PROJECT_ROOT, "packages", "nanogpt"),
    coverageDir: join(PROJECT_ROOT, "coverage", "nanogpt"),
    thresholds: { lines: 90, functions: 85, branches: 85 },
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
    branchesFound: 0,
    branchesHit: 0,
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
  } else if (line.startsWith("BRF:")) {
    fileCoverage.branchesFound = Number(line.slice(4));
  } else if (line.startsWith("BRH:")) {
    fileCoverage.branchesHit = Number(line.slice(4));
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
      branchesFound: acc.branchesFound + file.branchesFound,
      branchesHit: acc.branchesHit + file.branchesHit,
      functionsFound: acc.functionsFound + file.functionsFound,
      functionsHit: acc.functionsHit + file.functionsHit,
      linesFound: acc.linesFound + file.linesFound,
      linesHit: acc.linesHit + file.linesHit,
    }),
    {
      branchesFound: 0,
      branchesHit: 0,
      functionsFound: 0,
      functionsHit: 0,
      linesFound: 0,
      linesHit: 0,
    },
  );

  return { files, totals };
}

function isPackageSourceFile(pkg: PackageConfig, filePath: string): boolean {
  const normalized = filePath.replaceAll("\\", "/");
  const isSourcePath =
    normalized.startsWith("src/") || normalized.includes(`/packages/${pkg.name}/src/`);
  if (!isSourcePath) {
    return false;
  }
  return !normalized.endsWith(".test.ts");
}

function filterPackageFiles(
  pkg: PackageConfig,
  files: FileCoverage[],
): { files: FileCoverage[]; totals: CoverageTotals } {
  const packageFiles = files.filter((file) => isPackageSourceFile(pkg, file.path));
  const totals = packageFiles.reduce<CoverageTotals>(
    (acc, file) => ({
      branchesFound: acc.branchesFound + file.branchesFound,
      branchesHit: acc.branchesHit + file.branchesHit,
      functionsFound: acc.functionsFound + file.functionsFound,
      functionsHit: acc.functionsHit + file.functionsHit,
      linesFound: acc.linesFound + file.linesFound,
      linesHit: acc.linesHit + file.linesHit,
    }),
    {
      branchesFound: 0,
      branchesHit: 0,
      functionsFound: 0,
      functionsHit: 0,
      linesFound: 0,
      linesHit: 0,
    },
  );

  return { files: packageFiles, totals };
}

function runCoverage(pkg: PackageConfig): { files: FileCoverage[]; totals: CoverageTotals } {
  rmSync(pkg.coverageDir, { recursive: true, force: true });

  const result = Bun.spawnSync(
    ["bun", "test", "--coverage", "--coverage-reporter=lcov", `--coverage-dir=${pkg.coverageDir}`],
    {
      cwd: pkg.cwd,
      stdout: "inherit",
      stderr: "inherit",
    },
  );

  if (result.exitCode !== 0) {
    process.exit(result.exitCode);
  }

  const lcovPath = join(pkg.coverageDir, "lcov.info");
  if (!existsSync(lcovPath)) {
    throw new Error(`Coverage report not found for ${pkg.name}: ${lcovPath}`);
  }

  const parsed = parseLcov(lcovPath);
  return filterPackageFiles(pkg, parsed.files);
}

function printSummary(pkg: PackageConfig, totals: CoverageTotals): void {
  const linePercent = percent(totals.linesHit, totals.linesFound);
  const functionPercent = percent(totals.functionsHit, totals.functionsFound);
  const branchPercent = percent(totals.branchesHit, totals.branchesFound);
  const hasBranchData = totals.branchesFound > 0 || totals.branchesHit > 0;
  const gateLabel =
    pkg.thresholds === undefined
      ? "report-only"
      : hasBranchData && pkg.thresholds.branches !== undefined
        ? `gate ${pkg.thresholds.lines}% lines / ${pkg.thresholds.functions}% funcs / ${pkg.thresholds.branches}% branches`
        : `gate ${pkg.thresholds.lines}% lines / ${pkg.thresholds.functions}% funcs (branch data unavailable)`;

  console.log("");
  console.log(
    `${pkg.name} coverage: ${linePercent.toFixed(2)}% lines, ${functionPercent.toFixed(2)}% funcs (${gateLabel})`,
  );
  if (hasBranchData) {
    console.log(`Branch coverage: ${branchPercent.toFixed(2)}%`);
  }
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
  const branchPercent = percent(totals.branchesHit, totals.branchesFound);
  const hasBranchData = totals.branchesFound > 0 || totals.branchesHit > 0;

  if (
    linePercent < pkg.thresholds.lines ||
    functionPercent < pkg.thresholds.functions ||
    (hasBranchData &&
      pkg.thresholds.branches !== undefined &&
      branchPercent < pkg.thresholds.branches)
  ) {
    failed = true;
    console.error(
      `${pkg.name} coverage gate failed: expected at least ${pkg.thresholds.lines}% lines and ${pkg.thresholds.functions}% funcs${
        hasBranchData && pkg.thresholds.branches !== undefined
          ? ` and ${pkg.thresholds.branches}% branches`
          : ""
      }.`,
    );
    printWeakestFiles(files);
  } else if (pkg.thresholds.branches !== undefined && !hasBranchData) {
    console.log(
      `${pkg.name} coverage: branch data unavailable in LCOV output, so the branch threshold was not enforced.`,
    );
  }
}

if (failed) {
  process.exit(1);
}

console.log("");
console.log("Coverage thresholds satisfied.");
