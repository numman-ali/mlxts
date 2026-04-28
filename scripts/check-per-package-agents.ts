/**
 * Requires package-local AGENTS.md files for non-trivial workspace packages.
 */

import { existsSync, readdirSync, readFileSync, statSync } from "fs";
import { join } from "path";

const PROJECT_ROOT = join(import.meta.dirname, "..");
const PACKAGES_ROOT = join(PROJECT_ROOT, "packages");
const SOURCE_FILE_THRESHOLD = 5;
const SOURCE_LINE_THRESHOLD = 300;

type PackageSourceStats = {
  packageDir: string;
  sourceFiles: number;
  sourceLines: number;
};

function packageDirectories(): string[] {
  return readdirSync(PACKAGES_ROOT)
    .filter((entry) => {
      const packageRoot = join(PACKAGES_ROOT, entry);
      return statSync(packageRoot).isDirectory() && existsSync(join(packageRoot, "package.json"));
    })
    .sort((left, right) => left.localeCompare(right));
}

async function sourceFilesForPackage(packageDir: string): Promise<string[]> {
  const sourceRoot = join(PACKAGES_ROOT, packageDir, "src");
  if (!existsSync(sourceRoot)) {
    return [];
  }

  const files: string[] = [];
  const glob = new Bun.Glob("src/**/*.ts");
  for await (const path of glob.scan({ cwd: join(PACKAGES_ROOT, packageDir) })) {
    if (!path.endsWith(".test.ts")) {
      files.push(path);
    }
  }
  return files.sort((left, right) => left.localeCompare(right));
}

function countSourceLines(packageDir: string, files: readonly string[]): number {
  let lines = 0;
  for (const file of files) {
    const content = readFileSync(join(PACKAGES_ROOT, packageDir, file), "utf8");
    lines += content.split(/\r?\n/).length;
  }
  return lines;
}

function requiresAgent(stats: PackageSourceStats): boolean {
  return stats.sourceFiles > SOURCE_FILE_THRESHOLD || stats.sourceLines > SOURCE_LINE_THRESHOLD;
}

const stats: PackageSourceStats[] = [];
for (const packageDir of packageDirectories()) {
  const files = await sourceFilesForPackage(packageDir);
  stats.push({
    packageDir,
    sourceFiles: files.length,
    sourceLines: countSourceLines(packageDir, files),
  });
}

const missingAgents = stats.filter(
  (entry) =>
    requiresAgent(entry) && !existsSync(join(PACKAGES_ROOT, entry.packageDir, "AGENTS.md")),
);

if (missingAgents.length > 0) {
  console.error("Non-trivial packages must have package-local AGENTS.md files.\n");
  for (const entry of missingAgents) {
    console.error(
      `  packages/${entry.packageDir}: ${entry.sourceFiles} src files, ${entry.sourceLines} LOC`,
    );
  }
  console.error("");
  console.error(
    `Threshold: >${SOURCE_FILE_THRESHOLD} production src files or >${SOURCE_LINE_THRESHOLD} production LOC.`,
  );
  process.exit(1);
}

const checkedCount = stats.filter(requiresAgent).length;
console.log(
  `Checked ${checkedCount} non-trivial packages. All have package-local AGENTS.md files.`,
);
