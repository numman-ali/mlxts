/**
 * Enforces a maximum physical line count for canonical production source files.
 *
 * The current Phase 5 posture is package-first. The extracted `@mlxts/*`
 * packages are canonical and stay under the hard 500-line cap. Temporary
 * migration surfaces such as `packages/mlx-ts` and `packages/nanogpt` are
 * excluded until they are deleted or rewritten.
 */

import { readFileSync } from "fs";
import { join } from "path";

const PROJECT_ROOT = join(import.meta.dirname, "..");
const MAX_FILE_LINES = 500;
const INCLUDED_GLOBS = ["packages/*/src/**/*.ts"];
const EXCLUDED_PREFIXES = ["packages/mlx-ts/", "packages/nanogpt/"];

function isCheckedFile(path: string): boolean {
  const normalized = path.replaceAll("\\", "/");
  if (normalized.endsWith(".test.ts")) {
    return false;
  }
  if (EXCLUDED_PREFIXES.some((prefix) => normalized.startsWith(prefix))) {
    return false;
  }
  return true;
}

function countPhysicalLines(path: string): number {
  const content = readFileSync(join(PROJECT_ROOT, path), "utf8");
  return content.split(/\r?\n/).length;
}

const files: string[] = [];
for (const pattern of INCLUDED_GLOBS) {
  const glob = new Bun.Glob(pattern);
  for await (const path of glob.scan({ cwd: PROJECT_ROOT })) {
    if (isCheckedFile(path)) {
      files.push(path);
    }
  }
}

const offenders = files
  .map((path) => ({ path, lines: countPhysicalLines(path) }))
  .filter((entry) => entry.lines > MAX_FILE_LINES)
  .sort((left, right) => right.lines - left.lines);

if (offenders.length > 0) {
  console.error(`Production source files must stay at or under ${MAX_FILE_LINES} lines.\n`);
  for (const offender of offenders) {
    console.error(`  ${offender.path}: ${offender.lines} lines`);
  }
  process.exit(1);
}

console.log(
  `Checked ${files.length} production source files. All canonical package files are <= ${MAX_FILE_LINES} lines.`,
);
