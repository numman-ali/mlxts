/**
 * Installs git hooks from .hooks/ into .git/hooks/
 *
 * Runs automatically via `bun install` (package.json "prepare" script).
 * No external dependencies — just copies and chmod.
 */

import { chmodSync, copyFileSync, existsSync, mkdirSync } from "node:fs";
import { join } from "node:path";

const PROJECT_ROOT = join(import.meta.dirname, "..");
const HOOKS_SOURCE = join(PROJECT_ROOT, ".hooks");
const HOOKS_TARGET = join(PROJECT_ROOT, ".git", "hooks");

const HOOKS_TO_INSTALL = ["pre-commit"] as const;

if (!existsSync(join(PROJECT_ROOT, ".git"))) {
  console.log("No .git directory found — skipping hook installation.");
  process.exit(0);
}

if (!existsSync(HOOKS_TARGET)) {
  mkdirSync(HOOKS_TARGET, { recursive: true });
}

for (const hook of HOOKS_TO_INSTALL) {
  const source = join(HOOKS_SOURCE, hook);
  const target = join(HOOKS_TARGET, hook);

  if (!existsSync(source)) {
    console.warn(`Warning: hook source not found: ${source}`);
    continue;
  }

  copyFileSync(source, target);
  chmodSync(target, 0o755);
  console.log(`Installed git hook: ${hook}`);
}

console.log("Git hooks installed successfully.");
