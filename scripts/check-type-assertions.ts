/**
 * Checks for type assertions (`as SomeType`, `as unknown`) in production code.
 *
 * Biome has no rule to ban `as` type assertions, so this script fills the gap.
 * It scans TypeScript source files and flags any `as` type assertions outside
 * of allowed locations (FFI boundary, test files).
 *
 * Allowed patterns:
 * - `as const` and `as const satisfies` (TypeScript const assertions)
 * - Import/export aliases (`import { x as y }`, `export * as ns`)
 * - Lines inside comments (// or /* *\/)
 * - Files: ffi.ts (FFI boundary), *.test.ts (tests)
 *
 * Exit code 0 = clean, 1 = violations found.
 */

import { Glob } from "bun";

const SCAN_DIR = "packages/mlx-ts/src";
const EXCLUDED_FILES = /\/(ffi\.ts|.*\.test\.ts)$/;

// Match ` as <type>` but not `as const`, import/export aliases, or comments
const TYPE_ASSERTION_RE = /\bas\s+(?!const\b)[A-Za-z]/;
const COMMENT_LINE_RE = /^\s*(\/\/|\/\*|\*)/;
const IMPORT_EXPORT_RE = /^\s*(import|export)\b/;

const glob = new Glob("**/*.ts");
const violations: string[] = [];

for await (const path of glob.scan(SCAN_DIR)) {
  const fullPath = `${SCAN_DIR}/${path}`;

  if (EXCLUDED_FILES.test(fullPath)) continue;

  const content = await Bun.file(fullPath).text();
  const lines = content.split("\n");

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i] ?? "";

    // Skip comment lines
    if (COMMENT_LINE_RE.test(line)) continue;

    // Skip import/export lines (aliases, not assertions)
    if (IMPORT_EXPORT_RE.test(line)) continue;

    // Check for type assertions
    if (TYPE_ASSERTION_RE.test(line)) {
      violations.push(`${fullPath}:${i + 1}: ${line.trim()}`);
    }
  }
}

if (violations.length > 0) {
  console.error("Type assertions found in production code:\n");
  for (const v of violations) {
    console.error(`  ${v}`);
  }
  console.error("\nType assertions (as SomeType) are not allowed outside ffi.ts.");
  console.error("Use runtime checks, type narrowing, or improve the type design instead.");
  process.exit(1);
} else {
  console.log("No type assertions in production code.");
}
