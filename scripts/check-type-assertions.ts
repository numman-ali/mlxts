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
 * - Files: any package `src/core/ffi/` boundary and `*.test.ts[x]` test files
 *
 * Exit code 0 = clean, 1 = violations found.
 */

import { Glob } from "bun";
import ts from "typescript";

const SOURCE_GLOBS = ["packages/*/src/**/*.ts", "packages/*/src/**/*.tsx"];
const EXCLUDED_FILES = /packages\/[^/]+\/src\/core\/ffi\/.*\.tsx?$|.*\.test\.tsx?$/;

const violations: string[] = [];
const seenFiles = new Set<string>();

function isConstAssertion(node: ts.AsExpression, sourceFile: ts.SourceFile): boolean {
  return node.type.getText(sourceFile) === "const";
}

function collectTypeAssertions(sourceFile: ts.SourceFile, fullPath: string): void {
  function visit(node: ts.Node): void {
    if (ts.isAsExpression(node) && !isConstAssertion(node, sourceFile)) {
      const position = sourceFile.getLineAndCharacterOfPosition(node.getStart(sourceFile));
      violations.push(
        `${fullPath}:${position.line + 1}:${position.character + 1}: ${node.getText(sourceFile)}`,
      );
    }

    ts.forEachChild(node, visit);
  }

  visit(sourceFile);
}

for (const sourceGlob of SOURCE_GLOBS) {
  const glob = new Glob(sourceGlob);
  for await (const fullPath of glob.scan(".")) {
    if (seenFiles.has(fullPath) || EXCLUDED_FILES.test(fullPath)) continue;
    seenFiles.add(fullPath);

    const content = await Bun.file(fullPath).text();
    const sourceFile = ts.createSourceFile(
      fullPath,
      content,
      ts.ScriptTarget.Latest,
      true,
      fullPath.endsWith(".tsx") ? ts.ScriptKind.TSX : ts.ScriptKind.TS,
    );

    collectTypeAssertions(sourceFile, fullPath);
  }
}

if (violations.length > 0) {
  console.error("Type assertions found in production code:\n");
  for (const v of violations) {
    console.error(`  ${v}`);
  }
  console.error(
    "\nType assertions (as SomeType) are not allowed outside packages/*/src/core/ffi/.",
  );
  console.error("Use runtime checks, type narrowing, or improve the type design instead.");
  process.exit(1);
} else {
  console.log("No type assertions in production code.");
}
