/**
 * Flags suspicious nested tensor-producing calls in runtime-sensitive code.
 *
 * This is a narrow heuristic gate aimed at the concrete leak class we hit:
 * anonymous disposable intermediates hidden inside other tensor ops.
 */

import { Glob } from "bun";
import * as ts from "typescript";

import { isTrackedTensorProducingCallName } from "./runtime-sensitive-ops";

const SOURCE_GLOBS = [
  "packages/mlx-ts/src/core/**/*.ts",
  "packages/mlx-ts/src/nn/**/*.ts",
  "packages/mlx-ts/src/optimizers/**/*.ts",
  "packages/nanogpt/src/**/*.ts",
];

const EXCLUDED_FILE_RE =
  /\.test\.tsx?$|packages\/nanogpt\/src\/run\/acceptance\.ts$|packages\/nanogpt\/src\/run\/soak\.ts$|packages\/nanogpt\/src\/bench\/memory\.ts$/;

const EXCLUDED_CALLEE_BASES = new Set([
  "Array",
  "Boolean",
  "Date",
  "JSON",
  "Math",
  "Number",
  "Object",
  "Promise",
  "String",
  "console",
]);

type Violation = {
  file: string;
  line: number;
  column: number;
  outerCall: string;
  innerCall: string;
  snippet: string;
};

function getCallName(node: ts.CallExpression): string | null {
  const expression = node.expression;
  if (ts.isIdentifier(expression)) {
    return expression.text;
  }
  if (ts.isPropertyAccessExpression(expression) || ts.isPropertyAccessChain(expression)) {
    const base = expression.expression;
    if (ts.isIdentifier(base) && EXCLUDED_CALLEE_BASES.has(base.text)) {
      return null;
    }
    return expression.name.text;
  }
  return null;
}

function getLineSnippet(sourceFile: ts.SourceFile, position: number): string {
  const lineStart =
    sourceFile.getLineStarts()[sourceFile.getLineAndCharacterOfPosition(position).line] ?? 0;
  const nextLine = sourceFile.getLineStarts().find((start) => start > lineStart);
  const lineEnd = nextLine ?? sourceFile.end;
  return sourceFile.text.slice(lineStart, lineEnd).trim();
}

function collectViolations(sourceFile: ts.SourceFile): Violation[] {
  const violations: Violation[] = [];

  function visit(node: ts.Node, trackedAncestors: string[]): void {
    if (!ts.isCallExpression(node)) {
      ts.forEachChild(node, (child) => visit(child, trackedAncestors));
      return;
    }

    const callName = getCallName(node);
    const isTrackedCall = callName !== null && isTrackedTensorProducingCallName(callName);

    if (isTrackedCall && trackedAncestors.length > 0) {
      const { line, character } = sourceFile.getLineAndCharacterOfPosition(
        node.getStart(sourceFile),
      );
      const outerCall = trackedAncestors[trackedAncestors.length - 1] ?? "unknown";
      violations.push({
        file: sourceFile.fileName,
        line: line + 1,
        column: character + 1,
        outerCall,
        innerCall: callName,
        snippet: getLineSnippet(sourceFile, node.getStart(sourceFile)),
      });
    }

    const nextTrackedAncestors = isTrackedCall ? [...trackedAncestors, callName] : trackedAncestors;
    ts.forEachChild(node, (child) => visit(child, nextTrackedAncestors));
  }

  visit(sourceFile, []);
  return violations;
}

const violations: Violation[] = [];
const seenFiles = new Set<string>();

for (const sourceGlob of SOURCE_GLOBS) {
  const glob = new Glob(sourceGlob);
  for await (const fullPath of glob.scan(".")) {
    if (seenFiles.has(fullPath) || EXCLUDED_FILE_RE.test(fullPath)) {
      continue;
    }
    seenFiles.add(fullPath);

    const content = await Bun.file(fullPath).text();
    const sourceFile = ts.createSourceFile(
      fullPath,
      content,
      ts.ScriptTarget.Latest,
      true,
      ts.ScriptKind.TS,
    );
    violations.push(...collectViolations(sourceFile));
  }
}

if (violations.length > 0) {
  console.error("Suspicious nested tensor-producing calls found:\n");
  for (const violation of violations) {
    console.error(
      `  ${violation.file}:${violation.line}:${violation.column}: ${violation.outerCall}() contains nested ${violation.innerCall}()`,
    );
    console.error(`    ${violation.snippet}`);
  }
  console.error("");
  console.error(
    "Runtime-sensitive code must keep disposable MxArray lifetimes visible. Bind the inner tensor to a local name with `using` or free it explicitly.",
  );
  process.exit(1);
}

console.log("No suspicious nested tensor-producing calls found.");
