/**
 * Requires a runtime review artifact when runtime-sensitive production files change.
 *
 * The repo's runtime safety posture is that changes to hot paths and long-run
 * operator surfaces are not review-ready until they leave behind a review
 * record under docs/reviews/.
 */

import { existsSync, readFileSync } from "fs";
import { join, relative, resolve } from "path";

const PROJECT_ROOT = join(import.meta.dirname, "..");
const REVIEW_DIRECTORY = "docs/reviews/";
const REVIEW_TEMPLATE = "docs/reviews/_template.md";
const REVIEW_README = "docs/reviews/README.md";

const REQUIRED_HEADINGS = [
  "## Summary",
  "## Files Reviewed",
  "## Tensor Lifetime Audit",
  "## Memory / Performance Evidence",
  "## Independent Review",
  "## Remaining Risks / Follow-ups",
] as const;

const GENERATION_HOT_PATH_PREFIXES = [
  "packages/transformers/src/families/",
  "packages/transformers/src/infrastructure/",
] as const;

const GENERATION_HOT_PATH_FILES = new Set([
  "packages/transformers/src/generation.ts",
  "packages/nn/src/grouped-query-attention.ts",
  "packages/core/src/fast.ts",
]);

function runGit(args: string[]): string {
  const result = Bun.spawnSync(["git", ...args], {
    cwd: PROJECT_ROOT,
    stdout: "pipe",
    stderr: "pipe",
  });

  if (result.exitCode !== 0) {
    const stderr = new TextDecoder().decode(result.stderr).trim();
    throw new Error(stderr === "" ? `git ${args.join(" ")} failed` : stderr);
  }

  return new TextDecoder().decode(result.stdout).trim();
}

function listChangedTrackedFiles(): string[] {
  const output = runGit(["diff", "--name-only", "--diff-filter=ACMRD", "HEAD", "--"]);
  return output === "" ? [] : output.split("\n").filter(Boolean);
}

function listUntrackedFiles(): string[] {
  const output = runGit(["ls-files", "--others", "--exclude-standard"]);
  return output === "" ? [] : output.split("\n").filter(Boolean);
}

function uniqueSorted(paths: string[]): string[] {
  return [...new Set(paths)].sort((left, right) => left.localeCompare(right));
}

function isProductionSourceFile(path: string): boolean {
  return (
    (path.startsWith("packages/") || path.startsWith("examples/nanogpt/")) &&
    path.includes("/src/") &&
    !path.endsWith(".test.ts")
  );
}

function isRuntimeSensitiveFile(path: string): boolean {
  if (!isProductionSourceFile(path)) {
    return false;
  }

  return (
    path.startsWith("packages/core/src/") ||
    path.startsWith("packages/nn/src/") ||
    path.startsWith("packages/optimizers/src/") ||
    path.startsWith("packages/train/src/") ||
    path.startsWith("packages/data/src/") ||
    path.startsWith("packages/tokenizers/src/") ||
    path.startsWith("packages/transformers/src/") ||
    path.startsWith("packages/serve/src/") ||
    path.startsWith("examples/nanogpt/src/")
  );
}

function isReviewArtifact(path: string): boolean {
  if (!path.startsWith(REVIEW_DIRECTORY) || !path.endsWith(".md")) {
    return false;
  }

  return path !== REVIEW_TEMPLATE && path !== REVIEW_README;
}

function normalizeReviewedPath(rawPath: string): string | null {
  const cleaned = rawPath
    .trim()
    .replace(/^`/, "")
    .replace(/`$/, "")
    .replace(/^["']/, "")
    .replace(/["']$/, "");

  if (cleaned === "") {
    return null;
  }

  const normalized = relative(PROJECT_ROOT, resolve(PROJECT_ROOT, cleaned)).replaceAll("\\", "/");
  if (normalized.startsWith("..")) {
    return null;
  }

  return normalized;
}

function extractReviewedFiles(content: string): string[] {
  const reviewedFiles: string[] = [];

  const lines = content.split("\n");
  let inFilesReviewedSection = false;
  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!inFilesReviewedSection) {
      if (line === "## Files Reviewed") {
        inFilesReviewedSection = true;
      }
      continue;
    }

    if (line.startsWith("## ")) {
      break;
    }

    if (!line.startsWith("- ") && !line.startsWith("* ")) {
      continue;
    }

    const bulletText = line.slice(2).trim();
    const linkTargetMatch = bulletText.match(/\(([^)]+)\)/);
    const candidatePath = linkTargetMatch?.[1] ?? bulletText;
    const normalized = normalizeReviewedPath(candidatePath);
    if (normalized !== null) {
      reviewedFiles.push(normalized);
    }
  }

  return reviewedFiles;
}

function extractSectionContent(content: string, heading: string): string {
  const lines = content.split("\n");
  let inSection = false;
  const sectionLines: string[] = [];

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!inSection) {
      if (line === heading) {
        inSection = true;
      }
      continue;
    }

    if (line.startsWith("## ")) {
      break;
    }
    sectionLines.push(rawLine);
  }

  return sectionLines.join("\n");
}

function validateReviewArtifact(path: string): {
  errors: string[];
  reviewedFiles: string[];
  performanceEvidence: string;
} {
  if (!existsSync(join(PROJECT_ROOT, path))) {
    return {
      errors: [`${path}: file does not exist`],
      reviewedFiles: [],
      performanceEvidence: "",
    };
  }

  const content = readFileSync(join(PROJECT_ROOT, path), "utf8");
  const errors = REQUIRED_HEADINGS.flatMap((heading) =>
    content.includes(heading) ? [] : [`${path}: missing heading "${heading}"`],
  );

  return {
    errors,
    reviewedFiles: extractReviewedFiles(content),
    performanceEvidence: extractSectionContent(content, "## Memory / Performance Evidence"),
  };
}

function touchesGenerationHotPath(path: string): boolean {
  return (
    GENERATION_HOT_PATH_FILES.has(path) ||
    GENERATION_HOT_PATH_PREFIXES.some((prefix) => path.startsWith(prefix))
  );
}

const changedFiles = uniqueSorted([...listChangedTrackedFiles(), ...listUntrackedFiles()]);
const changedRuntimeFiles = changedFiles.filter((path) => isRuntimeSensitiveFile(path));

if (changedRuntimeFiles.length === 0) {
  console.log("No runtime-sensitive production changes detected.");
  process.exit(0);
}

const changedReviewArtifacts = changedFiles.filter((path) => isReviewArtifact(path));

if (changedReviewArtifacts.length === 0) {
  console.error(
    "Runtime-sensitive production files changed, but no runtime review artifact was added.",
  );
  console.error("");
  console.error("Changed runtime-sensitive files:");
  for (const path of changedRuntimeFiles) {
    console.error(`  ${path}`);
  }
  console.error("");
  console.error(
    "Add or update a review artifact under docs/reviews/ using docs/reviews/_template.md.",
  );
  process.exit(1);
}

const reviewFileSet = new Set<string>();
const artifactErrors: string[] = [];
const performanceEvidenceSections: string[] = [];

for (const path of changedReviewArtifacts) {
  const { errors, reviewedFiles, performanceEvidence } = validateReviewArtifact(path);
  artifactErrors.push(...errors);
  performanceEvidenceSections.push(performanceEvidence);
  for (const reviewedFile of reviewedFiles) {
    reviewFileSet.add(reviewedFile);
  }
}

if (artifactErrors.length > 0) {
  console.error("Runtime review artifacts are incomplete:\n");
  for (const error of artifactErrors) {
    console.error(`  ${error}`);
  }
  process.exit(1);
}

const missingReviewedFiles = changedRuntimeFiles.filter((path) => !reviewFileSet.has(path));
if (missingReviewedFiles.length > 0) {
  console.error("Runtime review artifacts do not list every changed runtime-sensitive file:\n");
  console.error("Missing from Files Reviewed:");
  for (const path of missingReviewedFiles) {
    console.error(`  ${path}`);
  }
  console.error("");
  console.error("Add the missing paths to the Files Reviewed section of the review artifact.");
  process.exit(1);
}

const requiresGenerationBenchmarks = changedRuntimeFiles.some((path) =>
  touchesGenerationHotPath(path),
);
if (requiresGenerationBenchmarks) {
  const combinedEvidence = performanceEvidenceSections.join("\n");
  const missingBenchmarks: string[] = [];
  if (!combinedEvidence.includes("bench:generation")) {
    missingBenchmarks.push("bench:generation");
  }
  if (!combinedEvidence.includes("bench:generation:parity")) {
    missingBenchmarks.push("bench:generation:parity");
  }

  if (missingBenchmarks.length > 0) {
    console.error(
      "Generation hot-path changes require benchmark evidence in the runtime review:\n",
    );
    for (const benchmark of missingBenchmarks) {
      console.error(`  missing mention of ${benchmark} in "## Memory / Performance Evidence"`);
    }
    process.exit(1);
  }
}

console.log("Runtime-sensitive production changes detected.");
console.log("Validated runtime review artifacts:");
for (const path of changedReviewArtifacts) {
  console.log(`  ${path}`);
}
