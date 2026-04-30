/**
 * Validates repo-local Codex skills before they reach the auto-loader.
 */

import { existsSync, readdirSync, readFileSync, statSync } from "fs";
import { basename, join, relative } from "path";

const PROJECT_ROOT = join(import.meta.dirname, "..");
const SKILLS_ROOT = join(PROJECT_ROOT, ".agents", "skills");
const REQUIRED_STRING_FIELDS = ["name", "description"] as const;
const SKILL_NAME_PATTERN = /^[a-z0-9][a-z0-9-]*$/;

type RequiredStringField = (typeof REQUIRED_STRING_FIELDS)[number];

export type SkillValidationIssue = {
  file: string;
  line?: number;
  message: string;
};

export type SkillValidationResult = {
  name: string;
  description: string;
};

type ParsedField = {
  value: string;
  line: number;
};

type FrontmatterEntry =
  | { kind: "skip"; nextIndex: number }
  | { kind: "field"; key: string; rawValue: string; lineNumber: number; nextIndex: number };

function repoPath(path: string): string {
  return relative(PROJECT_ROOT, path).replaceAll("\\", "/");
}

function frontmatterFenceEnd(lines: readonly string[]): number | undefined {
  for (let index = 1; index < lines.length; index += 1) {
    if (lines[index]?.trim() === "---") {
      return index;
    }
  }
  return undefined;
}

function blockScalarValue(
  lines: readonly string[],
  startIndex: number,
  marker: string,
): { value: string; endIndex: number } {
  const blockLines: string[] = [];
  let index = startIndex;
  for (; index < lines.length; index += 1) {
    const line = lines[index] ?? "";
    if (line.trim() !== "" && !/^\s/.test(line)) {
      break;
    }
    blockLines.push(line.trim());
  }

  const nonEmptyLines = blockLines.filter((line) => line !== "");
  const value = marker.startsWith("|") ? nonEmptyLines.join("\n") : nonEmptyLines.join(" ");
  return { value, endIndex: index };
}

function unquoteScalar(value: string): string {
  if (
    (value.startsWith('"') && value.endsWith('"')) ||
    (value.startsWith("'") && value.endsWith("'"))
  ) {
    return value.slice(1, -1);
  }
  return value;
}

function scalarFieldValue(
  fieldName: string,
  rawValue: string,
  line: number,
  file: string,
  lines: readonly string[],
  nextIndex: number,
  issues: SkillValidationIssue[],
): { parsed?: ParsedField; endIndex: number } {
  const value = rawValue.trim();
  if (value === "") {
    issues.push({
      file,
      line,
      message: `${fieldName} must be a scalar string, not an empty or nested YAML value.`,
    });
    return { endIndex: nextIndex };
  }

  if (value.startsWith("[") || value.startsWith("{")) {
    issues.push({
      file,
      line,
      message: `${fieldName} must be a scalar string, not a collection.`,
    });
    return { endIndex: nextIndex };
  }

  if (/^[>|][+-]?$/.test(value)) {
    const block = blockScalarValue(lines, nextIndex, value);
    return {
      parsed: { value: block.value.trim(), line },
      endIndex: block.endIndex,
    };
  }

  return {
    parsed: { value: unquoteScalar(value), line },
    endIndex: nextIndex,
  };
}

function isRequiredStringField(key: string): key is RequiredStringField {
  return key === "name" || key === "description";
}

function readFrontmatterEntry(
  file: string,
  frontmatter: readonly string[],
  index: number,
  issues: SkillValidationIssue[],
): FrontmatterEntry {
  const line = frontmatter[index] ?? "";
  const lineNumber = index + 2;
  const trimmed = line.trim();
  if (trimmed === "" || trimmed.startsWith("#")) {
    return { kind: "skip", nextIndex: index + 1 };
  }
  if (/^\s/.test(line)) {
    issues.push({
      file,
      line: lineNumber,
      message: "frontmatter must use top-level scalar fields only.",
    });
    return { kind: "skip", nextIndex: index + 1 };
  }

  const match = line.match(/^([A-Za-z0-9_-]+):(?:\s*(.*))?$/);
  if (match === null) {
    issues.push({
      file,
      line: lineNumber,
      message: "frontmatter lines must use `key: value` syntax.",
    });
    return { kind: "skip", nextIndex: index + 1 };
  }

  return {
    kind: "field",
    key: match[1] ?? "",
    rawValue: match[2] ?? "",
    lineNumber,
    nextIndex: index + 1,
  };
}

function skipOptionalField(
  entry: Extract<FrontmatterEntry, { kind: "field" }>,
  frontmatter: readonly string[],
  file: string,
  issues: SkillValidationIssue[],
): number {
  const value = entry.rawValue.trim();
  if (value === "") {
    issues.push({
      file,
      line: entry.lineNumber,
      message: "repo-local skill frontmatter uses flat scalar metadata only.",
    });
  }
  if (/^[>|][+-]?$/.test(value)) {
    return blockScalarValue(frontmatter, entry.nextIndex, value).endIndex;
  }
  return entry.nextIndex;
}

function recordRequiredField(
  entry: Extract<FrontmatterEntry, { kind: "field" }>,
  fields: Partial<Record<RequiredStringField, ParsedField>>,
  frontmatter: readonly string[],
  file: string,
  issues: SkillValidationIssue[],
): number {
  const key = entry.key;
  if (!isRequiredStringField(key)) {
    return skipOptionalField(entry, frontmatter, file, issues);
  }

  if (fields[key] !== undefined) {
    issues.push({ file, line: entry.lineNumber, message: `${key} is duplicated.` });
  }

  const { parsed, endIndex } = scalarFieldValue(
    key,
    entry.rawValue,
    entry.lineNumber,
    file,
    frontmatter,
    entry.nextIndex,
    issues,
  );
  if (parsed !== undefined) {
    fields[key] = parsed;
  }
  return endIndex;
}

function parseRequiredFields(
  file: string,
  frontmatter: readonly string[],
  issues: SkillValidationIssue[],
): Partial<Record<RequiredStringField, ParsedField>> {
  const fields: Partial<Record<RequiredStringField, ParsedField>> = {};
  let index = 0;
  while (index < frontmatter.length) {
    const entry = readFrontmatterEntry(file, frontmatter, index, issues);
    index =
      entry.kind === "skip"
        ? entry.nextIndex
        : recordRequiredField(entry, fields, frontmatter, file, issues);
  }
  return fields;
}

export function validateSkillMarkdown(
  filePath: string,
  content: string,
): { result?: SkillValidationResult; issues: SkillValidationIssue[] } {
  const file = repoPath(filePath);
  const issues: SkillValidationIssue[] = [];
  const lines = content.split(/\r?\n/);
  if (lines[0]?.trim() !== "---") {
    return {
      issues: [{ file, line: 1, message: "SKILL.md must start with YAML frontmatter." }],
    };
  }

  const fenceEnd = frontmatterFenceEnd(lines);
  if (fenceEnd === undefined) {
    return {
      issues: [{ file, line: 1, message: "SKILL.md frontmatter must close with `---`." }],
    };
  }

  const frontmatter = lines.slice(1, fenceEnd);
  const fields = parseRequiredFields(file, frontmatter, issues);
  for (const field of REQUIRED_STRING_FIELDS) {
    if (fields[field] === undefined) {
      issues.push({ file, message: `frontmatter is missing ${field}.` });
    }
  }

  const name = fields.name?.value.trim();
  const description = fields.description?.value.trim();
  if (name !== undefined && !SKILL_NAME_PATTERN.test(name)) {
    issues.push({
      file,
      line: fields.name?.line,
      message: "name must use lowercase letters, digits, and hyphens.",
    });
  }
  if (description !== undefined && description === "") {
    issues.push({
      file,
      line: fields.description?.line,
      message: "description must not be empty.",
    });
  }

  const body = lines
    .slice(fenceEnd + 1)
    .join("\n")
    .trim();
  if (body === "") {
    issues.push({ file, line: fenceEnd + 2, message: "SKILL.md must have body instructions." });
  }

  if (issues.length > 0 || name === undefined || description === undefined) {
    return { issues };
  }
  return { result: { name, description }, issues };
}

function skillFiles(root: string): string[] {
  if (!existsSync(root)) {
    return [];
  }
  return readdirSync(root)
    .map((entry) => join(root, entry))
    .filter((path) => statSync(path).isDirectory())
    .map((path) => join(path, "SKILL.md"))
    .filter((path) => existsSync(path))
    .sort((left, right) => left.localeCompare(right));
}

export function validateRepoSkills(root: string = SKILLS_ROOT): {
  checked: SkillValidationResult[];
  issues: SkillValidationIssue[];
} {
  const checked: SkillValidationResult[] = [];
  const issues: SkillValidationIssue[] = [];
  for (const filePath of skillFiles(root)) {
    const validation = validateSkillMarkdown(filePath, readFileSync(filePath, "utf8"));
    issues.push(...validation.issues);
    if (validation.result !== undefined) {
      const directoryName = basename(join(filePath, ".."));
      if (validation.result.name !== directoryName) {
        issues.push({
          file: repoPath(filePath),
          line: 2,
          message: `name must match skill directory "${directoryName}".`,
        });
      }
      checked.push(validation.result);
    }
  }
  return { checked, issues };
}

function printIssues(issues: readonly SkillValidationIssue[]): void {
  console.error("Repo-local skill metadata is invalid.\n");
  for (const issue of issues) {
    const location = issue.line === undefined ? issue.file : `${issue.file}:${issue.line}`;
    console.error(`  ${location}: ${issue.message}`);
  }
}

if (import.meta.main) {
  const { checked, issues } = validateRepoSkills();
  if (issues.length > 0) {
    printIssues(issues);
    process.exit(1);
  }

  console.log(
    `Checked ${checked.length} repo-local skills. All SKILL.md files have valid frontmatter.`,
  );
}
