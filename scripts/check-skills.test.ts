import { describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

import { validateRepoSkills, validateSkillMarkdown } from "./check-skills";

function skillMarkdown(frontmatter: string): string {
  return `---\n${frontmatter.trim()}\n---\n\n# Skill\n\nUse this skill for a focused workflow.\n`;
}

describe("check-skills", () => {
  test("accepts scalar and folded-string skill frontmatter", () => {
    const validation = validateSkillMarkdown(
      join(process.cwd(), ".agents", "skills", "axi", "SKILL.md"),
      skillMarkdown(
        [
          "name: axi",
          "description: >",
          "  Agent eXperience Interface standards for CLIs that agents",
          "  use through shell execution.",
        ].join("\n"),
      ),
    );

    expect(validation.issues).toEqual([]);
    expect(validation.result).toEqual({
      name: "axi",
      description:
        "Agent eXperience Interface standards for CLIs that agents use through shell execution.",
    });
  });

  test("rejects sequence-shaped descriptions before the skill loader sees them", () => {
    const validation = validateSkillMarkdown(
      join(process.cwd(), ".agents", "skills", "serve-cache-qa", "SKILL.md"),
      skillMarkdown(
        ["name: serve-cache-qa", "description:", "  - Diagnose serve cache behavior."].join("\n"),
      ),
    );

    expect(validation.issues.map((issue) => issue.message)).toContain(
      "description must be a scalar string, not an empty or nested YAML value.",
    );
  });

  test("validates repo-local skill directory names", () => {
    const root = mkdtempSync(join(tmpdir(), "mlxts-skills-"));
    try {
      mkdirSync(join(root, "serve-cache-qa"), { recursive: true });
      writeFileSync(
        join(root, "serve-cache-qa", "SKILL.md"),
        skillMarkdown(
          ["name: serve-cache-qa", "description: Diagnose serve cache behavior."].join("\n"),
        ),
      );

      expect(validateRepoSkills(root)).toMatchObject({
        checked: [{ name: "serve-cache-qa" }],
        issues: [],
      });

      mkdirSync(join(root, "bad-name"), { recursive: true });
      writeFileSync(
        join(root, "bad-name", "SKILL.md"),
        skillMarkdown(["name: other-name", "description: Diagnose a mismatch."].join("\n")),
      );

      expect(validateRepoSkills(root).issues.map((issue) => issue.message)).toContain(
        'name must match skill directory "bad-name".',
      );
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });
});
