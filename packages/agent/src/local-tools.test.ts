import { describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

import { createReadOnlyFileTools } from "./local-tools";
import type { AgentTool } from "./types";

function toolByName(tools: readonly AgentTool[], name: string): AgentTool {
  const tool = tools.find((entry) => entry.name === name);
  if (tool === undefined) {
    throw new Error(`Missing tool ${name}`);
  }
  return tool;
}

describe("createReadOnlyFileTools", () => {
  test("lists and reads files inside the configured root", async () => {
    const root = mkdtempSync(join(tmpdir(), "mlxts-agent-"));
    try {
      writeFileSync(join(root, "README.md"), "hello agent");
      const tools = createReadOnlyFileTools({ root, maxBytes: 5 });
      const listFiles = toolByName(tools, "list_files");
      const readFile = toolByName(tools, "read_file");

      expect(await listFiles?.execute({ pattern: "*.md" }, context())).toBe("README.md");
      expect(await readFile?.execute({ path: "README.md" }, context())).toBe(
        "hello\n...[truncated 6 chars]",
      );
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  test("rejects paths outside the configured root", async () => {
    const root = mkdtempSync(join(tmpdir(), "mlxts-agent-"));
    try {
      const tools = createReadOnlyFileTools({ root });
      const readFile = toolByName(tools, "read_file");

      await expect(readFile.execute({ path: "../secret.txt" }, context())).rejects.toThrow(
        "inside",
      );
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  test("reports missing or invalid read paths", async () => {
    const root = mkdtempSync(join(tmpdir(), "mlxts-agent-"));
    try {
      const tools = createReadOnlyFileTools({ root });
      const readFile = toolByName(tools, "read_file");

      await expect(Promise.resolve(readFile.execute({}, context()))).rejects.toThrow(
        "non-empty string",
      );
      await expect(readFile.execute({ path: "missing.txt" }, context())).resolves.toEqual({
        content: "File not found: missing.txt",
        isError: true,
      });
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });
});

function context() {
  return {
    messages: [],
    iteration: 0,
    toolCall: { id: "call-1", name: "test", arguments: {} },
  };
}
