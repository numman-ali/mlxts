import { describe, expect, test } from "bun:test";

import { formatToolInstructions, parseToolCalls } from "./tool-calls";
import type { AgentTool } from "./types";

const readTool: AgentTool = {
  name: "read_file",
  description: "Read a file",
  parameters: {
    type: "object",
    properties: { path: { type: "string" } },
    required: ["path"],
  },
  execute() {
    return "";
  },
};

describe("tool call parsing", () => {
  test("parses XML-style tool calls", () => {
    const parsed = parseToolCalls(
      'I will inspect it.\n<tool_call>{"name":"read_file","arguments":{"path":"README.md"}}</tool_call>',
    );

    expect(parsed.text).toBe("I will inspect it.");
    expect(parsed.calls).toEqual([
      { id: "tool-1", name: "read_file", arguments: { path: "README.md" } },
    ]);
  });

  test("parses Qwen-style function tool calls", () => {
    expect(
      parseToolCalls("<tool_call>\n<function=list_files>\n</function>\n</tool_call>").calls,
    ).toEqual([{ id: "tool-1", name: "list_files", arguments: {} }]);
    expect(
      parseToolCalls(
        "<tool_call>\n<function=read_file>\n<parameter=path>README.md</parameter>\n</function>\n</tool_call>",
      ).calls,
    ).toEqual([{ id: "tool-1", name: "read_file", arguments: { path: "README.md" } }]);
    expect(
      parseToolCalls('<tool_call><function=read_file>{"path":"README.md"}</function></tool_call>')
        .calls,
    ).toEqual([{ id: "tool-1", name: "read_file", arguments: { path: "README.md" } }]);
  });

  test("parses bare JSON only when enabled and guarded by arguments", () => {
    expect(parseToolCalls('{"name":"Nomi"}', { allowBareJson: true }).calls).toEqual([]);
    expect(
      parseToolCalls('{"name":"read_file","arguments":{"path":"README.md"}}', {
        allowBareJson: true,
      }).calls,
    ).toEqual([{ id: "tool-1", name: "read_file", arguments: { path: "README.md" } }]);
    expect(parseToolCalls('{"name":"read_file","arguments":{}}').calls).toEqual([]);
  });

  test("rejects malformed XML tool calls", () => {
    expect(() => parseToolCalls("<tool_call>{</tool_call>")).toThrow("Invalid tool_call JSON");
    expect(() => parseToolCalls("<tool_call>[]</tool_call>")).toThrow("must be a JSON object");
    expect(() => parseToolCalls('<tool_call>{"name":"","arguments":{}}</tool_call>')).toThrow(
      "non-empty name",
    );
    expect(() =>
      parseToolCalls('<tool_call>{"name":"read_file","arguments":[]}</tool_call>'),
    ).toThrow("arguments must be a JSON object");
    expect(() =>
      parseToolCalls("<tool_call><function=read_file>README.md</function></tool_call>"),
    ).toThrow("Unsupported function-style");
    expect(parseToolCalls("{", { allowBareJson: true }).calls).toEqual([]);
  });

  test("formats tool instructions", () => {
    const instructions = formatToolInstructions([readTool]);

    expect(instructions).toContain("<tool_call>");
    expect(instructions).toContain("read_file");
    expect(instructions).toContain("parameters");
  });
});
