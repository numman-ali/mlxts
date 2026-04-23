/**
 * Tool-call parsing and prompt instructions.
 * @module
 */

import type { AgentTool, AgentToolCall } from "./types";

const TOOL_CALL_PATTERN = /<tool_call>([\s\S]*?)<\/tool_call>/g;
const NATIVE_FUNCTION_PATTERN =
  /^<function=([A-Za-z_][A-Za-z0-9_.:-]*)>\s*([\s\S]*?)\s*<\/function>$/;
const NATIVE_PARAMETER_PATTERN =
  /<parameter=([A-Za-z_][A-Za-z0-9_.:-]*)>\s*([\s\S]*?)\s*<\/parameter>/g;

export class AgentToolCallParseError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "AgentToolCallParseError";
  }
}

export type ParsedToolCalls = {
  calls: AgentToolCall[];
  text: string;
};

export type ParseToolCallsOptions = {
  allowBareJson?: boolean;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function parseJsonPayload(payload: string): unknown {
  try {
    return JSON.parse(payload);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new AgentToolCallParseError(`Invalid tool_call JSON: ${message}`);
  }
}

function nativeParameterValue(value: string): unknown {
  const trimmed = value.trim();
  if (trimmed === "") {
    return "";
  }
  try {
    return JSON.parse(trimmed);
  } catch {
    return trimmed;
  }
}

function parseNativeFunctionArgs(payload: string): Record<string, unknown> {
  const trimmed = payload.trim();
  if (trimmed === "") {
    return {};
  }
  if (trimmed.startsWith("{")) {
    const parsed = parseJsonPayload(trimmed);
    if (!isRecord(parsed)) {
      throw new AgentToolCallParseError("Function-style tool call arguments must be an object.");
    }
    return parsed;
  }

  const args: Record<string, unknown> = {};
  const unmatched = trimmed.replace(
    NATIVE_PARAMETER_PATTERN,
    (_match, name: string | undefined, value: string | undefined) => {
      if (name !== undefined && value !== undefined) {
        args[name] = nativeParameterValue(value);
      }
      return "";
    },
  );
  if (unmatched.trim() !== "") {
    throw new AgentToolCallParseError("Unsupported function-style tool_call arguments.");
  }
  return args;
}

function parseNativeFunctionPayload(payload: string, index: number): AgentToolCall | null {
  const match = NATIVE_FUNCTION_PATTERN.exec(payload.trim());
  if (match === null) {
    return null;
  }

  const name = match[1];
  if (name === undefined) {
    return null;
  }
  const args = parseNativeFunctionArgs(match[2] ?? "");
  return {
    id: `tool-${index + 1}`,
    name,
    arguments: args,
  };
}

function parseToolCallPayload(payload: string, index: number): AgentToolCall {
  const nativeCall = parseNativeFunctionPayload(payload, index);
  if (nativeCall !== null) {
    return nativeCall;
  }

  const parsed = parseJsonPayload(payload);
  if (!isRecord(parsed)) {
    throw new AgentToolCallParseError("Tool call payload must be a JSON object.");
  }

  const name = parsed.name;
  if (typeof name !== "string" || name.trim() === "") {
    throw new AgentToolCallParseError("Tool call payload must include a non-empty name.");
  }

  const args = parsed.arguments ?? {};
  if (!isRecord(args)) {
    throw new AgentToolCallParseError("Tool call arguments must be a JSON object.");
  }

  const id = parsed.id;
  return {
    id: typeof id === "string" && id.trim() !== "" ? id : `tool-${index + 1}`,
    name,
    arguments: args,
  };
}

function parseWholeJsonToolCall(content: string): AgentToolCall[] {
  const trimmed = content.trim();
  if (!trimmed.startsWith("{") || !trimmed.endsWith("}")) {
    return [];
  }

  let parsed: unknown;
  try {
    parsed = parseJsonPayload(trimmed);
  } catch {
    return [];
  }
  if (!isRecord(parsed) || typeof parsed.name !== "string" || !("arguments" in parsed)) {
    return [];
  }
  return [parseToolCallPayload(trimmed, 0)];
}

/** Parse conservative XML-style tool-call envelopes from a model response. */
export function parseToolCalls(
  content: string,
  options: ParseToolCallsOptions = {},
): ParsedToolCalls {
  const calls: AgentToolCall[] = [];
  const stripped = content.replace(TOOL_CALL_PATTERN, (_match, payload: string | undefined) => {
    if (payload === undefined) {
      return "";
    }
    calls.push(parseToolCallPayload(payload.trim(), calls.length));
    return "";
  });

  if (calls.length > 0) {
    return { calls, text: stripped.trim() };
  }

  return {
    calls: options.allowBareJson === true ? parseWholeJsonToolCall(content) : [],
    text: content,
  };
}

function formatToolSchema(tool: AgentTool): string {
  return JSON.stringify(tool.parameters ?? { type: "object", properties: {} });
}

/** Build model-facing instructions for the package's conservative tool-call envelope. */
export function formatToolInstructions(tools: readonly AgentTool[]): string {
  if (tools.length === 0) {
    return "";
  }

  return [
    "You can use tools when needed.",
    "To call a tool, respond with one or more tool_call blocks and no extra prose:",
    '<tool_call>{"name":"tool_name","arguments":{}}</tool_call>',
    "Some models may instead emit <tool_call><function=tool_name>{}</function></tool_call>; that is also accepted.",
    "After tool results are provided, answer the user normally.",
    "",
    "Available tools:",
    ...tools.map(
      (tool) => `- ${tool.name}: ${tool.description}\n  parameters: ${formatToolSchema(tool)}`,
    ),
  ].join("\n");
}
