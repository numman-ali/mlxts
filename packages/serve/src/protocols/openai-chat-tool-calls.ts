/**
 * Tool-call extraction for generated OpenAI chat responses.
 * @module
 */

import type { ChatToolCall } from "@mlxts/transformers";
import { isRecord } from "../errors";

export type ExtractedOpenAIChatToolCalls = {
  content: string;
  toolCalls: ChatToolCall[];
};

const TOOL_CALL_PATTERN = /<tool_call>([\s\S]*?)<\/tool_call>/g;
const NATIVE_FUNCTION_PATTERN =
  /^<function=([A-Za-z_][A-Za-z0-9_.:-]*)>\s*([\s\S]*?)\s*<\/function>$/;
const NATIVE_PARAMETER_PATTERN =
  /<parameter=([A-Za-z_][A-Za-z0-9_.:-]*)>\s*([\s\S]*?)\s*<\/parameter>/g;

function parseJsonPayload(payload: string): unknown {
  return JSON.parse(payload);
}

function nativeParameterValue(value: string): unknown {
  const trimmed = value.trim();
  if (trimmed === "") {
    return "";
  }
  try {
    return parseJsonPayload(trimmed);
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
      throw new Error("Function-style tool call arguments must be an object.");
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
    throw new Error("Unsupported function-style tool call arguments.");
  }
  return args;
}

function toolCallId(index: number, value: unknown): string {
  if (isRecord(value) && typeof value.id === "string" && value.id.trim() !== "") {
    return value.id;
  }
  return `call_${index + 1}`;
}

function parseNativeFunctionPayload(payload: string, index: number): ChatToolCall | null {
  const match = NATIVE_FUNCTION_PATTERN.exec(payload.trim());
  if (match === null) {
    return null;
  }

  const name = match[1];
  if (name === undefined) {
    return null;
  }
  return {
    id: `call_${index + 1}`,
    type: "function",
    function: {
      name,
      arguments: JSON.stringify(parseNativeFunctionArgs(match[2] ?? "")),
    },
  };
}

function parseJsonToolCall(payload: string, index: number): ChatToolCall {
  const parsed = parseJsonPayload(payload);
  if (!isRecord(parsed)) {
    throw new Error("Tool call payload must be an object.");
  }
  const name = parsed.name;
  if (typeof name !== "string" || name.trim() === "") {
    throw new Error("Tool call payload must include a non-empty name.");
  }
  const args = parsed.arguments ?? {};
  if (!isRecord(args)) {
    throw new Error("Tool call arguments must be an object.");
  }
  return {
    id: toolCallId(index, parsed),
    type: "function",
    function: {
      name,
      arguments: JSON.stringify(args),
    },
  };
}

function parseToolCallPayload(payload: string, index: number): ChatToolCall {
  const nativeCall = parseNativeFunctionPayload(payload, index);
  return nativeCall ?? parseJsonToolCall(payload, index);
}

/** Extract conservative generated tool-call envelopes from assistant text. */
export function extractOpenAIChatToolCalls(text: string): ExtractedOpenAIChatToolCalls | null {
  const calls: ChatToolCall[] = [];
  try {
    const content = text.replace(TOOL_CALL_PATTERN, (_match, payload: string | undefined) => {
      if (payload === undefined) {
        return "";
      }
      calls.push(parseToolCallPayload(payload.trim(), calls.length));
      return "";
    });
    return calls.length === 0 ? null : { content: content.trim(), toolCalls: calls };
  } catch {
    return null;
  }
}
