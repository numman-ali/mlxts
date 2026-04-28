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

const TOOL_CALL_PATTERN = /(?:<tool_call>|<\|tool_call>)([\s\S]*?)(?:<\/tool_call>|<tool_call\|>)/g;
const NATIVE_FUNCTION_PATTERN =
  /^<function=([A-Za-z_][A-Za-z0-9_.:-]*)>\s*([\s\S]*?)\s*<\/function>$/;
const NATIVE_PARAMETER_PATTERN =
  /<parameter=([A-Za-z_][A-Za-z0-9_.:-]*)>\s*([\s\S]*?)\s*<\/parameter>/g;
const GEMMA_FUNCTION_PATTERN = /^call:([A-Za-z_][A-Za-z0-9_.:-]*)\{([\s\S]*)\}$/;
const GEMMA_STRING_MARKER = '<|"|>';

type GemmaScanState = {
  depth: number;
  index: number;
  inString: boolean;
};

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

function advanceGemmaStringMarker(text: string, state: GemmaScanState): boolean {
  if (!text.startsWith(GEMMA_STRING_MARKER, state.index)) {
    return false;
  }
  state.inString = !state.inString;
  state.index += GEMMA_STRING_MARKER.length;
  return true;
}

function updateGemmaContainerDepth(state: GemmaScanState, char: string | undefined): void {
  if (state.inString) {
    return;
  }
  if (char === "{" || char === "[") {
    state.depth += 1;
    return;
  }
  if (char === "}" || char === "]") {
    state.depth -= 1;
  }
}

function isTopLevelGemmaChar(
  state: GemmaScanState,
  char: string | undefined,
  expected: string,
): boolean {
  return !state.inString && state.depth === 0 && char === expected;
}

function splitTopLevelEntries(text: string): string[] {
  const entries: string[] = [];
  const state: GemmaScanState = { depth: 0, index: 0, inString: false };
  let start = 0;

  while (state.index < text.length) {
    if (advanceGemmaStringMarker(text, state)) {
      continue;
    }
    const char = text[state.index];
    if (isTopLevelGemmaChar(state, char, ",")) {
      entries.push(text.slice(start, state.index).trim());
      start = state.index + 1;
    } else {
      updateGemmaContainerDepth(state, char);
    }
    state.index += 1;
  }

  const tail = text.slice(start).trim();
  return tail === "" ? entries : [...entries, tail];
}

function topLevelColonIndex(text: string): number {
  const state: GemmaScanState = { depth: 0, index: 0, inString: false };
  while (state.index < text.length) {
    if (advanceGemmaStringMarker(text, state)) {
      continue;
    }
    const char = text[state.index];
    if (isTopLevelGemmaChar(state, char, ":")) {
      return state.index;
    }
    updateGemmaContainerDepth(state, char);
    state.index += 1;
  }
  return -1;
}

function parseGemmaValue(value: string): unknown {
  const trimmed = value.trim();
  if (trimmed.startsWith(GEMMA_STRING_MARKER) && trimmed.endsWith(GEMMA_STRING_MARKER)) {
    return trimmed.slice(GEMMA_STRING_MARKER.length, -GEMMA_STRING_MARKER.length);
  }
  if (trimmed.startsWith("{") && trimmed.endsWith("}")) {
    return parseGemmaObject(trimmed.slice(1, -1));
  }
  if (trimmed.startsWith("[") && trimmed.endsWith("]")) {
    return splitTopLevelEntries(trimmed.slice(1, -1)).map(parseGemmaValue);
  }
  if (trimmed === "true") {
    return true;
  }
  if (trimmed === "false") {
    return false;
  }
  if (trimmed === "null") {
    return null;
  }
  const numberValue = Number(trimmed);
  return Number.isFinite(numberValue) && trimmed !== "" ? numberValue : trimmed;
}

function parseGemmaObject(payload: string): Record<string, unknown> {
  const args: Record<string, unknown> = {};
  for (const entry of splitTopLevelEntries(payload)) {
    const colon = topLevelColonIndex(entry);
    if (colon <= 0) {
      throw new Error("Gemma tool call arguments must use key:value entries.");
    }
    const key = entry.slice(0, colon).trim();
    if (!/^[A-Za-z_][A-Za-z0-9_.:-]*$/.test(key)) {
      throw new Error("Gemma tool call argument key is invalid.");
    }
    args[key] = parseGemmaValue(entry.slice(colon + 1));
  }
  return args;
}

function parseGemmaFunctionPayload(payload: string, index: number): ChatToolCall | null {
  const match = GEMMA_FUNCTION_PATTERN.exec(payload.trim());
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
      arguments: JSON.stringify(parseGemmaObject(match[2] ?? "")),
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
  const gemmaCall = nativeCall ?? parseGemmaFunctionPayload(payload, index);
  return gemmaCall ?? parseJsonToolCall(payload, index);
}

/** Parse one generated tool-call payload into the OpenAI-compatible message shape. */
export function parseOpenAIChatToolCallPayload(payload: string, index: number): ChatToolCall {
  return parseToolCallPayload(payload, index);
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
