/**
 * OpenAI chat-completion SSE parsing for agent adapters.
 * @module
 */

import { cleanReasoningFromText } from "./reasoning-tags";
import type { AgentModelResponse, AgentModelStreamEvent, AgentToolCall } from "./types";

type StreamingToolCallState = {
  id?: string;
  name?: string;
  arguments: string;
};

type StreamingResponseState = {
  content: string;
  reasoningContent: string;
  toolCalls: Map<number, StreamingToolCallState>;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function parseArguments(value: string): Record<string, unknown> {
  if (value.trim() === "") {
    return {};
  }
  try {
    const parsed: unknown = JSON.parse(value);
    return isRecord(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

function appendStringField(
  record: Record<string, unknown>,
  key: string,
  append: (value: string) => void,
): void {
  const value = record[key];
  if (typeof value === "string" && value !== "") {
    append(value);
  }
}

function toolCallIndex(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isInteger(value) && value >= 0 ? value : fallback;
}

function streamingToolCallAt(state: StreamingResponseState, index: number): StreamingToolCallState {
  const existing = state.toolCalls.get(index);
  if (existing !== undefined) {
    return existing;
  }
  const created = { arguments: "" };
  state.toolCalls.set(index, created);
  return created;
}

function streamingToolCallDelta(
  value: unknown,
  fallbackIndex: number,
): AgentModelStreamEvent | null {
  if (!isRecord(value)) {
    return null;
  }
  const event: AgentModelStreamEvent = {
    type: "tool_call_delta",
    index: toolCallIndex(value.index, fallbackIndex),
  };
  if (typeof value.id === "string" && value.id.trim() !== "") {
    event.id = value.id;
  }
  if (isRecord(value.function)) {
    if (typeof value.function.name === "string" && value.function.name !== "") {
      event.nameDelta = value.function.name;
    }
    if (typeof value.function.arguments === "string" && value.function.arguments !== "") {
      event.argumentsDelta = value.function.arguments;
    }
  }
  return event;
}

function streamingChoiceEvents(value: unknown): AgentModelStreamEvent[] {
  if (!isRecord(value) || !isRecord(value.delta)) {
    return [];
  }
  const events: AgentModelStreamEvent[] = [];
  appendStringField(value.delta, "content", (content) => {
    events.push({ type: "content_delta", contentDelta: content });
  });
  appendStringField(value.delta, "reasoning_content", (reasoning) => {
    events.push({ type: "reasoning_delta", reasoningContentDelta: reasoning });
  });

  const toolCalls = value.delta.tool_calls;
  if (Array.isArray(toolCalls)) {
    toolCalls.forEach((call, index) => {
      const event = streamingToolCallDelta(call, index);
      if (event !== null) {
        events.push(event);
      }
    });
  }
  return events;
}

function streamingChunkEvents(value: unknown): AgentModelStreamEvent[] {
  if (!isRecord(value) || !Array.isArray(value.choices)) {
    return [];
  }
  const events: AgentModelStreamEvent[] = [];
  for (const choice of value.choices) {
    events.push(...streamingChoiceEvents(choice));
  }
  return events;
}

function applyModelStreamEvent(state: StreamingResponseState, event: AgentModelStreamEvent): void {
  switch (event.type) {
    case "content_delta":
      state.content += event.contentDelta;
      return;
    case "reasoning_delta":
      state.reasoningContent += event.reasoningContentDelta;
      return;
    case "tool_call_delta": {
      const entry = streamingToolCallAt(state, event.index);
      if (event.id !== undefined && event.id.trim() !== "") {
        entry.id = event.id;
      }
      if (event.nameDelta !== undefined && event.nameDelta !== "") {
        entry.name = `${entry.name ?? ""}${event.nameDelta}`;
      }
      if (event.argumentsDelta !== undefined) {
        entry.arguments += event.argumentsDelta;
      }
    }
  }
}

function streamingToolCalls(state: StreamingResponseState): AgentToolCall[] | undefined {
  const calls = [...state.toolCalls.entries()]
    .sort(([left], [right]) => left - right)
    .map(([index, call]) => {
      if (call.name === undefined || call.name.trim() === "") {
        return null;
      }
      return {
        id: call.id ?? `tool-${index + 1}`,
        name: call.name,
        arguments: parseArguments(call.arguments),
      };
    })
    .filter((call): call is AgentToolCall => call !== null);
  return calls.length === 0 ? undefined : calls;
}

function ssePayloads(block: string): string | null {
  const payload = block
    .split("\n")
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice("data:".length).trimStart())
    .join("\n")
    .trim();
  return payload === "" ? null : payload;
}

async function* readSsePayloads(response: Response): AsyncIterable<string> {
  if (response.body === null) {
    throw new Error("Streaming chat completion response did not include a body.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const next = await reader.read();
    if (next.done) {
      buffer += decoder.decode();
      break;
    }
    buffer += decoder.decode(next.value, { stream: true });
    buffer = buffer.replace(/\r\n/g, "\n");

    let boundary = buffer.indexOf("\n\n");
    while (boundary >= 0) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const payload = ssePayloads(block);
      if (payload !== null) {
        yield payload;
      }
      boundary = buffer.indexOf("\n\n");
    }
  }

  const payload = ssePayloads(buffer.replace(/\r\n/g, "\n"));
  if (payload !== null) {
    yield payload;
  }
}

/** Stream OpenAI chat-completion SSE chunks as protocol-neutral agent model events. */
export async function* streamOpenAIChatCompletionEvents(
  response: Response,
): AsyncIterable<AgentModelStreamEvent> {
  for await (const payload of readSsePayloads(response)) {
    if (payload === "[DONE]") {
      break;
    }
    let parsed: unknown;
    try {
      parsed = JSON.parse(payload);
    } catch (error) {
      throw new Error(
        `Streaming chat completion response included malformed JSON: ${
          error instanceof Error ? error.message : String(error)
        }`,
      );
    }
    yield* streamingChunkEvents(parsed);
  }
}

/** Aggregate streamed agent model events into the final response shape used by the tool loop. */
export async function aggregateAgentModelStream(
  events: AsyncIterable<AgentModelStreamEvent>,
): Promise<AgentModelResponse> {
  const state: StreamingResponseState = {
    content: "",
    reasoningContent: "",
    toolCalls: new Map(),
  };

  for await (const event of events) {
    applyModelStreamEvent(state, event);
  }

  const parsedContent = cleanReasoningFromText(state.content);
  const reasoningContent =
    state.reasoningContent.trim() === ""
      ? parsedContent.reasoningContent
      : state.reasoningContent.trim();
  const toolCalls = streamingToolCalls(state);
  return toolCalls === undefined
    ? {
        content: parsedContent.content,
        ...(reasoningContent === undefined || reasoningContent === "" ? {} : { reasoningContent }),
      }
    : {
        content: parsedContent.content,
        ...(reasoningContent === undefined || reasoningContent === "" ? {} : { reasoningContent }),
        toolCalls,
      };
}
