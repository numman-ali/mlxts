/**
 * Agent loop orchestration.
 * @module
 */

import { parseToolCalls } from "./tool-calls";
import type {
  AgentEvent,
  AgentMessage,
  AgentModelResponse,
  AgentModelStreamEvent,
  AgentRunOptions,
  AgentRunResult,
  AgentTool,
  AgentToolCall,
  AgentToolContext,
  AgentToolOutput,
  AgentToolResult,
} from "./types";

const DEFAULT_MAX_ITERATIONS = 8;
const DEFAULT_MAX_TOOL_RESULT_CHARS = 12_000;

function normalizeToolOutput(output: AgentToolOutput): AgentToolResult {
  return typeof output === "string" ? { content: output } : output;
}

function truncateToolResult(result: AgentToolResult, maxChars: number): AgentToolResult {
  if (result.content.length <= maxChars) {
    return result;
  }
  return {
    content: `${result.content.slice(0, maxChars)}\n...[truncated ${result.content.length - maxChars} chars]`,
    ...(result.isError === undefined ? {} : { isError: result.isError }),
  };
}

function validateTool(tool: AgentTool): void {
  if (tool.name.trim() === "") {
    throw new Error("Agent tools must have non-empty names.");
  }
}

function toolMap(tools: readonly AgentTool[]): Map<string, AgentTool> {
  const map = new Map<string, AgentTool>();
  for (const tool of tools) {
    validateTool(tool);
    if (map.has(tool.name)) {
      throw new Error(`Duplicate agent tool name: ${tool.name}`);
    }
    map.set(tool.name, tool);
  }
  return map;
}

async function emit(onEvent: AgentRunOptions["onEvent"], event: AgentEvent): Promise<void> {
  await onEvent?.(event);
}

function unknownToolResult(
  call: AgentToolCall,
  tools: ReadonlyMap<string, AgentTool>,
): AgentToolResult {
  const names = [...tools.keys()].sort((left, right) => left.localeCompare(right));
  return {
    content: `Unknown tool "${call.name}". Available tools: ${names.join(", ") || "(none)"}.`,
    isError: true,
  };
}

function parseErrorCall(iteration: number): AgentToolCall {
  return {
    id: `tool-parse-error-${iteration + 1}`,
    name: "tool_parse_error",
    arguments: {},
  };
}

function parseErrorResult(error: unknown): AgentToolResult {
  return {
    content: `Malformed tool call: ${error instanceof Error ? error.message : String(error)}`,
    isError: true,
  };
}

async function executeTool(
  tool: AgentTool | undefined,
  call: AgentToolCall,
  context: AgentToolContext,
  tools: ReadonlyMap<string, AgentTool>,
): Promise<AgentToolResult> {
  if (tool === undefined) {
    return unknownToolResult(call, tools);
  }

  try {
    return normalizeToolOutput(await tool.execute(call.arguments, context));
  } catch (error) {
    return {
      content: error instanceof Error ? error.message : String(error),
      isError: true,
    };
  }
}

function assistantMessageFromResponse(response: AgentModelResponse): AgentMessage {
  return {
    role: "assistant",
    content: response.content,
    ...(response.reasoningContent === undefined
      ? {}
      : { reasoningContent: response.reasoningContent }),
  };
}

function modelResponseEvent(iteration: number, response: AgentModelResponse): AgentEvent {
  return {
    type: "model_response",
    iteration,
    content: response.content,
    ...(response.reasoningContent === undefined
      ? {}
      : { reasoningContent: response.reasoningContent }),
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

type StreamingToolCallState = {
  id?: string;
  name?: string;
  arguments: string;
};

type StreamingModelState = {
  content: string;
  reasoningContent: string;
  toolCalls: Map<number, StreamingToolCallState>;
};

function parseToolArguments(value: string): Record<string, unknown> {
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

function streamingToolCallAt(state: StreamingModelState, index: number): StreamingToolCallState {
  const existing = state.toolCalls.get(index);
  if (existing !== undefined) {
    return existing;
  }
  const created = { arguments: "" };
  state.toolCalls.set(index, created);
  return created;
}

function applyStreamEvent(
  state: StreamingModelState,
  event: AgentModelStreamEvent,
): AgentEvent | null {
  switch (event.type) {
    case "content_delta":
      state.content += event.contentDelta;
      return { type: "model_delta", iteration: 0, contentDelta: event.contentDelta };
    case "reasoning_delta":
      state.reasoningContent += event.reasoningContentDelta;
      return {
        type: "model_delta",
        iteration: 0,
        reasoningContentDelta: event.reasoningContentDelta,
      };
    case "tool_call_delta": {
      const call = streamingToolCallAt(state, event.index);
      if (event.id !== undefined && event.id.trim() !== "") {
        call.id = event.id;
      }
      if (event.nameDelta !== undefined && event.nameDelta !== "") {
        call.name = `${call.name ?? ""}${event.nameDelta}`;
      }
      if (event.argumentsDelta !== undefined) {
        call.arguments += event.argumentsDelta;
      }
      return null;
    }
  }
}

function streamingToolCalls(state: StreamingModelState): AgentToolCall[] | undefined {
  const calls = [...state.toolCalls.entries()]
    .sort(([left], [right]) => left - right)
    .map(([index, call]) => {
      if (call.name === undefined || call.name.trim() === "") {
        return null;
      }
      return {
        id: call.id ?? `tool-${index + 1}`,
        name: call.name,
        arguments: parseToolArguments(call.arguments),
      };
    })
    .filter((call): call is AgentToolCall => call !== null);
  return calls.length === 0 ? undefined : calls;
}

async function modelResponse(
  options: AgentRunOptions,
  request: {
    messages: readonly AgentMessage[];
    tools: readonly AgentTool[];
    iteration: number;
  },
): Promise<AgentModelResponse> {
  if (options.stream === false || options.model.stream === undefined) {
    return await options.model.complete(request);
  }

  const state: StreamingModelState = {
    content: "",
    reasoningContent: "",
    toolCalls: new Map(),
  };
  const stream = await options.model.stream(request);
  for await (const event of stream) {
    const visible = applyStreamEvent(state, event);
    if (visible !== null) {
      await emit(options.onEvent, { ...visible, iteration: request.iteration });
    }
  }

  const toolCalls = streamingToolCalls(state);
  return toolCalls === undefined
    ? {
        content: state.content,
        ...(state.reasoningContent.trim() === ""
          ? {}
          : { reasoningContent: state.reasoningContent.trim() }),
      }
    : {
        content: state.content,
        ...(state.reasoningContent.trim() === ""
          ? {}
          : { reasoningContent: state.reasoningContent.trim() }),
        toolCalls,
      };
}

/** Run one user-visible agent turn until a final answer or max-iteration stop. */
export async function runAgentTurn(options: AgentRunOptions): Promise<AgentRunResult> {
  const tools = toolMap(options.tools ?? []);
  const messages = [...options.messages];
  const toolCalls: AgentToolCall[] = [];
  const maxIterations = options.maxIterations ?? DEFAULT_MAX_ITERATIONS;
  const maxToolResultChars = options.maxToolResultChars ?? DEFAULT_MAX_TOOL_RESULT_CHARS;

  for (let iteration = 0; iteration < maxIterations; iteration += 1) {
    const response = await modelResponse(options, {
      messages,
      tools: [...tools.values()],
      iteration,
    });
    messages.push(assistantMessageFromResponse(response));
    await emit(options.onEvent, modelResponseEvent(iteration, response));

    let calls: readonly AgentToolCall[] = response.toolCalls ?? [];
    if (calls.length === 0 && tools.size > 0) {
      try {
        calls = parseToolCalls(response.content, { allowBareJson: true }).calls;
      } catch (error) {
        const call = parseErrorCall(iteration);
        const result = parseErrorResult(error);
        toolCalls.push(call);
        messages.push({
          role: "tool",
          name: call.name,
          toolCallId: call.id,
          content: result.content,
        });
        await emit(options.onEvent, { type: "tool_result", iteration, call, result });
        continue;
      }
    }

    if (calls.length === 0) {
      await emit(options.onEvent, { type: "final", iteration, content: response.content });
      return {
        messages,
        finalText: response.content,
        finishReason: "stop",
        iterations: iteration + 1,
        toolCalls,
      };
    }

    for (const call of calls) {
      toolCalls.push(call);
      await emit(options.onEvent, { type: "tool_call", iteration, call });
      const result = truncateToolResult(
        await executeTool(
          tools.get(call.name),
          call,
          { messages, iteration, toolCall: call },
          tools,
        ),
        maxToolResultChars,
      );
      messages.push({
        role: "tool",
        name: call.name,
        toolCallId: call.id,
        content: result.content,
      });
      await emit(options.onEvent, { type: "tool_result", iteration, call, result });
    }
  }

  return {
    messages,
    finalText: "",
    finishReason: "max_iterations",
    iterations: maxIterations,
    toolCalls,
  };
}
