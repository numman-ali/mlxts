/**
 * Agent loop orchestration.
 * @module
 */

import { parseToolCalls } from "./tool-calls";
import type {
  AgentEvent,
  AgentMessage,
  AgentModelResponse,
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

/** Run one user-visible agent turn until a final answer or max-iteration stop. */
export async function runAgentTurn(options: AgentRunOptions): Promise<AgentRunResult> {
  const tools = toolMap(options.tools ?? []);
  const messages = [...options.messages];
  const toolCalls: AgentToolCall[] = [];
  const maxIterations = options.maxIterations ?? DEFAULT_MAX_ITERATIONS;
  const maxToolResultChars = options.maxToolResultChars ?? DEFAULT_MAX_TOOL_RESULT_CHARS;

  for (let iteration = 0; iteration < maxIterations; iteration += 1) {
    const response = await options.model.complete({
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
