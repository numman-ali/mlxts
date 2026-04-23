/**
 * Public contracts for agent loops, model adapters, and tools.
 * @module
 */

export type AgentRole = "system" | "user" | "assistant" | "tool";

export type AgentMessage = {
  role: AgentRole;
  content: string;
  reasoningContent?: string;
  name?: string;
  toolCallId?: string;
};

export type AgentJsonSchema = {
  type: "object";
  description?: string;
  properties?: Record<string, unknown>;
  required?: readonly string[];
  additionalProperties?: boolean;
};

export type AgentToolCall = {
  id: string;
  name: string;
  arguments: Record<string, unknown>;
};

export type AgentToolResult = {
  content: string;
  isError?: boolean;
};

export type AgentToolOutput = string | AgentToolResult;

export type AgentToolContext = {
  messages: readonly AgentMessage[];
  iteration: number;
  toolCall: AgentToolCall;
};

export type AgentTool = {
  name: string;
  description: string;
  parameters?: AgentJsonSchema;
  execute(
    args: Record<string, unknown>,
    context: AgentToolContext,
  ): AgentToolOutput | Promise<AgentToolOutput>;
};

export type AgentModelRequest = {
  messages: readonly AgentMessage[];
  tools: readonly AgentTool[];
  iteration: number;
};

export type AgentModelResponse = {
  content: string;
  reasoningContent?: string;
  toolCalls?: readonly AgentToolCall[];
};

export type AgentModelStreamEvent =
  | {
      type: "content_delta";
      contentDelta: string;
    }
  | {
      type: "reasoning_delta";
      reasoningContentDelta: string;
    }
  | {
      type: "tool_call_delta";
      index: number;
      id?: string;
      nameDelta?: string;
      argumentsDelta?: string;
    };

export type AgentModel = {
  complete(request: AgentModelRequest): AgentModelResponse | Promise<AgentModelResponse>;
  stream?(
    request: AgentModelRequest,
  ): AsyncIterable<AgentModelStreamEvent> | Promise<AsyncIterable<AgentModelStreamEvent>>;
};

export type AgentEvent =
  | {
      type: "model_delta";
      iteration: number;
      contentDelta?: string;
      reasoningContentDelta?: string;
    }
  | {
      type: "model_response";
      iteration: number;
      content: string;
      reasoningContent?: string;
    }
  | {
      type: "tool_call";
      iteration: number;
      call: AgentToolCall;
    }
  | {
      type: "tool_result";
      iteration: number;
      call: AgentToolCall;
      result: AgentToolResult;
    }
  | {
      type: "final";
      iteration: number;
      content: string;
    };

export type AgentRunOptions = {
  model: AgentModel;
  messages: readonly AgentMessage[];
  tools?: readonly AgentTool[];
  stream?: boolean;
  maxIterations?: number;
  maxToolResultChars?: number;
  onEvent?: (event: AgentEvent) => void | Promise<void>;
};

export type AgentFinishReason = "stop" | "max_iterations";

export type AgentRunResult = {
  messages: AgentMessage[];
  finalText: string;
  finishReason: AgentFinishReason;
  iterations: number;
  toolCalls: AgentToolCall[];
};
