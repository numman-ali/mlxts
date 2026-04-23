/**
 * Agent loop primitives for mlxts.
 * @module
 */

export {
  type ChatAgentModelOptions,
  createOpenAIChatAgentModel,
  DEFAULT_AGENT_MAX_TOKENS,
} from "./chat-model";
export {
  createReadOnlyFileTools,
  type ReadOnlyFileToolsOptions,
} from "./local-tools";
export { runAgentTurn } from "./loop";
export {
  AgentToolCallParseError,
  formatToolInstructions,
  type ParsedToolCalls,
  parseToolCalls,
} from "./tool-calls";
export type {
  AgentEvent,
  AgentFinishReason,
  AgentJsonSchema,
  AgentMessage,
  AgentModel,
  AgentModelRequest,
  AgentModelResponse,
  AgentRole,
  AgentRunOptions,
  AgentRunResult,
  AgentTool,
  AgentToolCall,
  AgentToolContext,
  AgentToolOutput,
  AgentToolResult,
} from "./types";
