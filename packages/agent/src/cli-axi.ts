import type { AgentRunResult } from "./types";

/** Structured CLI error classes with stable exit-code meaning. */
export type AgentCliErrorCode = "usage" | "runtime";

function quoteScalar(value: string): string {
  return JSON.stringify(value);
}

/** Format an agent CLI error for agent-readable stdout. */
export function formatAgentCliError(
  message: string,
  code: AgentCliErrorCode,
  help: readonly string[] = [],
): string {
  const lines = ["error:", `  code: ${code}`, `  message: ${quoteScalar(message)}`];
  if (help.length > 0) {
    lines.push(`help[${help.length}]:`);
    for (const item of help) {
      lines.push(`  ${quoteScalar(item)}`);
    }
  }
  return lines.join("\n");
}

/** Format a completed one-shot agent turn for agent-readable stdout. */
export function formatAgentRunResult(result: AgentRunResult, model?: string): string {
  const lines = [
    "agent_run:",
    ...(model === undefined ? [] : [`  model: ${quoteScalar(model)}`]),
    `  finish_reason: ${result.finishReason}`,
    `  iterations: ${result.iterations}`,
    `  tool_call_count: ${result.toolCalls.length}`,
    "assistant:",
    `  ${quoteScalar(result.finalText)}`,
  ];
  if (result.toolCalls.length > 0) {
    lines.push(`tool_calls[${result.toolCalls.length}]{id,name}:`);
    for (const call of result.toolCalls) {
      lines.push(`  ${quoteScalar(call.id)},${quoteScalar(call.name)}`);
    }
  }
  return lines.join("\n");
}
