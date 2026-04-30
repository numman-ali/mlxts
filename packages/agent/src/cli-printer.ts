import type { AgentEvent } from "./types";

function indentBlock(content: string): string {
  const trimmed = content.trimEnd();
  if (trimmed === "") {
    return "  (empty)";
  }
  return trimmed
    .split("\n")
    .map((line) => `  ${line}`)
    .join("\n");
}

/** Format an interactive transcript section. */
export function formatCliSection(title: string, content: string): string {
  return `\n${title}\n${indentBlock(content)}`;
}

function formatToolArguments(args: Record<string, unknown>): string {
  return JSON.stringify(args, null, 2);
}

/** Print a non-streamed interactive agent event. */
export function printAgentEvent(
  event: AgentEvent,
  log: (message: string) => void = console.log,
): void {
  switch (event.type) {
    case "model_delta":
      if (event.reasoningContentDelta !== undefined) {
        log(formatCliSection("[thinking]", event.reasoningContentDelta));
      }
      if (event.contentDelta !== undefined) {
        log(formatCliSection("[assistant]", event.contentDelta));
      }
      return;
    case "tool_call":
      log(
        formatCliSection(
          `[tool call] ${event.call.name}`,
          formatToolArguments(event.call.arguments),
        ),
      );
      return;
    case "tool_result":
      log(
        formatCliSection(
          `[tool result] ${event.call.name}`,
          `${event.result.isError === true ? "error: " : ""}${event.result.content}`,
        ),
      );
      return;
    case "final":
      log(`${formatCliSection("[assistant]", event.content)}\n`);
      return;
    case "model_response":
      if (event.reasoningContent !== undefined && event.reasoningContent.trim() !== "") {
        log(formatCliSection("[thinking]", event.reasoningContent));
      }
      return;
  }
}

type AgentEventPrinterState = {
  iteration: number;
  streamedContent: boolean;
  streamedReasoning: boolean;
  activeStreamSection: "assistant" | "thinking" | null;
};

function resetPrinterState(state: AgentEventPrinterState, eventIteration: number): void {
  if (eventIteration === state.iteration) {
    return;
  }
  state.iteration = eventIteration;
  state.streamedContent = false;
  state.streamedReasoning = false;
  state.activeStreamSection = null;
}

function finishStreamSection(
  state: AgentEventPrinterState,
  writeChunk: (chunk: string) => void,
): void {
  if (state.activeStreamSection !== null) {
    writeChunk("\n");
    state.activeStreamSection = null;
  }
}

function beginStreamSection(
  state: AgentEventPrinterState,
  section: "assistant" | "thinking",
  writeChunk: (chunk: string) => void,
): void {
  if (state.activeStreamSection === section) {
    return;
  }
  finishStreamSection(state, writeChunk);
  const title = section === "assistant" ? "[assistant]" : "[thinking]";
  writeChunk(`\n${title}\n  `);
  state.activeStreamSection = section;
}

function printStreamDelta(
  event: Extract<AgentEvent, { type: "model_delta" }>,
  state: AgentEventPrinterState,
  writeChunk: (chunk: string) => void,
): void {
  resetPrinterState(state, event.iteration);
  if (event.reasoningContentDelta !== undefined) {
    beginStreamSection(state, "thinking", writeChunk);
    writeChunk(event.reasoningContentDelta);
    state.streamedReasoning = true;
  }
  if (event.contentDelta !== undefined) {
    beginStreamSection(state, "assistant", writeChunk);
    writeChunk(event.contentDelta);
    state.streamedContent = true;
  }
}

function printAggregateReasoning(
  event: Extract<AgentEvent, { type: "model_response" }>,
  state: AgentEventPrinterState,
  log: (message: string) => void,
  writeChunk: (chunk: string) => void,
): void {
  resetPrinterState(state, event.iteration);
  if (
    !state.streamedReasoning &&
    event.reasoningContent !== undefined &&
    event.reasoningContent.trim() !== ""
  ) {
    finishStreamSection(state, writeChunk);
    log(formatCliSection("[thinking]", event.reasoningContent));
  }
}

function printFinalAnswer(
  event: Extract<AgentEvent, { type: "final" }>,
  state: AgentEventPrinterState,
  log: (message: string) => void,
  writeChunk: (chunk: string) => void,
): void {
  resetPrinterState(state, event.iteration);
  if (state.streamedContent) {
    finishStreamSection(state, writeChunk);
    writeChunk("\n");
    return;
  }
  finishStreamSection(state, writeChunk);
  log(`${formatCliSection("[assistant]", event.content)}\n`);
}

/** Create an interactive transcript printer for streamed and aggregate events. */
export function createAgentEventPrinter(
  log: (message: string) => void = console.log,
  write?: (chunk: string) => void,
): (event: AgentEvent) => void {
  const writeChunk =
    write ??
    (log === console.log
      ? (chunk: string) => {
          process.stdout.write(chunk);
        }
      : log);
  const state: AgentEventPrinterState = {
    iteration: -1,
    streamedContent: false,
    streamedReasoning: false,
    activeStreamSection: null,
  };
  return (event) => {
    switch (event.type) {
      case "model_delta":
        printStreamDelta(event, state, writeChunk);
        return;
      case "model_response":
        printAggregateReasoning(event, state, log, writeChunk);
        return;
      case "final":
        printFinalAnswer(event, state, log, writeChunk);
        return;
      default:
        finishStreamSection(state, writeChunk);
        printAgentEvent(event, log);
    }
  };
}
