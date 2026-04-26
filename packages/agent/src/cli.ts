#!/usr/bin/env bun

import { createOpenAIChatAgentModel, DEFAULT_AGENT_MAX_TOKENS } from "./chat-model";
import { createReadOnlyFileTools } from "./local-tools";
import { runAgentTurn } from "./loop";
import type { AgentEvent, AgentMessage, AgentModel, AgentTool } from "./types";

export type AgentCliOptions = {
  endpoint: string;
  model: string;
  cwd: string;
  apiKey?: string;
  maxTokens: number;
  temperature?: number;
  enableThinking?: boolean;
  stream: boolean;
  maxIterations: number;
  verbose: boolean;
};

export type AgentCliParseResult =
  | { kind: "agent"; options: AgentCliOptions }
  | { kind: "help"; exitCode: number; message?: string };

export type AgentCliRuntime = {
  prompt?: (message?: string) => string | null;
  log?: (message: string) => void;
  write?: (chunk: string) => void;
  model?: AgentModel;
  tools?: readonly AgentTool[];
};

type ParseState = {
  endpoint: string;
  model?: string;
  cwd: string;
  apiKey?: string;
  maxTokens: number;
  temperature?: number;
  enableThinking?: boolean;
  stream: boolean;
  maxIterations: number;
  verbose: boolean;
};

function readStringFlag(flag: string, value: string | undefined): string {
  if (value === undefined || value.trim() === "") {
    throw new Error(`Missing value for ${flag}.`);
  }
  return value;
}

function readNumberFlag(
  flag: string,
  value: string | undefined,
  isValid: (value: number) => boolean,
  description: string,
): number {
  const raw = readStringFlag(flag, value);
  const parsed = Number(raw);
  if (!Number.isFinite(parsed) || !isValid(parsed)) {
    throw new Error(`Expected ${flag} to be ${description}, got "${raw}".`);
  }
  return parsed;
}

function createParseState(): ParseState {
  return {
    endpoint: "http://127.0.0.1:8000",
    cwd: process.cwd(),
    maxTokens: DEFAULT_AGENT_MAX_TOKENS,
    stream: true,
    maxIterations: 8,
    verbose: false,
  };
}

function applyFlag(state: ParseState, argv: readonly string[], index: number): number {
  const arg = argv[index];
  switch (arg) {
    case "--endpoint":
      state.endpoint = readStringFlag(arg, argv[index + 1]);
      return index + 1;
    case "--model":
      state.model = readStringFlag(arg, argv[index + 1]);
      return index + 1;
    case "--cwd":
      state.cwd = readStringFlag(arg, argv[index + 1]);
      return index + 1;
    case "--api-key":
      state.apiKey = readStringFlag(arg, argv[index + 1]);
      return index + 1;
    case "--max-tokens":
      state.maxTokens = readNumberFlag(
        arg,
        argv[index + 1],
        (value) => Number.isInteger(value) && value > 0,
        "a positive integer",
      );
      return index + 1;
    case "--temperature":
      state.temperature = readNumberFlag(
        arg,
        argv[index + 1],
        (value) => value >= 0 && value <= 2,
        "a number between 0 and 2",
      );
      return index + 1;
    case "--greedy":
    case "--deterministic":
      state.temperature = 0;
      return index;
    case "--thinking":
      state.enableThinking = true;
      return index;
    case "--no-thinking":
      state.enableThinking = false;
      return index;
    case "--stream":
      state.stream = true;
      return index;
    case "--no-stream":
      state.stream = false;
      return index;
    case "--max-iterations":
      state.maxIterations = readNumberFlag(
        arg,
        argv[index + 1],
        (value) => Number.isInteger(value) && value > 0,
        "a positive integer",
      );
      return index + 1;
    case "--verbose":
      state.verbose = true;
      return index;
    default:
      throw new Error(`Unknown argument: ${arg ?? "<missing>"}`);
  }
}

function parseState(argv: readonly string[]): ParseState {
  const state = createParseState();
  for (let index = 0; index < argv.length; index += 1) {
    index = applyFlag(state, argv, index);
  }
  return state;
}

function stateToOptions(state: ParseState): AgentCliParseResult {
  if (state.model === undefined || state.model.trim() === "") {
    return { kind: "help", exitCode: 1, message: "Missing required --model <id>." };
  }
  return {
    kind: "agent",
    options: {
      endpoint: state.endpoint,
      model: state.model,
      cwd: state.cwd,
      ...(state.apiKey === undefined ? {} : { apiKey: state.apiKey }),
      maxTokens: state.maxTokens,
      ...(state.temperature === undefined ? {} : { temperature: state.temperature }),
      ...(state.enableThinking === undefined ? {} : { enableThinking: state.enableThinking }),
      stream: state.stream,
      maxIterations: state.maxIterations,
      verbose: state.verbose,
    },
  };
}

export function formatAgentUsage(): string {
  return [
    "Talk to an OpenAI-compatible local chat endpoint with read-only tools.",
    "",
    "Usage:",
    "  mlxts-agent --model <served-model-id> [options]",
    "",
    "Options:",
    "  --endpoint <url>          Base endpoint (default: http://127.0.0.1:8000)",
    "  --model <id>              Served model id",
    "  --cwd <path>              Directory exposed to read-only file tools (default: current directory)",
    "  --api-key <key>           Authorization bearer token",
    `  --max-tokens <n>          Max assistant tokens per loop step (default: ${DEFAULT_AGENT_MAX_TOKENS})`,
    "  --temperature <n>         Sampling temperature, 0 to 2 (default: model config)",
    "  --greedy                  Alias for --temperature 0",
    "  --deterministic           Alias for --temperature 0",
    "  --thinking                Ask compatible chat templates to enable thinking",
    "  --no-thinking             Ask compatible chat templates to disable thinking",
    "  --stream                  Use chat-completion streaming transport (default)",
    "  --no-stream               Use non-streaming chat completions",
    "  --max-iterations <n>      Max model/tool loop steps per user turn (default: 8)",
    "  --verbose                 Enable verbose fetch diagnostics",
    "  --help                    Show this help",
  ].join("\n");
}

export function parseAgentArgs(argv: readonly string[]): AgentCliParseResult {
  if (argv.includes("--help")) {
    return { kind: "help", exitCode: 0 };
  }
  try {
    return stateToOptions(parseState(argv));
  } catch (error) {
    return {
      kind: "help",
      exitCode: 1,
      message: error instanceof Error ? error.message : String(error),
    };
  }
}

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

function formatCliSection(title: string, content: string): string {
  return `\n${title}\n${indentBlock(content)}`;
}

function formatToolArguments(args: Record<string, unknown>): string {
  return JSON.stringify(args, null, 2);
}

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

function defaultWrite(chunk: string): void {
  process.stdout.write(chunk);
}

function indentedDelta(content: string): string {
  return content.replaceAll("\n", "\n  ");
}

function beginStreamSection(
  state: AgentEventPrinterState,
  section: "assistant" | "thinking",
  write: (chunk: string) => void,
): void {
  if (state.activeStreamSection === section) {
    return;
  }
  write(state.activeStreamSection === null ? `\n[${section}]\n  ` : `\n\n[${section}]\n  `);
  state.activeStreamSection = section;
}

function finishStreamSection(state: AgentEventPrinterState, write: (chunk: string) => void): void {
  if (state.activeStreamSection === null) {
    return;
  }
  write("\n");
  state.activeStreamSection = null;
}

function printStreamDelta(
  event: Extract<AgentEvent, { type: "model_delta" }>,
  state: AgentEventPrinterState,
  write: (chunk: string) => void,
): void {
  if (event.reasoningContentDelta !== undefined) {
    beginStreamSection(state, "thinking", write);
    state.streamedReasoning = true;
    write(indentedDelta(event.reasoningContentDelta));
  }
  if (event.contentDelta !== undefined) {
    beginStreamSection(state, "assistant", write);
    state.streamedContent = true;
    write(indentedDelta(event.contentDelta));
  }
}

function printAggregateReasoning(
  event: Extract<AgentEvent, { type: "model_response" }>,
  state: AgentEventPrinterState,
  log: (message: string) => void,
  write: (chunk: string) => void,
): void {
  finishStreamSection(state, write);
  if (
    !state.streamedReasoning &&
    event.reasoningContent !== undefined &&
    event.reasoningContent.trim() !== ""
  ) {
    log(formatCliSection("[thinking]", event.reasoningContent));
  }
}

function printFinalAnswer(
  event: Extract<AgentEvent, { type: "final" }>,
  state: AgentEventPrinterState,
  log: (message: string) => void,
  write: (chunk: string) => void,
): void {
  if (state.streamedContent) {
    finishStreamSection(state, write);
    write("\n");
    return;
  }
  finishStreamSection(state, write);
  log(`${formatCliSection("[assistant]", event.content)}\n`);
}

export function createAgentEventPrinter(
  log: (message: string) => void = console.log,
  write?: (chunk: string) => void,
): (event: AgentEvent) => void {
  const writeChunk = write ?? (log === console.log ? defaultWrite : log);
  const state: AgentEventPrinterState = {
    iteration: -1,
    streamedContent: false,
    streamedReasoning: false,
    activeStreamSection: null,
  };

  return (event) => {
    resetPrinterState(state, event.iteration);
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

export async function runAgentRepl(
  options: AgentCliOptions,
  runtime: AgentCliRuntime = {},
): Promise<void> {
  const model = runtime.model ?? createOpenAIChatAgentModel(options);
  const tools = runtime.tools ?? createReadOnlyFileTools({ root: options.cwd });
  const readInput = runtime.prompt ?? prompt;
  const log = runtime.log ?? console.log;
  const write = runtime.write ?? (runtime.log === undefined ? defaultWrite : log);
  const printEvent = createAgentEventPrinter(log, write);
  const messages: AgentMessage[] = [];

  log(`Talking to ${options.model} at ${options.endpoint}. Type "exit" to quit.`);
  while (true) {
    const input = readInput("> ");
    if (input === null || input.trim() === "exit" || input.trim() === "quit") {
      return;
    }
    if (input.trim() === "") {
      continue;
    }

    messages.push({ role: "user", content: input });
    const result = await runAgentTurn({
      model,
      tools,
      messages,
      stream: options.stream,
      maxIterations: options.maxIterations,
      onEvent: (event) => printEvent(event),
    });
    if (result.finishReason === "max_iterations") {
      log(
        formatCliSection(
          "[agent]",
          `Stopped after ${result.iterations} iteration(s) without a final answer. Increase --max-iterations or narrow the request.`,
        ),
      );
    }
    messages.splice(0, messages.length, ...result.messages);
  }
}

export async function main(argv: readonly string[] = Bun.argv.slice(2)): Promise<void> {
  const parsed = parseAgentArgs(argv);
  if (parsed.kind === "help") {
    if (parsed.message !== undefined) {
      console.error(parsed.message);
      console.error("");
    }
    console.error(formatAgentUsage());
    process.exit(parsed.exitCode);
  }

  await runAgentRepl(parsed.options);
}

if (import.meta.main) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exit(1);
  });
}
