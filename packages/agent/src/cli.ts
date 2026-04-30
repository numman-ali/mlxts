#!/usr/bin/env bun

import { createOpenAIChatAgentModel, DEFAULT_AGENT_MAX_TOKENS } from "./chat-model";
import { formatAgentCliError, formatAgentRunResult } from "./cli-axi";
import { createAgentEventPrinter, formatCliSection, printAgentEvent } from "./cli-printer";
import { formatAgentUsage } from "./cli-usage";
import { createReadOnlyFileTools } from "./local-tools";
import { runAgentTurn } from "./loop";
import type { AgentMessage, AgentModel, AgentTool } from "./types";

export type AgentCliOptions = {
  endpoint: string;
  model: string;
  prompt?: string;
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
  command: "repl" | "run";
  endpoint: string;
  model?: string;
  prompt?: string;
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
    command: "repl",
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
    case "--prompt":
      state.prompt = readStringFlag(arg, argv[index + 1]);
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
  const first = argv[0];
  const startIndex = first === "run" ? 1 : 0;
  if (first === "run") {
    state.command = "run";
  } else if (first !== undefined && !first.startsWith("-")) {
    throw new Error(`Unknown command: ${first}`);
  }
  for (let index = startIndex; index < argv.length; index += 1) {
    index = applyFlag(state, argv, index);
  }
  return state;
}

function stateToOptions(state: ParseState): AgentCliParseResult {
  if (state.model === undefined || state.model.trim() === "") {
    return { kind: "help", exitCode: 2, message: "Missing required --model <id>." };
  }
  if (state.command === "run" && (state.prompt === undefined || state.prompt.trim() === "")) {
    return { kind: "help", exitCode: 2, message: "Missing required --prompt <text> for run." };
  }
  return {
    kind: "agent",
    options: {
      endpoint: state.endpoint,
      model: state.model,
      ...(state.prompt === undefined ? {} : { prompt: state.prompt }),
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

export function parseAgentArgs(argv: readonly string[]): AgentCliParseResult {
  if (argv.includes("--help")) {
    return { kind: "help", exitCode: 0 };
  }
  try {
    return stateToOptions(parseState(argv));
  } catch (error) {
    return {
      kind: "help",
      exitCode: 2,
      message: error instanceof Error ? error.message : String(error),
    };
  }
}

function defaultWrite(chunk: string): void {
  process.stdout.write(chunk);
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

/** Run one finite agent turn and return AXI-shaped stdout. */
export async function runAgentOnce(
  options: AgentCliOptions,
  runtime: AgentCliRuntime = {},
): Promise<string> {
  if (options.prompt === undefined || options.prompt.trim() === "") {
    throw new Error("runAgentOnce requires a non-empty prompt.");
  }
  const model = runtime.model ?? createOpenAIChatAgentModel(options);
  const tools = runtime.tools ?? createReadOnlyFileTools({ root: options.cwd });
  const result = await runAgentTurn({
    model,
    tools,
    messages: [{ role: "user", content: options.prompt }],
    stream: options.stream,
    maxIterations: options.maxIterations,
  });
  return formatAgentRunResult(result, options.model);
}

function nonTtyUsageError(): string {
  return formatAgentCliError(
    "mlxts-agent requires run --prompt in non-TTY environments.",
    "usage",
    ['mlxts-agent run --model <served-model-id> --prompt "..." [options]'],
  );
}

/** Process-level CLI completion status. */
export type AgentCliRunResult = {
  exitCode: number;
};

/** Injectable process context for CLI tests and embedders. */
export type AgentCliProcessRuntime = AgentCliRuntime & {
  isTTY?: boolean;
  stdout?: (message: string) => void;
};

/** Run the process-level CLI contract without calling process.exit. */
export async function runAgentCli(
  argv: readonly string[] = Bun.argv.slice(2),
  runtime: AgentCliProcessRuntime = {},
): Promise<AgentCliRunResult> {
  const stdout = runtime.stdout ?? console.log;
  const parsed = parseAgentArgs(argv);
  if (parsed.kind === "help") {
    if (parsed.message !== undefined) {
      stdout(
        formatAgentCliError(parsed.message, "usage", [
          'mlxts-agent run --model <served-model-id> --prompt "..." [options]',
        ]),
      );
    } else {
      stdout(formatAgentUsage());
    }
    return { exitCode: parsed.exitCode };
  }

  try {
    if (parsed.options.prompt !== undefined) {
      stdout(await runAgentOnce(parsed.options, runtime));
      return { exitCode: 0 };
    }
    if (runtime.isTTY !== true) {
      stdout(nonTtyUsageError());
      return { exitCode: 2 };
    }
    await runAgentRepl(parsed.options, runtime);
    return { exitCode: 0 };
  } catch (error) {
    stdout(
      formatAgentCliError(error instanceof Error ? error.message : String(error), "runtime", [
        "Check the endpoint, served model id, and local tool permissions.",
      ]),
    );
    return { exitCode: 1 };
  }
}

export async function main(argv: readonly string[] = Bun.argv.slice(2)): Promise<void> {
  const result = await runAgentCli(argv, { isTTY: process.stdin.isTTY === true });
  if (result.exitCode !== 0) {
    process.exit(result.exitCode);
  }
}

if (import.meta.main) {
  main().catch((error) => {
    console.log(
      formatAgentCliError(error instanceof Error ? error.message : String(error), "runtime"),
    );
    process.exit(1);
  });
}

export {
  createAgentEventPrinter,
  formatAgentCliError,
  formatAgentRunResult,
  formatAgentUsage,
  printAgentEvent,
};
