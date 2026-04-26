import { describe, expect, test } from "bun:test";

import {
  createAgentEventPrinter,
  formatAgentUsage,
  parseAgentArgs,
  printAgentEvent,
  runAgentRepl,
} from "./cli";
import type { AgentModel, AgentTool } from "./types";

describe("agent CLI args", () => {
  test("parses chat endpoint options", () => {
    expect(
      parseAgentArgs([
        "--model",
        "qwen-local",
        "--endpoint",
        "http://localhost:8000",
        "--cwd",
        ".",
        "--api-key",
        "secret",
        "--max-tokens",
        "256",
        "--temperature",
        "0.7",
        "--max-iterations",
        "4",
        "--no-thinking",
        "--verbose",
      ]),
    ).toEqual({
      kind: "agent",
      options: {
        model: "qwen-local",
        endpoint: "http://localhost:8000",
        cwd: ".",
        apiKey: "secret",
        maxTokens: 256,
        temperature: 0.7,
        enableThinking: false,
        stream: true,
        maxIterations: 4,
        verbose: true,
      },
    });
  });

  test("defaults agent turns to model-native sampling", () => {
    expect(parseAgentArgs(["--model", "qwen-local"])).toEqual({
      kind: "agent",
      options: {
        model: "qwen-local",
        endpoint: "http://127.0.0.1:8000",
        cwd: process.cwd(),
        maxTokens: 512,
        stream: true,
        maxIterations: 8,
        verbose: false,
      },
    });
  });

  test("supports explicit deterministic and thinking controls", () => {
    expect(parseAgentArgs(["--model", "qwen-local", "--greedy", "--thinking"])).toEqual({
      kind: "agent",
      options: {
        model: "qwen-local",
        endpoint: "http://127.0.0.1:8000",
        cwd: process.cwd(),
        maxTokens: 512,
        temperature: 0,
        enableThinking: true,
        stream: true,
        maxIterations: 8,
        verbose: false,
      },
    });
    expect(parseAgentArgs(["--model", "qwen-local", "--no-stream"])).toMatchObject({
      kind: "agent",
      options: { stream: false },
    });
  });

  test("returns help for missing model and invalid options", () => {
    expect(parseAgentArgs([])).toMatchObject({
      kind: "help",
      exitCode: 1,
      message: "Missing required --model <id>.",
    });
    expect(parseAgentArgs(["--help"])).toEqual({ kind: "help", exitCode: 0 });
    expect(parseAgentArgs(["--model", "qwen", "--max-tokens", "0"])).toMatchObject({
      kind: "help",
      exitCode: 1,
    });
    expect(parseAgentArgs(["--model"])).toMatchObject({
      kind: "help",
      message: "Missing value for --model.",
    });
    expect(parseAgentArgs(["--unknown"])).toMatchObject({
      kind: "help",
      message: "Unknown argument: --unknown",
    });
    expect(formatAgentUsage()).toContain("mlxts-agent");
  });

  test("runs a prompt-driven REPL loop with injectable model and tools", async () => {
    const logs: string[] = [];
    const inputs = ["read it", "exit"];
    const model: AgentModel = {
      complete(request) {
        if (request.messages.some((message) => message.role === "tool")) {
          return { content: "final answer" };
        }
        return { content: '<tool_call>{"name":"lookup","arguments":{}}</tool_call>' };
      },
    };
    const tool: AgentTool = {
      name: "lookup",
      description: "Lookup",
      execute() {
        return "tool output";
      },
    };

    await runAgentRepl(
      {
        model: "qwen-local",
        endpoint: "http://localhost:8000",
        cwd: ".",
        maxTokens: 32,
        stream: true,
        maxIterations: 4,
        verbose: false,
      },
      {
        model,
        tools: [tool],
        prompt() {
          return inputs.shift() ?? "exit";
        },
        log(message) {
          logs.push(message);
        },
      },
    );

    expect(logs.join("\n")).toContain("[tool call] lookup");
    expect(logs.join("\n")).toContain("[tool result] lookup");
    expect(logs.join("\n")).toContain("final answer");
  });

  test("keeps conversation state across multiple REPL turns", async () => {
    const inputs = ["first question", "follow up", "exit"];
    const seenUserCounts: number[] = [];
    const model: AgentModel = {
      complete(request) {
        seenUserCounts.push(request.messages.filter((message) => message.role === "user").length);
        return {
          content: `answer ${seenUserCounts.length}`,
          reasoningContent: `reason ${seenUserCounts.length}`,
        };
      },
    };
    const logs: string[] = [];

    await runAgentRepl(
      {
        model: "qwen-local",
        endpoint: "http://localhost:8000",
        cwd: ".",
        maxTokens: 32,
        stream: true,
        maxIterations: 4,
        verbose: false,
      },
      {
        model,
        tools: [],
        prompt() {
          return inputs.shift() ?? "exit";
        },
        log(message) {
          logs.push(message);
        },
      },
    );

    expect(seenUserCounts).toEqual([1, 2]);
    expect(logs.join("\n")).toContain("[thinking]\n  reason 1");
    expect(logs.join("\n")).toContain("[thinking]\n  reason 2");
    expect(logs.join("\n")).toContain("[assistant]\n  answer 2");
  });

  test("prints streamed assistant content without duplicating the final answer", async () => {
    const inputs = ["hello", "exit"];
    const logs: string[] = [];
    let terminal = "";
    const model: AgentModel = {
      complete() {
        throw new Error("stream should be used");
      },
      async *stream() {
        yield { type: "reasoning_delta", reasoningContentDelta: "Say hello." };
        yield { type: "content_delta", contentDelta: "Hel" };
        yield { type: "content_delta", contentDelta: "lo" };
      },
    };

    await runAgentRepl(
      {
        model: "qwen-local",
        endpoint: "http://localhost:8000",
        cwd: ".",
        maxTokens: 32,
        stream: true,
        maxIterations: 4,
        verbose: false,
      },
      {
        model,
        tools: [],
        prompt() {
          return inputs.shift() ?? "exit";
        },
        log(message) {
          logs.push(message);
        },
        write(chunk) {
          terminal += chunk;
        },
      },
    );

    expect(terminal).toContain("[thinking]\n  Say hello.");
    expect(terminal).toContain("[assistant]\n  Hello");
    expect(terminal.match(/\[assistant\]/g)).toHaveLength(1);
    expect(terminal).not.toContain("Hel\n  lo");
  });

  test("honors non-streaming REPL mode even when the model can stream", async () => {
    const inputs = ["hello", "exit"];
    const calls: string[] = [];
    const logs: string[] = [];
    const model: AgentModel = {
      complete() {
        calls.push("complete");
        return { content: "whole answer" };
      },
      async *stream() {
        calls.push("stream");
        yield { type: "content_delta", contentDelta: "streamed answer" };
      },
    };

    await runAgentRepl(
      {
        model: "qwen-local",
        endpoint: "http://localhost:8000",
        cwd: ".",
        maxTokens: 32,
        stream: false,
        maxIterations: 4,
        verbose: false,
      },
      {
        model,
        tools: [],
        prompt() {
          return inputs.shift() ?? "exit";
        },
        log(message) {
          logs.push(message);
        },
      },
    );

    expect(calls).toEqual(["complete"]);
    expect(logs.join("\n")).toContain("[assistant]\n  whole answer");
  });

  test("prints a clear notice when a turn exhausts max iterations", async () => {
    const logs: string[] = [];
    const inputs = ["loop", "exit"];

    await runAgentRepl(
      {
        model: "qwen-local",
        endpoint: "http://localhost:8000",
        cwd: ".",
        maxTokens: 32,
        stream: true,
        maxIterations: 1,
        verbose: false,
      },
      {
        model: {
          complete() {
            return { content: '<tool_call>{"name":"lookup","arguments":{}}</tool_call>' };
          },
        },
        tools: [
          {
            name: "lookup",
            description: "Lookup",
            execute() {
              return "tool output";
            },
          },
        ],
        prompt() {
          return inputs.shift() ?? "exit";
        },
        log(message) {
          logs.push(message);
        },
      },
    );

    expect(logs.join("\n")).toContain("[agent]\n  Stopped after 1 iteration");
  });

  test("ignores blank input and accepts quit", async () => {
    const logs: string[] = [];
    const inputs = ["", "quit"];

    await runAgentRepl(
      {
        model: "qwen-local",
        endpoint: "http://localhost:8000",
        cwd: ".",
        maxTokens: 32,
        stream: true,
        maxIterations: 4,
        verbose: false,
      },
      {
        model: {
          complete() {
            throw new Error("blank input should not call the model");
          },
        },
        tools: [],
        prompt() {
          return inputs.shift() ?? "quit";
        },
        log(message) {
          logs.push(message);
        },
      },
    );

    expect(logs).toEqual(['Talking to qwen-local at http://localhost:8000. Type "exit" to quit.']);
  });

  test("prints visible agent events", () => {
    const logs: string[] = [];
    const log = (message: string) => logs.push(message);

    printAgentEvent(
      { type: "model_response", iteration: 0, content: "hidden", reasoningContent: "thinking" },
      log,
    );
    printAgentEvent(
      { type: "tool_call", iteration: 0, call: { id: "1", name: "lookup", arguments: {} } },
      log,
    );
    printAgentEvent(
      {
        type: "tool_result",
        iteration: 0,
        call: { id: "1", name: "lookup", arguments: {} },
        result: { content: "nope", isError: true },
      },
      log,
    );
    printAgentEvent({ type: "final", iteration: 0, content: "done" }, log);

    expect(logs).toEqual([
      "\n[thinking]\n  thinking",
      "\n[tool call] lookup\n  {}",
      "\n[tool result] lookup\n  error: nope",
      "\n[assistant]\n  done\n",
    ]);
  });

  test("prints streamed deltas as continuous terminal text", () => {
    const logs: string[] = [];
    let terminal = "";
    const print = createAgentEventPrinter(
      (message) => logs.push(message),
      (chunk) => {
        terminal += chunk;
      },
    );

    print({ type: "model_delta", iteration: 0, reasoningContentDelta: "The user is ask" });
    print({ type: "model_delta", iteration: 0, reasoningContentDelta: "ing how I am." });
    print({ type: "model_delta", iteration: 0, contentDelta: "I'm doin" });
    print({ type: "model_delta", iteration: 0, contentDelta: "g well." });
    print({ type: "final", iteration: 0, content: "I'm doing well." });

    expect(logs).toEqual([]);
    expect(terminal).toBe(
      "\n[thinking]\n  The user is asking how I am.\n\n[assistant]\n  I'm doing well.\n\n",
    );
  });
});
