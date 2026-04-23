import { describe, expect, test } from "bun:test";

import { runAgentTurn } from "./loop";
import type { AgentModel, AgentTool } from "./types";

describe("runAgentTurn", () => {
  test("returns final answer when no tool is requested", async () => {
    const model: AgentModel = {
      complete() {
        return { content: "hello" };
      },
    };

    const result = await runAgentTurn({
      model,
      messages: [{ role: "user", content: "hi" }],
    });

    expect(result.finalText).toBe("hello");
    expect(result.finishReason).toBe("stop");
    expect(result.iterations).toBe(1);
  });

  test("executes a tool, appends observation, and continues", async () => {
    const seenToolMessages: string[] = [];
    const model: AgentModel = {
      complete(request) {
        const toolMessage = request.messages.find((message) => message.role === "tool");
        if (toolMessage === undefined) {
          return {
            content:
              '<tool_call>{"id":"call-1","name":"lookup","arguments":{"topic":"mlxts"}}</tool_call>',
          };
        }
        seenToolMessages.push(toolMessage.content);
        return { content: `final: ${toolMessage.content}` };
      },
    };
    const tool: AgentTool = {
      name: "lookup",
      description: "Lookup a topic",
      execute(args) {
        return `value:${args.topic}`;
      },
    };

    const result = await runAgentTurn({
      model,
      tools: [tool],
      messages: [{ role: "user", content: "lookup mlxts" }],
    });

    expect(seenToolMessages).toEqual(["value:mlxts"]);
    expect(result.finalText).toBe("final: value:mlxts");
    expect(result.toolCalls).toEqual([
      { id: "call-1", name: "lookup", arguments: { topic: "mlxts" } },
    ]);
  });

  test("preserves reasoning across multi-step tool turns", async () => {
    const events: string[] = [];
    const model: AgentModel = {
      complete(request) {
        const priorAssistant = request.messages.find((message) => message.role === "assistant");
        if (priorAssistant === undefined) {
          return {
            content:
              "<tool_call><function=lookup><parameter=topic>mlxts</parameter></function></tool_call>",
            reasoningContent: "I should inspect the project before answering.",
          };
        }
        expect(priorAssistant.reasoningContent).toBe(
          "I should inspect the project before answering.",
        );
        return {
          content: "final: mlxts is a local Apple Silicon ML stack.",
          reasoningContent: "The tool result gives the project context.",
        };
      },
    };
    const tool: AgentTool = {
      name: "lookup",
      description: "Lookup a topic",
      execute(args) {
        return `topic:${args.topic}`;
      },
    };

    const result = await runAgentTurn({
      model,
      tools: [tool],
      messages: [{ role: "user", content: "Explain mlxts" }],
      onEvent(event) {
        if (event.type === "model_response" && event.reasoningContent !== undefined) {
          events.push(`thinking:${event.reasoningContent}`);
        }
        if (event.type === "tool_call") {
          events.push(`tool:${event.call.name}`);
        }
        if (event.type === "final") {
          events.push(`final:${event.content}`);
        }
      },
    });

    expect(events).toEqual([
      "thinking:I should inspect the project before answering.",
      "tool:lookup",
      "thinking:The tool result gives the project context.",
      "final:final: mlxts is a local Apple Silicon ML stack.",
    ]);
    expect(result.messages.filter((message) => message.role === "assistant")).toEqual([
      {
        role: "assistant",
        content:
          "<tool_call><function=lookup><parameter=topic>mlxts</parameter></function></tool_call>",
        reasoningContent: "I should inspect the project before answering.",
      },
      {
        role: "assistant",
        content: "final: mlxts is a local Apple Silicon ML stack.",
        reasoningContent: "The tool result gives the project context.",
      },
    ]);
  });

  test("turns unknown tools, tool exceptions, and malformed calls into observations", async () => {
    let step = 0;
    const model: AgentModel = {
      complete(request) {
        step += 1;
        if (step === 1) {
          return { content: '<tool_call>{"name":"missing","arguments":{}}</tool_call>' };
        }
        if (step === 2) {
          expect(request.messages.at(-1)?.content).toContain("Unknown tool");
          return { content: '<tool_call>{"name":"broken","arguments":{}}</tool_call>' };
        }
        if (step === 3) {
          expect(request.messages.at(-1)?.content).toBe("boom");
          return { content: '<tool_call>{"name":"broken","arguments":[]}</tool_call>' };
        }
        expect(request.messages.at(-1)?.content).toContain("Malformed tool call");
        return { content: "recovered" };
      },
    };
    const broken: AgentTool = {
      name: "broken",
      description: "Throw",
      execute() {
        throw new Error("boom");
      },
    };

    const result = await runAgentTurn({
      model,
      tools: [broken],
      messages: [{ role: "user", content: "test" }],
    });

    expect(result.finalText).toBe("recovered");
  });

  test("guards max iterations and truncates tool results", async () => {
    const model: AgentModel = {
      complete() {
        return { content: '<tool_call>{"name":"long","arguments":{}}</tool_call>' };
      },
    };
    const tool: AgentTool = {
      name: "long",
      description: "Long output",
      execute() {
        return "abcdef";
      },
    };

    const result = await runAgentTurn({
      model,
      tools: [tool],
      messages: [{ role: "user", content: "loop" }],
      maxIterations: 2,
      maxToolResultChars: 3,
    });

    expect(result.finishReason).toBe("max_iterations");
    expect(result.messages.at(-1)?.content).toContain("...[truncated 3 chars]");
  });
});
