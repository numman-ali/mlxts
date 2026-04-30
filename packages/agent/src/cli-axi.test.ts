import { describe, expect, test } from "bun:test";

import { formatAgentRunResult, runAgentCli, runAgentOnce } from "./cli";
import type { AgentModel } from "./types";

describe("agent CLI AXI paths", () => {
  test("formats one-shot results as compact AXI stdout", async () => {
    const model: AgentModel = {
      complete(request) {
        expect(request.messages).toEqual([{ role: "user", content: "hello" }]);
        return { content: "final answer" };
      },
    };
    const output = await runAgentOnce(
      {
        model: "qwen",
        endpoint: "http://localhost:8000",
        prompt: "hello",
        cwd: ".",
        maxTokens: 32,
        stream: true,
        maxIterations: 4,
        verbose: false,
      },
      { model, tools: [] },
    );

    expect(output).toContain("agent_run:");
    expect(output).toContain('model: "qwen"');
    expect(output).toContain("finish_reason: stop");
    expect(output).toContain("tool_call_count: 0");
    expect(output).toContain('assistant:\n  "final answer"');
    expect(
      formatAgentRunResult({
        messages: [],
        finalText: "done",
        finishReason: "stop",
        iterations: 1,
        toolCalls: [],
      }),
    ).toContain('assistant:\n  "done"');
  });

  test("process runner keeps one-shot output finite and non-interactive", async () => {
    const stdout: string[] = [];
    let prompted = false;
    const result = await runAgentCli(["run", "--model", "qwen", "--prompt", "hello"], {
      isTTY: false,
      stdout(message) {
        stdout.push(message);
      },
      prompt() {
        prompted = true;
        return "unexpected";
      },
      model: {
        complete() {
          return { content: "done" };
        },
      },
      tools: [],
    });

    expect(result.exitCode).toBe(0);
    expect(prompted).toBe(false);
    expect(stdout).toHaveLength(1);
    expect(stdout.join("\n")).toContain("agent_run:");
    expect(stdout.join("\n")).not.toContain("Talking to");
  });

  test("process runner emits AXI stdout errors and stable exit codes", async () => {
    const nonTty: string[] = [];
    const usage = await runAgentCli(["--model", "qwen"], {
      isTTY: false,
      stdout(message) {
        nonTty.push(message);
      },
    });
    expect(usage.exitCode).toBe(2);
    expect(nonTty.join("\n")).toContain("code: usage");
    expect(nonTty.join("\n")).toContain("run --prompt");

    const runtime: string[] = [];
    const failed = await runAgentCli(["run", "--model", "qwen", "--prompt", "hello"], {
      stdout(message) {
        runtime.push(message);
      },
      model: {
        complete() {
          throw new Error("model unavailable");
        },
      },
      tools: [],
    });
    expect(failed.exitCode).toBe(1);
    expect(runtime.join("\n")).toContain("code: runtime");
    expect(runtime.join("\n")).toContain("model unavailable");
  });
});
