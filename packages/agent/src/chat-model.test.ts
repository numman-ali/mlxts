import { describe, expect, test } from "bun:test";

import { createOpenAIChatAgentModel } from "./chat-model";
import { runAgentTurn } from "./loop";
import type { AgentModelStreamEvent } from "./types";

function sseResponse(payloads: readonly unknown[]): Response {
  const body = payloads
    .map((payload) => `data: ${payload === "[DONE]" ? payload : JSON.stringify(payload)}\n\n`)
    .join("");
  return new Response(body, {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  });
}

describe("createOpenAIChatAgentModel", () => {
  test("posts messages and tools to chat completions", async () => {
    const requests: {
      input: string | URL | Request;
      init?: RequestInit & { verbose?: boolean };
    }[] = [];
    const model = createOpenAIChatAgentModel({
      endpoint: "http://localhost:8000",
      model: "qwen-local",
      apiKey: "secret",
      temperature: 0.7,
      enableThinking: false,
      verbose: true,
      fetch(input, init) {
        requests.push(init === undefined ? { input } : { input, init });
        return Promise.resolve(
          new Response(
            JSON.stringify({
              choices: [{ message: { content: "done" } }],
            }),
            { status: 200 },
          ),
        );
      },
    });

    const response = await model.complete({
      iteration: 0,
      messages: [{ role: "user", content: "hi" }],
      tools: [{ name: "lookup", description: "Lookup", execute: () => "ok" }],
    });

    expect(response).toEqual({ content: "done" });
    const request = requests[0];
    if (request === undefined) {
      throw new Error("Expected one chat request.");
    }
    expect(request.input).toBe("http://localhost:8000/v1/chat/completions");
    expect(new Headers(request.init?.headers).get("authorization")).toBe("Bearer secret");
    expect(request.init?.verbose).toBe(true);
    const body = JSON.parse(String(request.init?.body));
    expect(body).toMatchObject({
      model: "qwen-local",
      tools: [{ type: "function", function: { name: "lookup" } }],
      tool_choice: "auto",
      max_tokens: 512,
      temperature: 0.7,
      chat_template_kwargs: { enable_thinking: false },
    });
    expect(body.messages[0]?.role).toBe("system");
    expect(body.messages[0]?.content).toContain("<tool_call>");
    expect(body.messages[1]).toEqual({ role: "user", content: "hi" });
  });

  test("parses reasoning content separately from visible content", async () => {
    const model = createOpenAIChatAgentModel({
      endpoint: "http://localhost:8000/v1/chat/completions",
      model: "qwen-local",
      fetch() {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              choices: [
                {
                  message: {
                    content: "I should greet the user.</think>\n\nHello!",
                  },
                },
              ],
            }),
            { status: 200 },
          ),
        );
      },
    });

    const response = await model.complete({
      iteration: 0,
      messages: [{ role: "user", content: "hi" }],
      tools: [],
    });

    expect(response).toEqual({
      content: "Hello!",
      reasoningContent: "I should greet the user.",
    });
  });

  test("serializes prior reasoning and tool observations in multi-turn history", async () => {
    let requestBody: unknown;
    const model = createOpenAIChatAgentModel({
      endpoint: "http://localhost:8000",
      model: "qwen-local",
      enableThinking: true,
      fetch(_input, init) {
        requestBody = JSON.parse(String(init?.body));
        return Promise.resolve(
          new Response(
            JSON.stringify({
              choices: [{ message: { content: "done", reasoning_content: "finished" } }],
            }),
            { status: 200 },
          ),
        );
      },
    });

    const response = await model.complete({
      iteration: 1,
      messages: [
        { role: "user", content: "Read the file" },
        {
          role: "assistant",
          content: '<tool_call><function=read_file>{"path":"README.md"}</function></tool_call>',
          reasoningContent: "I need to read the file before answering.",
        },
        { role: "tool", name: "read_file", toolCallId: "tool-1", content: "README contents" },
        { role: "user", content: "Now summarize it" },
      ],
      tools: [],
    });

    expect(response).toEqual({ content: "done", reasoningContent: "finished" });
    expect(requestBody).toMatchObject({
      model: "qwen-local",
      chat_template_kwargs: { enable_thinking: true },
      messages: [
        { role: "user", content: "Read the file" },
        {
          role: "assistant",
          reasoning_content: "I need to read the file before answering.",
        },
        { role: "tool", name: "read_file", tool_call_id: "tool-1", content: "README contents" },
        { role: "user", content: "Now summarize it" },
      ],
    });
  });

  test("parses standard OpenAI tool calls", async () => {
    const model = createOpenAIChatAgentModel({
      endpoint: "http://localhost:8000/v1/chat/completions",
      model: "qwen-local",
      fetch() {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              choices: [
                {
                  message: {
                    content: null,
                    tool_calls: [
                      {
                        type: "function",
                        function: { name: "lookup", arguments: "" },
                      },
                    ],
                  },
                },
              ],
            }),
            { status: 200 },
          ),
        );
      },
    });

    const response = await model.complete({
      iteration: 0,
      messages: [{ role: "user", content: "hi" }],
      tools: [],
    });

    expect(response).toEqual({
      content: "",
      toolCalls: [{ id: "tool-1", name: "lookup", arguments: {} }],
    });
  });

  test("streams chat completions and aggregates reasoning and content", async () => {
    let requestBody: unknown;
    const model = createOpenAIChatAgentModel({
      endpoint: "http://localhost:8000",
      model: "qwen-local",
      stream: true,
      fetch(_input, init) {
        requestBody = JSON.parse(String(init?.body));
        return Promise.resolve(
          sseResponse([
            { choices: [{ delta: { role: "assistant" }, finish_reason: null }] },
            { choices: [{ delta: { reasoning_content: "Think " }, finish_reason: null }] },
            { choices: [{ delta: { reasoning_content: "carefully." }, finish_reason: null }] },
            { choices: [{ delta: { content: "Hel" }, finish_reason: null }] },
            { choices: [{ delta: { content: "lo" }, finish_reason: null }] },
            { choices: [{ delta: {}, finish_reason: "stop" }] },
            "[DONE]",
          ]),
        );
      },
    });

    const response = await model.complete({
      iteration: 0,
      messages: [{ role: "user", content: "hi" }],
      tools: [],
    });

    expect(requestBody).toMatchObject({ model: "qwen-local", stream: true });
    expect(response).toEqual({ content: "Hello", reasoningContent: "Think carefully." });
  });

  test("streams through end-of-body and falls back to Qwen think-tag parsing", async () => {
    const model = createOpenAIChatAgentModel({
      endpoint: "http://localhost:8000",
      model: "qwen-local",
      stream: true,
      fetch() {
        return Promise.resolve(
          new Response(
            [
              'data: {"choices":[{"delta":{"content":"I should answer.</think>\\n\\nHel"}}]}',
              "",
              'data: {"choices":[{"delta":{"content":"lo"}}]}',
            ].join("\n"),
            {
              status: 200,
              headers: { "content-type": "text/event-stream" },
            },
          ),
        );
      },
    });

    const response = await model.complete({
      iteration: 0,
      messages: [{ role: "user", content: "hi" }],
      tools: [],
    });

    expect(response).toEqual({ content: "Hello", reasoningContent: "I should answer." });
  });

  test("streams open think tags as fallback reasoning when no reasoning delta is present", async () => {
    const model = createOpenAIChatAgentModel({
      endpoint: "http://localhost:8000",
      model: "qwen-local",
      stream: true,
      fetch() {
        return Promise.resolve(
          sseResponse([
            { choices: [{ delta: { content: "Visible <think>still reasoning" } }] },
            "[DONE]",
          ]),
        );
      },
    });

    const response = await model.complete({
      iteration: 0,
      messages: [{ role: "user", content: "hi" }],
      tools: [],
    });

    expect(response).toEqual({ content: "Visible", reasoningContent: "still reasoning" });
  });

  test("ignores malformed streamed chunks and malformed tool-call deltas", async () => {
    const model = createOpenAIChatAgentModel({
      endpoint: "http://localhost:8000",
      model: "qwen-local",
      stream: true,
      fetch() {
        return Promise.resolve(
          sseResponse([
            {},
            { choices: [{}] },
            { choices: [{ delta: { tool_calls: [null] } }] },
            {
              choices: [
                {
                  delta: {
                    tool_calls: [
                      { index: 0, id: "nameless", function: { arguments: "{" } },
                      { index: 1, function: { name: "lookup", arguments: "{" } },
                      { index: 2, function: { name: "empty", arguments: "" } },
                    ],
                  },
                },
              ],
            },
            "[DONE]",
          ]),
        );
      },
    });

    const response = await model.complete({
      iteration: 0,
      messages: [{ role: "user", content: "tools" }],
      tools: [],
    });

    expect(response).toEqual({
      content: "",
      toolCalls: [
        { id: "tool-2", name: "lookup", arguments: {} },
        { id: "tool-3", name: "empty", arguments: {} },
      ],
    });
  });

  test("exposes and aggregates streamed tool call deltas by index", async () => {
    const model = createOpenAIChatAgentModel({
      endpoint: "http://localhost:8000",
      model: "qwen-local",
      stream: true,
      fetch() {
        return Promise.resolve(
          sseResponse([
            {
              choices: [
                {
                  delta: {
                    tool_calls: [
                      {
                        index: 0,
                        id: "call-read",
                        type: "function",
                        function: { name: "read_", arguments: '{"path"' },
                      },
                    ],
                  },
                  finish_reason: null,
                },
              ],
            },
            {
              choices: [
                {
                  delta: {
                    tool_calls: [
                      { index: 0, function: { name: "file", arguments: ':"README.md"}' } },
                      {
                        index: 1,
                        id: "call-list",
                        type: "function",
                        function: { name: "list_files", arguments: "{}" },
                      },
                    ],
                  },
                  finish_reason: null,
                },
              ],
            },
            { choices: [{ delta: {}, finish_reason: "tool_calls" }] },
            "[DONE]",
          ]),
        );
      },
    });

    const stream = model.stream?.({
      iteration: 0,
      messages: [{ role: "user", content: "read and list" }],
      tools: [],
    });
    expect(stream).toBeDefined();
    if (stream === undefined) {
      throw new Error("expected streaming model support");
    }
    const streamed: AgentModelStreamEvent[] = [];
    for await (const event of await stream) {
      streamed.push(event);
    }
    const response = await model.complete({
      iteration: 0,
      messages: [{ role: "user", content: "read and list" }],
      tools: [],
    });

    expect(streamed).toContainEqual({
      type: "tool_call_delta",
      index: 0,
      id: "call-read",
      nameDelta: "read_",
      argumentsDelta: '{"path"',
    });
    expect(streamed).toContainEqual({
      type: "tool_call_delta",
      index: 0,
      nameDelta: "file",
      argumentsDelta: ':"README.md"}',
    });
    expect(response).toEqual({
      content: "",
      toolCalls: [
        { id: "call-read", name: "read_file", arguments: { path: "README.md" } },
        { id: "call-list", name: "list_files", arguments: {} },
      ],
    });
  });

  test("keeps programmatic streaming opt-out on the non-streaming path", async () => {
    let requestBody: unknown;
    const model = createOpenAIChatAgentModel({
      endpoint: "http://localhost:8000",
      model: "qwen-local",
      stream: false,
      fetch(_input, init) {
        requestBody = JSON.parse(String(init?.body));
        return Promise.resolve(
          new Response(JSON.stringify({ choices: [{ message: { content: "whole answer" } }] }), {
            status: 200,
          }),
        );
      },
    });

    const result = await runAgentTurn({
      model,
      messages: [{ role: "user", content: "hi" }],
    });

    expect(model.stream).toBeUndefined();
    expect(requestBody).toMatchObject({ model: "qwen-local" });
    expect(requestBody).not.toHaveProperty("stream");
    expect(result.finalText).toBe("whole answer");
  });

  test("ignores malformed standard tool calls", async () => {
    const model = createOpenAIChatAgentModel({
      endpoint: "http://localhost:8000/v1",
      model: "qwen-local",
      fetch() {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              choices: [
                {
                  message: {
                    content: "no tools",
                    tool_calls: [
                      { type: "function", function: { name: "", arguments: "{" } },
                      { type: "function" },
                    ],
                  },
                },
              ],
            }),
            { status: 200 },
          ),
        );
      },
    });

    await expect(
      model.complete({ iteration: 0, messages: [{ role: "user", content: "hi" }], tools: [] }),
    ).resolves.toEqual({ content: "no tools" });
  });

  test("throws clear errors for failed requests and malformed responses", async () => {
    const failed = createOpenAIChatAgentModel({
      endpoint: "http://localhost:8000",
      model: "qwen-local",
      fetch() {
        return Promise.resolve(new Response("nope", { status: 500 }));
      },
    });

    await expect(failed.complete({ iteration: 0, messages: [], tools: [] })).rejects.toThrow(
      "Chat completion request failed",
    );

    const malformed = createOpenAIChatAgentModel({
      endpoint: "http://localhost:8000",
      model: "qwen-local",
      fetch() {
        return Promise.resolve(new Response(JSON.stringify({ choices: [] }), { status: 200 }));
      },
    });

    await expect(malformed.complete({ iteration: 0, messages: [], tools: [] })).rejects.toThrow(
      "message choice",
    );

    const missingChoices = createOpenAIChatAgentModel({
      endpoint: "http://localhost:8000",
      model: "qwen-local",
      fetch() {
        return Promise.resolve(new Response(JSON.stringify({}), { status: 200 }));
      },
    });

    await expect(
      missingChoices.complete({ iteration: 0, messages: [], tools: [] }),
    ).rejects.toThrow("choices");

    const malformedStream = createOpenAIChatAgentModel({
      endpoint: "http://localhost:8000",
      model: "qwen-local",
      stream: true,
      fetch() {
        return Promise.resolve(
          new Response("data: {bad\n\n", {
            status: 200,
            headers: { "content-type": "text/event-stream" },
          }),
        );
      },
    });

    await expect(
      malformedStream.complete({ iteration: 0, messages: [], tools: [] }),
    ).rejects.toThrow("Streaming chat completion response included malformed JSON");

    const missingStreamBody = createOpenAIChatAgentModel({
      endpoint: "http://localhost:8000",
      model: "qwen-local",
      stream: true,
      fetch() {
        return Promise.resolve(new Response(null, { status: 200 }));
      },
    });

    await expect(
      missingStreamBody.complete({ iteration: 0, messages: [], tools: [] }),
    ).rejects.toThrow("Streaming chat completion response did not include a body");

    const failedStream = createOpenAIChatAgentModel({
      endpoint: "http://localhost:8000",
      model: "qwen-local",
      apiKey: "secret",
      verbose: true,
      fetch(_input, init) {
        expect(new Headers(init?.headers).get("authorization")).toBe("Bearer secret");
        expect(init?.verbose).toBe(true);
        return Promise.resolve(new Response("stream nope", { status: 502 }));
      },
    });

    if (failedStream.stream === undefined) {
      throw new Error("expected streaming model support");
    }
    await expect(failedStream.stream({ iteration: 0, messages: [], tools: [] })).rejects.toThrow(
      "Chat completion request failed (502): stream nope",
    );
  });
});
