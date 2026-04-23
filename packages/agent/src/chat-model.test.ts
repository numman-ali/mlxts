import { describe, expect, test } from "bun:test";

import { createOpenAIChatAgentModel } from "./chat-model";

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
  });
});
