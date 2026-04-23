import { describe, expect, test } from "bun:test";

import {
  createOpenAIChatCompletionReasoningStream,
  formatOpenAIChatCompletionResponse,
  formatOpenAIChatCompletionStreamChunk,
  formatOpenAIChatCompletionUsageStreamChunk,
  normalizeOpenAIChatCompletionRequest,
} from "./openai-chat-completions";

describe("OpenAI chat completions adapter", () => {
  test("normalizes messages and tools into a protocol-neutral request", () => {
    const normalized = normalizeOpenAIChatCompletionRequest(
      {
        model: "qwen-local",
        messages: [{ role: "user", content: "Read README.md" }],
        tools: [
          {
            type: "function",
            function: {
              name: "read_file",
              description: "Read a file",
              parameters: { type: "object", properties: { path: { type: "string" } } },
            },
          },
        ],
        max_tokens: 32,
        temperature: 0,
        top_k: 20,
        stop: ["Observation:", "Final answer:"],
        chat_template_kwargs: { enable_thinking: false },
      },
      { id: "chat-test" },
    );

    expect(normalized.request).toMatchObject({
      id: "chat-test",
      model: "qwen-local",
      input: {
        kind: "messages",
        messages: [{ role: "user", content: "Read README.md" }],
        tools: [{ type: "function", function: { name: "read_file" } }],
        chatTemplate: { enableThinking: false },
      },
      sampling: {
        maxTokens: 32,
        temperature: 0,
        topK: 20,
        stop: ["Observation:", "Final answer:"],
      },
      protocol: "openai.chat_completions",
    });
    expect(normalized.stream).toBe(false);
    expect(normalized.streamOptions).toEqual({ includeUsage: false });
  });

  test("omits sampling overrides so model generation config can apply", () => {
    const normalized = normalizeOpenAIChatCompletionRequest(
      { model: "qwen-local", messages: [{ role: "user", content: "Hi" }] },
      { id: "chat-test" },
    );

    expect(normalized.request.sampling).toEqual({ maxTokens: 16 });
    expect(normalized.stream).toBe(false);
  });

  test("preserves assistant tool calls and tool observations", () => {
    const normalized = normalizeOpenAIChatCompletionRequest(
      {
        model: "qwen-local",
        messages: [
          {
            role: "assistant",
            content: null,
            tool_calls: [
              {
                id: "call-1",
                type: "function",
                function: { name: "read_file", arguments: '{"path":"README.md"}' },
              },
            ],
          },
          { role: "tool", tool_call_id: "call-1", name: "read_file", content: "hello" },
        ],
      },
      { id: "chat-test" },
    );

    expect(normalized.request.input).toEqual({
      kind: "messages",
      messages: [
        {
          role: "assistant",
          content: "",
          tool_calls: [
            {
              id: "call-1",
              type: "function",
              function: { name: "read_file", arguments: '{"path":"README.md"}' },
            },
          ],
        },
        { role: "tool", content: "hello", name: "read_file", tool_call_id: "call-1" },
      ],
    });
  });

  test("rejects unsupported chat shapes explicitly", () => {
    expect(() =>
      normalizeOpenAIChatCompletionRequest(
        { model: "tiny", messages: [{ role: "user", content: "hi" }], stream: "yes" },
        { id: "chat-test" },
      ),
    ).toThrow("stream");
    expect(() =>
      normalizeOpenAIChatCompletionRequest(
        { model: "tiny", messages: [], tool_choice: { type: "function" } },
        { id: "chat-test" },
      ),
    ).toThrow("tool_choice");
    expect(() =>
      normalizeOpenAIChatCompletionRequest({ model: "tiny", messages: [] }, { id: "chat-test" }),
    ).toThrow("non-empty array");
    expect(() => normalizeOpenAIChatCompletionRequest(null, { id: "chat-test" })).toThrow(
      "request body",
    );
    expect(() =>
      normalizeOpenAIChatCompletionRequest(
        { model: "", messages: [{ role: "user" }] },
        { id: "chat-test" },
      ),
    ).toThrow("model");
    expect(() =>
      normalizeOpenAIChatCompletionRequest(
        { model: "tiny", messages: ["bad"] },
        { id: "chat-test" },
      ),
    ).toThrow("entries must be objects");
    expect(() =>
      normalizeOpenAIChatCompletionRequest(
        { model: "tiny", messages: [{ role: "bad", content: "hi" }] },
        { id: "chat-test" },
      ),
    ).toThrow("message role");
    expect(() =>
      normalizeOpenAIChatCompletionRequest(
        { model: "tiny", messages: [{ role: "user", content: 1 }] },
        { id: "chat-test" },
      ),
    ).toThrow("content");
    expect(() =>
      normalizeOpenAIChatCompletionRequest(
        { model: "tiny", messages: [{ role: "assistant", tool_calls: {} }] },
        { id: "chat-test" },
      ),
    ).toThrow("tool_calls");
    expect(() =>
      normalizeOpenAIChatCompletionRequest(
        {
          model: "tiny",
          messages: [{ role: "assistant", tool_calls: [{ type: "bad" }] }],
        },
        { id: "chat-test" },
      ),
    ).toThrow("function calls");
    expect(() =>
      normalizeOpenAIChatCompletionRequest(
        {
          model: "tiny",
          messages: [{ role: "assistant", tool_calls: [{ type: "function", function: {} }] }],
        },
        { id: "chat-test" },
      ),
    ).toThrow("function name");
    expect(() =>
      normalizeOpenAIChatCompletionRequest(
        { model: "tiny", messages: [{ role: "user", content: "hi" }], tools: {} },
        { id: "chat-test" },
      ),
    ).toThrow("tools");
    expect(() =>
      normalizeOpenAIChatCompletionRequest(
        {
          model: "tiny",
          messages: [{ role: "user", content: "hi" }],
          tools: [{ type: "bad" }],
        },
        { id: "chat-test" },
      ),
    ).toThrow("function tools");
    expect(() =>
      normalizeOpenAIChatCompletionRequest(
        {
          model: "tiny",
          messages: [{ role: "user", content: "hi" }],
          tools: [{ type: "function", function: { name: "" } }],
        },
        { id: "chat-test" },
      ),
    ).toThrow("tool function name");
    expect(() =>
      normalizeOpenAIChatCompletionRequest(
        { model: "tiny", messages: [{ role: "user", content: "hi" }], max_tokens: -1 },
        { id: "chat-test" },
      ),
    ).toThrow("max_tokens");
    expect(() =>
      normalizeOpenAIChatCompletionRequest(
        { model: "tiny", messages: [{ role: "user", content: "hi" }], top_k: 0 },
        { id: "chat-test" },
      ),
    ).toThrow("top_k");
    expect(() =>
      normalizeOpenAIChatCompletionRequest(
        {
          model: "tiny",
          messages: [{ role: "user", content: "hi" }],
          stop: 1,
        },
        { id: "chat-test" },
      ),
    ).toThrow("stop");
    expect(() =>
      normalizeOpenAIChatCompletionRequest(
        {
          model: "tiny",
          messages: [{ role: "user", content: "hi" }],
          stop: ["1", "2", "3", "4", "5"],
        },
        { id: "chat-test" },
      ),
    ).toThrow("at most 4");
    expect(() =>
      normalizeOpenAIChatCompletionRequest(
        {
          model: "tiny",
          messages: [{ role: "user", content: "hi" }],
          chat_template_kwargs: "bad",
        },
        { id: "chat-test" },
      ),
    ).toThrow("chat_template_kwargs");
    expect(() =>
      normalizeOpenAIChatCompletionRequest(
        {
          model: "tiny",
          messages: [{ role: "user", content: "hi" }],
          stream_options: { include_usage: true },
        },
        { id: "chat-test" },
      ),
    ).toThrow("stream_options");
  });

  test("normalizes chat streaming flags and usage streaming options", () => {
    const normalized = normalizeOpenAIChatCompletionRequest(
      {
        model: "tiny",
        messages: [{ role: "user", content: "hi" }],
        stream: true,
        stream_options: { include_usage: true },
      },
      { id: "chat-test" },
    );

    expect(normalized.stream).toBe(true);
    expect(normalized.streamOptions).toEqual({ includeUsage: true });
    expect(normalized.request.stream).toBe(true);
  });

  test("formats chat completions with usage", () => {
    const chat = normalizeOpenAIChatCompletionRequest(
      { model: "tiny", messages: [{ role: "user", content: "hi" }] },
      { id: "chat-test" },
    );

    const response = formatOpenAIChatCompletionResponse(
      chat,
      {
        text: "hello",
        finishReason: "stop",
        usage: { promptTokens: 1, completionTokens: 2, totalTokens: 3 },
      },
      { id: "chat-test", created: 123 },
    );

    expect(response).toEqual({
      id: "chat-test",
      object: "chat.completion",
      created: 123,
      model: "tiny",
      choices: [
        {
          index: 0,
          message: { role: "assistant", content: "hello" },
          finish_reason: "stop",
        },
      ],
      usage: { prompt_tokens: 1, completion_tokens: 2, total_tokens: 3 },
    });
  });

  test("moves Qwen thinking text into reasoning_content", () => {
    const chat = normalizeOpenAIChatCompletionRequest(
      { model: "tiny", messages: [{ role: "user", content: "hi" }] },
      { id: "chat-test" },
    );

    const response = formatOpenAIChatCompletionResponse(
      chat,
      {
        text: "I should greet the user.</think>\n\nHello!",
        finishReason: "stop",
      },
      { id: "chat-test", created: 123 },
    );

    expect(response.choices[0]?.message).toEqual({
      role: "assistant",
      content: "Hello!",
      reasoning_content: "I should greet the user.",
    });
  });

  test("maps non-stop finish reasons", () => {
    const chat = normalizeOpenAIChatCompletionRequest(
      { model: "tiny", messages: [{ role: "user", content: "hi" }] },
      { id: "chat-test" },
    );

    expect(
      formatOpenAIChatCompletionResponse(
        chat,
        { text: "", finishReason: "length" },
        { id: "chat-test", created: 123 },
      ).choices[0]?.finish_reason,
    ).toBe("length");
    expect(
      formatOpenAIChatCompletionResponse(
        chat,
        { text: "", finishReason: "cancelled" },
        { id: "chat-test", created: 123 },
      ).choices[0]?.finish_reason,
    ).toBeNull();
    expect(
      formatOpenAIChatCompletionResponse(
        chat,
        { text: "", finishReason: "error" },
        { id: "chat-test", created: 123 },
      ).choices[0]?.finish_reason,
    ).toBe("content_filter");
  });

  test("formats chat completion stream chunks and usage chunks", () => {
    const chat = normalizeOpenAIChatCompletionRequest(
      { model: "tiny", messages: [{ role: "user", content: "hi" }], stream: true },
      { id: "chat-test" },
    );

    expect(
      formatOpenAIChatCompletionStreamChunk(
        chat,
        { content: "Hello" },
        { id: "chat-test", created: 123, includeRole: true },
      ),
    ).toEqual({
      id: "chat-test",
      object: "chat.completion.chunk",
      created: 123,
      model: "tiny",
      choices: [
        {
          index: 0,
          delta: { role: "assistant", content: "Hello" },
          finish_reason: null,
        },
      ],
    });

    expect(
      formatOpenAIChatCompletionStreamChunk(
        chat,
        {},
        { id: "chat-test", created: 123, finishReason: "error" },
      ).choices[0]?.finish_reason,
    ).toBe("content_filter");
    expect(
      formatOpenAIChatCompletionStreamChunk(
        chat,
        {},
        { id: "chat-test", created: 123, finishReason: "cancelled" },
      ).choices[0]?.finish_reason,
    ).toBeNull();

    expect(
      formatOpenAIChatCompletionUsageStreamChunk(
        chat,
        { promptTokens: 1, completionTokens: 2, totalTokens: 3 },
        { id: "chat-test", created: 123 },
      ),
    ).toEqual({
      id: "chat-test",
      object: "chat.completion.chunk",
      created: 123,
      model: "tiny",
      choices: [],
      usage: { prompt_tokens: 1, completion_tokens: 2, total_tokens: 3 },
    });
    expect(
      formatOpenAIChatCompletionUsageStreamChunk(chat, undefined, {
        id: "chat-test",
        created: 123,
      }).usage,
    ).toBeNull();
  });

  test("splits streamed Qwen reasoning from visible content", () => {
    const stream = createOpenAIChatCompletionReasoningStream();

    expect(stream.push("<think>I should ")).toEqual([{ reasoningContent: "I " }]);
    expect(stream.push("greet.</think>\n\nHel")).toEqual([{ reasoningContent: "should greet." }]);
    expect(stream.push("lo")).toEqual([]);
    expect(stream.finish()).toEqual([{ content: "\n\nHello" }]);
  });
});
