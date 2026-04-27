import { describe, expect, test } from "bun:test";

import {
  createOpenAIChatCompletionReasoningStream,
  createOpenAIChatCompletionToolCallStream,
  formatOpenAIChatCompletionResponse,
  formatOpenAIChatCompletionStreamChunk,
  formatOpenAIChatCompletionUsageStreamChunk,
  normalizeOpenAIChatCompletionRequest,
} from "./openai-chat-completions";

describe("OpenAI chat completions adapter", () => {
  test("normalizes messages and tools into a protocol-neutral request", () => {
    const normalized = normalizeOpenAIChatCompletionRequest(
      {
        model: "mlx-community/Qwen3.6-27B-4bit",
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
        ignore_eos: true,
        stop: ["Observation:", "Final answer:"],
        chat_template_kwargs: { enable_thinking: false },
      },
      { id: "chat-test" },
    );

    expect(normalized.request).toMatchObject({
      id: "chat-test",
      model: "mlx-community/Qwen3.6-27B-4bit",
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
        ignoreEos: true,
        stop: ["Observation:", "Final answer:"],
      },
      protocol: "openai.chat_completions",
    });
    expect(normalized.stream).toBe(false);
    expect(normalized.streamOptions).toEqual({ includeUsage: false });
  });

  test("omits sampling overrides so model generation config can apply", () => {
    const normalized = normalizeOpenAIChatCompletionRequest(
      { model: "mlx-community/Qwen3.6-27B-4bit", messages: [{ role: "user", content: "Hi" }] },
      { id: "chat-test" },
    );

    expect(normalized.request.sampling).toEqual({ maxTokens: 16 });
    expect(normalized.stream).toBe(false);
  });

  test("normalizes OpenAI chat option parity fields", () => {
    const normalized = normalizeOpenAIChatCompletionRequest(
      {
        model: "mlx-community/Qwen3.6-27B-4bit",
        messages: [{ role: "user", content: "Hi" }],
        max_completion_tokens: 64,
        n: 1,
        seed: 123,
        user: "user-1",
        frequency_penalty: 0,
        presence_penalty: 0,
        logit_bias: {},
        logprobs: false,
        parallel_tool_calls: true,
        response_format: { type: "text" },
      },
      { id: "chat-test" },
    );

    expect(normalized.request.sampling).toEqual({ maxTokens: 64, seed: 123 });
    expect(normalized.request.metadata).toEqual({ user: "user-1" });
  });

  test("preserves assistant tool calls and tool observations", () => {
    const normalized = normalizeOpenAIChatCompletionRequest(
      {
        model: "mlx-community/Qwen3.6-27B-4bit",
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

  test("accepts developer messages and text content parts", () => {
    const normalized = normalizeOpenAIChatCompletionRequest(
      {
        model: "mlx-community/Qwen3.6-27B-4bit",
        messages: [
          { role: "developer", content: [{ type: "text", text: "Use concise answers." }] },
          {
            role: "user",
            content: [
              { type: "text", text: "Hello " },
              { type: "text", text: "from parts." },
            ],
          },
        ],
      },
      { id: "chat-test" },
    );

    expect(normalized.request.input).toEqual({
      kind: "messages",
      messages: [
        { role: "system", content: "Use concise answers." },
        { role: "user", content: "Hello from parts." },
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
        {
          model: "tiny",
          messages: [{ role: "user", content: [{ type: "image_url", image_url: "file://x" }] }],
        },
        { id: "chat-test" },
      ),
    ).toThrow("image content parts");
    expect(() =>
      normalizeOpenAIChatCompletionRequest(
        {
          model: "tiny",
          messages: [{ role: "user", content: [{ type: "text", text: 1 }] }],
        },
        { id: "chat-test" },
      ),
    ).toThrow('string "text"');
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

  test("rejects unsupported OpenAI chat option semantics explicitly", () => {
    const base = { model: "tiny", messages: [{ role: "user", content: "hi" }] };
    const cases: Array<{ extra: Record<string, unknown>; message: string }> = [
      { extra: { n: 2 }, message: "n" },
      { extra: { max_completion_tokens: -1 }, message: "max_completion_tokens" },
      { extra: { max_tokens: 2, max_completion_tokens: 3 }, message: "must match" },
      { extra: { seed: -1 }, message: "seed" },
      { extra: { user: 1 }, message: "user" },
      { extra: { presence_penalty: 0.5 }, message: "presence_penalty" },
      { extra: { frequency_penalty: -0.5 }, message: "frequency_penalty" },
      { extra: { logit_bias: { "1": 10 } }, message: "logit_bias" },
      { extra: { logprobs: true }, message: "logprobs" },
      { extra: { top_logprobs: 1 }, message: "top_logprobs" },
      { extra: { parallel_tool_calls: false }, message: "parallel_tool_calls" },
      { extra: { response_format: "json" }, message: "response_format" },
      { extra: { response_format: { type: "json_object" } }, message: "json_object" },
    ];

    for (const item of cases) {
      expect(() =>
        normalizeOpenAIChatCompletionRequest({ ...base, ...item.extra }, { id: "chat-test" }),
      ).toThrow(item.message);
    }
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
      usage: {
        prompt_tokens: 1,
        completion_tokens: 2,
        total_tokens: 3,
        prompt_tokens_details: { cached_tokens: 0, cache_write_tokens: 0 },
      },
    });
  });

  test("formats generated tool-call envelopes as OpenAI tool calls", () => {
    const chat = normalizeOpenAIChatCompletionRequest(
      {
        model: "tiny",
        messages: [{ role: "user", content: "read the file" }],
        tools: [
          {
            type: "function",
            function: {
              name: "read_file",
              parameters: { type: "object", properties: { path: { type: "string" } } },
            },
          },
        ],
      },
      { id: "chat-test" },
    );

    const response = formatOpenAIChatCompletionResponse(
      chat,
      {
        text: '<tool_call>{"name":"read_file","arguments":{"path":"README.md"}}</tool_call>',
        finishReason: "stop",
      },
      { id: "chat-test", created: 123 },
    );

    expect(response.choices[0]).toEqual({
      index: 0,
      message: {
        role: "assistant",
        content: null,
        tool_calls: [
          {
            id: "call_1",
            type: "function",
            function: { name: "read_file", arguments: '{"path":"README.md"}' },
          },
        ],
      },
      finish_reason: "tool_calls",
    });
  });

  test("formats generated Qwen-style native tool calls", () => {
    const chat = normalizeOpenAIChatCompletionRequest(
      {
        model: "tiny",
        messages: [{ role: "user", content: "list files" }],
        tools: [{ type: "function", function: { name: "list_files" } }],
      },
      { id: "chat-test" },
    );

    const response = formatOpenAIChatCompletionResponse(
      chat,
      {
        text: "<tool_call><function=list_files><parameter=path>.</parameter></function></tool_call>",
        finishReason: "stop",
      },
      { id: "chat-test", created: 123 },
    );

    expect(response.choices[0]?.message.tool_calls?.[0]).toEqual({
      id: "call_1",
      type: "function",
      function: { name: "list_files", arguments: '{"path":"."}' },
    });
    expect(response.choices[0]?.finish_reason).toBe("tool_calls");
  });

  test("formats reasoning plus multiple generated tool calls without XML leakage", () => {
    const chat = normalizeOpenAIChatCompletionRequest(
      {
        model: "tiny",
        messages: [{ role: "user", content: "read and list" }],
        tools: [
          { type: "function", function: { name: "read_file" } },
          { type: "function", function: { name: "list_files" } },
        ],
      },
      { id: "chat-test" },
    );

    const response = formatOpenAIChatCompletionResponse(
      chat,
      {
        text: [
          "<think>I need two tools.</think>",
          '<tool_call>{"id":"call-read","name":"read_file","arguments":{"path":"README.md"}}</tool_call>',
          '<tool_call>{"name":"list_files","arguments":{"path":"."}}</tool_call>',
        ].join("\n"),
        finishReason: "stop",
      },
      { id: "chat-test", created: 123 },
    );

    expect(response.choices[0]).toEqual({
      index: 0,
      message: {
        role: "assistant",
        content: null,
        reasoning_content: "I need two tools.",
        tool_calls: [
          {
            id: "call-read",
            type: "function",
            function: { name: "read_file", arguments: '{"path":"README.md"}' },
          },
          {
            id: "call_2",
            type: "function",
            function: { name: "list_files", arguments: '{"path":"."}' },
          },
        ],
      },
      finish_reason: "tool_calls",
    });
  });

  test("leaves malformed generated tool-call text as content", () => {
    const chat = normalizeOpenAIChatCompletionRequest(
      {
        model: "tiny",
        messages: [{ role: "user", content: "read" }],
        tools: [{ type: "function", function: { name: "read_file" } }],
      },
      { id: "chat-test" },
    );

    const response = formatOpenAIChatCompletionResponse(
      chat,
      {
        text: '<tool_call>{"arguments":{"path":"README.md"}}</tool_call>',
        finishReason: "stop",
      },
      { id: "chat-test", created: 123 },
    );

    expect(response.choices[0]?.message.tool_calls).toBeUndefined();
    expect(response.choices[0]?.message.content).toContain("<tool_call>");
    expect(response.choices[0]?.finish_reason).toBe("stop");
  });

  test("leaves tool-call-looking text alone when tools are not enabled", () => {
    const chat = normalizeOpenAIChatCompletionRequest(
      { model: "tiny", messages: [{ role: "user", content: "hi" }] },
      { id: "chat-test" },
    );

    const response = formatOpenAIChatCompletionResponse(
      chat,
      {
        text: '<tool_call>{"name":"read_file","arguments":{"path":"README.md"}}</tool_call>',
        finishReason: "stop",
      },
      { id: "chat-test", created: 123 },
    );

    expect(response.choices[0]?.message.tool_calls).toBeUndefined();
    expect(response.choices[0]?.message.content).toContain("<tool_call>");
    expect(response.choices[0]?.finish_reason).toBe("stop");
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
      {
        model: "tiny",
        messages: [{ role: "user", content: "hi" }],
        stream: true,
        stream_options: { include_usage: true },
      },
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
      usage: null,
    });

    expect(
      formatOpenAIChatCompletionStreamChunk(
        chat,
        {
          toolCalls: [
            {
              index: 0,
              id: "call_1",
              type: "function",
              function: { name: "read_file", arguments: '{"path":"README.md"}' },
            },
          ],
        },
        { id: "chat-test", created: 123, finishReason: "tool_calls" },
      ),
    ).toEqual({
      id: "chat-test",
      object: "chat.completion.chunk",
      created: 123,
      model: "tiny",
      choices: [
        {
          index: 0,
          delta: {
            tool_calls: [
              {
                index: 0,
                id: "call_1",
                type: "function",
                function: { name: "read_file", arguments: '{"path":"README.md"}' },
              },
            ],
          },
          finish_reason: "tool_calls",
        },
      ],
      usage: null,
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
        {
          promptTokens: 6,
          completionTokens: 2,
          totalTokens: 8,
          cacheReadTokens: 3,
          cacheWriteTokens: 1,
        },
        { id: "chat-test", created: 123 },
      ),
    ).toEqual({
      id: "chat-test",
      object: "chat.completion.chunk",
      created: 123,
      model: "tiny",
      choices: [],
      usage: {
        prompt_tokens: 6,
        completion_tokens: 2,
        total_tokens: 8,
        prompt_tokens_details: { cached_tokens: 4, cache_write_tokens: 1 },
      },
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
    expect(stream.finish()).toEqual([{ content: "Hello" }]);
  });

  test("buffers generated streaming tool-call envelopes", () => {
    const stream = createOpenAIChatCompletionToolCallStream(true);

    expect(stream.push("Before ")).toEqual([]);
    expect(
      stream.push('<tool_call>{"name":"read_file","arguments":{"path":"README.md"}}</tool_call>'),
    ).toEqual([
      { content: "Before " },
      {
        toolCalls: [
          {
            index: 0,
            id: "call_1",
            type: "function",
            function: { name: "read_file", arguments: '{"path":"README.md"}' },
          },
        ],
      },
    ]);
    expect(stream.finish()).toEqual([]);
  });
});
