import { describe, expect, test } from "bun:test";

import {
  formatAnthropicMessageResponse,
  normalizeAnthropicMessageRequest,
} from "./anthropic-messages";

describe("Anthropic messages adapter", () => {
  test("normalizes text-only Messages requests into protocol-neutral message input", () => {
    const normalized = normalizeAnthropicMessageRequest(
      {
        model: "mlx-community/Qwen3.6-27B-4bit",
        system: [{ type: "text", text: "Be concise." }],
        messages: [
          { role: "user", content: [{ type: "text", text: "Hello" }] },
          {
            role: "assistant",
            content: [{ type: "thinking", thinking: "I should greet.", signature: "sig" }],
          },
        ],
        max_tokens: 64,
        temperature: 0.7,
        top_p: 0.95,
        top_k: 20,
        stop_sequences: ["\n\nHuman:"],
        stream: true,
        thinking: { type: "enabled", budget_tokens: 1024 },
        metadata: { user_id: "nomi" },
      },
      { id: "msg-test" },
    );

    expect(normalized).toMatchObject({
      model: "mlx-community/Qwen3.6-27B-4bit",
      stream: true,
      maxTokens: 64,
      temperature: 0.7,
      topP: 0.95,
      topK: 20,
      request: {
        id: "msg-test",
        model: "mlx-community/Qwen3.6-27B-4bit",
        input: {
          kind: "messages",
          messages: [
            { role: "system", content: "Be concise." },
            { role: "user", content: "Hello" },
            { role: "assistant", content: "", reasoning_content: "I should greet." },
          ],
          chatTemplate: { enableThinking: true },
        },
        sampling: {
          maxTokens: 64,
          temperature: 0.7,
          topP: 0.95,
          topK: 20,
          stop: ["\n\nHuman:"],
        },
        stream: true,
        protocol: "anthropic.messages",
        metadata: { user_id: "nomi", user: "nomi" },
      },
    });
  });

  test("normalizes ordered Anthropic image blocks as media content", () => {
    const normalized = normalizeAnthropicMessageRequest(
      {
        model: "mlx-community/Qwen3.6-27B-4bit",
        system: "Be visual.",
        messages: [
          {
            role: "user",
            content: [
              {
                type: "image",
                source: { type: "base64", media_type: "image/png", data: "abcd" },
              },
              { type: "text", text: "Describe this." },
              {
                type: "image",
                source: { type: "url", url: "https://example.com/image.png" },
              },
              {
                type: "image",
                source: { type: "file", file_id: "file-123" },
              },
            ],
          },
        ],
        max_tokens: 32,
        chat_template_kwargs: { enable_thinking: false },
      },
      { id: "msg-image" },
    );

    expect(normalized.request.input).toEqual({
      kind: "content",
      messages: [
        { role: "system", content: [{ kind: "text", text: "Be visual." }] },
        {
          role: "user",
          content: [
            {
              kind: "image",
              source: { kind: "data", mediaType: "image/png", data: "abcd" },
            },
            { kind: "text", text: "Describe this." },
            {
              kind: "image",
              source: { kind: "url", url: "https://example.com/image.png" },
            },
            {
              kind: "image",
              source: { kind: "file", fileId: "file-123" },
            },
          ],
        },
      ],
      chatTemplate: { enableThinking: false },
    });
  });

  test("normalizes Anthropic tools and tool-result turns into internal chat messages", () => {
    const normalized = normalizeAnthropicMessageRequest(
      {
        model: "mlx-community/Qwen3.6-27B-4bit",
        tools: [
          {
            name: "get_weather",
            description: "Get weather for a city.",
            input_schema: {
              type: "object",
              properties: { location: { type: "string" } },
              required: ["location"],
            },
          },
        ],
        messages: [
          { role: "user", content: "What is the weather in London?" },
          {
            role: "assistant",
            content: [
              { type: "text", text: "I will check." },
              {
                type: "tool_use",
                id: "toolu_1",
                name: "get_weather",
                input: { location: "London" },
              },
            ],
          },
          {
            role: "user",
            content: [
              {
                type: "tool_result",
                tool_use_id: "toolu_1",
                content: [{ type: "text", text: "15 degrees and raining." }],
              },
              { type: "text", text: "Summarize it." },
            ],
          },
        ],
        max_tokens: 32,
      },
      { id: "msg-tools" },
    );

    expect(normalized.request.input).toEqual({
      kind: "messages",
      messages: [
        { role: "user", content: "What is the weather in London?" },
        {
          role: "assistant",
          content: "I will check.",
          tool_calls: [
            {
              id: "toolu_1",
              type: "function",
              function: {
                name: "get_weather",
                arguments: JSON.stringify({ location: "London" }),
              },
            },
          ],
        },
        {
          role: "tool",
          content: "15 degrees and raining.",
          tool_call_id: "toolu_1",
        },
        { role: "user", content: "Summarize it." },
      ],
      tools: [
        {
          type: "function",
          function: {
            name: "get_weather",
            description: "Get weather for a city.",
            parameters: {
              type: "object",
              properties: { location: { type: "string" } },
              required: ["location"],
            },
          },
        },
      ],
    });
  });

  test("formats visible and thinking output as Anthropic content blocks", () => {
    const normalized = normalizeAnthropicMessageRequest(
      {
        model: "mlx-community/Qwen3.6-27B-4bit",
        messages: [{ role: "user", content: "Hi" }],
        max_tokens: 8,
      },
      { id: "msg-format" },
    );

    const response = formatAnthropicMessageResponse(
      normalized,
      {
        text: "<think>I should greet.</think>Hello",
        finishReason: "length",
        usage: { promptTokens: 5, completionTokens: 8, totalTokens: 13 },
      },
      { id: "msg-format" },
    );

    expect(response).toEqual({
      id: "msg-format",
      type: "message",
      role: "assistant",
      content: [
        { type: "thinking", thinking: "I should greet.", signature: "" },
        { type: "text", text: "Hello" },
      ],
      model: "mlx-community/Qwen3.6-27B-4bit",
      stop_reason: "max_tokens",
      stop_sequence: null,
      usage: { input_tokens: 5, output_tokens: 8 },
    });
  });

  test("formats generated tool calls as Anthropic tool_use blocks", () => {
    const normalized = normalizeAnthropicMessageRequest(
      {
        model: "mlx-community/Qwen3.6-27B-4bit",
        tools: [
          {
            name: "get_weather",
            input_schema: {
              type: "object",
              properties: { location: { type: "string" } },
            },
          },
        ],
        messages: [{ role: "user", content: "Weather?" }],
        max_tokens: 64,
      },
      { id: "msg-tool-format" },
    );

    const response = formatAnthropicMessageResponse(
      normalized,
      {
        text: 'Let me check. <tool_call>{"name":"get_weather","arguments":{"location":"London"}}</tool_call>',
        finishReason: "stop",
        usage: { promptTokens: 12, completionTokens: 9, totalTokens: 21 },
      },
      { id: "msg-tool-format" },
    );

    expect(response).toEqual({
      id: "msg-tool-format",
      type: "message",
      role: "assistant",
      content: [
        { type: "text", text: "Let me check." },
        {
          type: "tool_use",
          id: "call_1",
          name: "get_weather",
          input: { location: "London" },
        },
      ],
      model: "mlx-community/Qwen3.6-27B-4bit",
      stop_reason: "tool_use",
      stop_sequence: null,
      usage: { input_tokens: 12, output_tokens: 9 },
    });
  });

  test("leaves generated tool-call-looking text alone when tools are inactive", () => {
    const normalized = normalizeAnthropicMessageRequest(
      {
        model: "mlx-community/Qwen3.6-27B-4bit",
        messages: [{ role: "user", content: "Weather?" }],
        max_tokens: 64,
      },
      { id: "msg-no-tool-format" },
    );

    const response = formatAnthropicMessageResponse(
      normalized,
      {
        text: '<tool_call>{"name":"get_weather","arguments":{"location":"London"}}</tool_call>',
        finishReason: "stop",
      },
      { id: "msg-no-tool-format" },
    );

    expect(response.content).toEqual([
      {
        type: "text",
        text: '<tool_call>{"name":"get_weather","arguments":{"location":"London"}}</tool_call>',
      },
    ]);
    expect(response.stop_reason).toBe("end_turn");
  });

  test("rejects unsupported Anthropic shapes explicitly", () => {
    expect(() =>
      normalizeAnthropicMessageRequest(
        { model: "mlx-community/Qwen3.6-27B-4bit", messages: [{ role: "user", content: "Hi" }] },
        { id: "missing-max" },
      ),
    ).toThrow('"max_tokens" is required');

    expect(() =>
      normalizeAnthropicMessageRequest(
        {
          model: "mlx-community/Qwen3.6-27B-4bit",
          max_tokens: 8,
          messages: [{ role: "system", content: "Nope" }],
        },
        { id: "system-role" },
      ),
    ).toThrow('use top-level "system"');

    expect(() =>
      normalizeAnthropicMessageRequest(
        {
          model: "mlx-community/Qwen3.6-27B-4bit",
          max_tokens: 8,
          stream: true,
          messages: [{ role: "user", content: "Hi" }],
          tools: [{ name: "read_file", input_schema: { type: "object" } }],
        },
        { id: "tools-stream" },
      ),
    ).toThrow("streaming tool use is not supported");
  });

  test("normalizes optional Anthropic variants without sampling overrides", () => {
    const disabledThinking = normalizeAnthropicMessageRequest(
      {
        model: "mlx-community/Qwen3.6-27B-4bit",
        system: "Be brief.",
        messages: [
          { role: "user", content: "Hi" },
          { role: "assistant", content: null },
        ],
        max_tokens: 4,
        thinking: { type: "disabled" },
        chat_template_kwargs: { preserve_thinking: true },
        tools: [],
        top_k: 0,
      },
      { id: "variants" },
    );

    expect(disabledThinking.request.input).toMatchObject({
      kind: "messages",
      messages: [
        { role: "system", content: "Be brief." },
        { role: "user", content: "Hi" },
        { role: "assistant", content: "" },
      ],
      chatTemplate: { enableThinking: false, preserveThinking: true },
    });
    expect(disabledThinking.request.sampling).toEqual({ maxTokens: 4 });

    const noToolChoice = normalizeAnthropicMessageRequest(
      {
        model: "mlx-community/Qwen3.6-27B-4bit",
        messages: [{ role: "user", content: "Hi" }],
        max_tokens: 4,
        stream: true,
        tools: [{ name: "read_file", input_schema: { type: "object" } }],
        tool_choice: { type: "none" },
      },
      { id: "tool-none" },
    );

    expect(noToolChoice.request.input).toMatchObject({
      kind: "messages",
      messages: [{ role: "user", content: "Hi" }],
    });
    expect(noToolChoice.stream).toBe(true);
    expect(
      noToolChoice.request.input.kind === "messages" ? noToolChoice.request.input.tools : null,
    ).toBeUndefined();

    const assistantText = normalizeAnthropicMessageRequest(
      {
        model: "mlx-community/Qwen3.6-27B-4bit",
        messages: [{ role: "assistant", content: [{ type: "text", text: "Prefill" }] }],
        max_tokens: 1,
      },
      { id: "assistant-text" },
    );

    expect(assistantText.request.input).toMatchObject({
      kind: "messages",
      messages: [{ role: "assistant", content: "Prefill" }],
    });
  });

  test("rejects malformed Anthropic scalar and content fields", () => {
    const base = {
      model: "mlx-community/Qwen3.6-27B-4bit",
      messages: [{ role: "user", content: "Hi" }],
      max_tokens: 8,
    };

    expect(() => normalizeAnthropicMessageRequest("nope", { id: "body" })).toThrow(
      "request body must be a JSON object",
    );
    expect(() => normalizeAnthropicMessageRequest({ ...base, model: "" }, { id: "model" })).toThrow(
      '"model" must be a non-empty string',
    );
    expect(() =>
      normalizeAnthropicMessageRequest({ ...base, stream: "yes" }, { id: "stream" }),
    ).toThrow('"stream" must be a boolean');
    expect(() =>
      normalizeAnthropicMessageRequest({ ...base, temperature: 2 }, { id: "temp" }),
    ).toThrow('"temperature" must be a number between 0 and 1');
    expect(() =>
      normalizeAnthropicMessageRequest({ ...base, stop_sequences: "STOP" }, { id: "stop" }),
    ).toThrow('"stop_sequences" must be a string array');
    expect(() =>
      normalizeAnthropicMessageRequest({ ...base, metadata: [] }, { id: "metadata" }),
    ).toThrow('"metadata" must be an object');
    expect(() =>
      normalizeAnthropicMessageRequest(
        { ...base, metadata: { user_id: 123 } },
        { id: "metadata-user" },
      ),
    ).toThrow('"user_id" must be a string');
    expect(() =>
      normalizeAnthropicMessageRequest({ ...base, thinking: "on" }, { id: "thinking" }),
    ).toThrow('"thinking" must be an object');
    expect(() =>
      normalizeAnthropicMessageRequest(
        { ...base, thinking: { type: "maybe" } },
        { id: "thinking-type" },
      ),
    ).toThrow('"thinking.type" currently supports');
    expect(() =>
      normalizeAnthropicMessageRequest({ ...base, chat_template_kwargs: [] }, { id: "template" }),
    ).toThrow('"chat_template_kwargs" must be an object');
    expect(() =>
      normalizeAnthropicMessageRequest({ ...base, tools: "read_file" }, { id: "tools-shape" }),
    ).toThrow('"tools" must be an array');
    expect(() =>
      normalizeAnthropicMessageRequest(
        { ...base, tools: [{ name: "bad space", input_schema: { type: "object" } }] },
        { id: "tool-name" },
      ),
    ).toThrow('tool "name" must match');
    expect(() =>
      normalizeAnthropicMessageRequest(
        { ...base, tools: [{ name: "read_file" }] },
        { id: "tool-schema" },
      ),
    ).toThrow('tool "input_schema" must be an object');
    expect(() =>
      normalizeAnthropicMessageRequest(
        {
          ...base,
          tools: [{ name: "read_file", input_schema: { type: "object" } }],
          tool_choice: "auto",
        },
        { id: "tool-choice-shape" },
      ),
    ).toThrow('"tool_choice" must be an object');
    expect(() =>
      normalizeAnthropicMessageRequest(
        {
          ...base,
          tools: [{ name: "read_file", input_schema: { type: "object" } }],
          tool_choice: { type: "tool", name: "read_file" },
        },
        { id: "tool-choice-forced" },
      ),
    ).toThrow('tool_choice "any" and "tool" are not supported');
  });

  test("rejects malformed Anthropic message content fields", () => {
    const base = { model: "mlx-community/Qwen3.6-27B-4bit", max_tokens: 8 };

    expect(() =>
      normalizeAnthropicMessageRequest({ ...base, messages: [] }, { id: "empty" }),
    ).toThrow('"messages" must be a non-empty array');
    expect(() =>
      normalizeAnthropicMessageRequest({ ...base, messages: ["bad"] }, { id: "entry" }),
    ).toThrow('"messages" entries must be objects');
    expect(() =>
      normalizeAnthropicMessageRequest(
        { ...base, messages: [{ role: "alien", content: "Hi" }] },
        { id: "role" },
      ),
    ).toThrow('message role must be "user" or "assistant"');
    expect(() =>
      normalizeAnthropicMessageRequest(
        { ...base, messages: [{ role: "user", content: 42 }] },
        { id: "user-content" },
      ),
    ).toThrow('user "content" must be a string or content block array');
    expect(() =>
      normalizeAnthropicMessageRequest(
        { ...base, system: 42, messages: [{ role: "user", content: "Hi" }] },
        { id: "system" },
      ),
    ).toThrow('"system" must be a string');
    expect(() =>
      normalizeAnthropicMessageRequest(
        {
          ...base,
          system: [
            { type: "image", source: { type: "base64", media_type: "image/png", data: "x" } },
          ],
          messages: [{ role: "user", content: "Hi" }],
        },
        { id: "system-image" },
      ),
    ).toThrow("image content blocks are only supported in user messages");
    expect(() =>
      normalizeAnthropicMessageRequest(
        {
          ...base,
          messages: [{ role: "user", content: [123] }],
        },
        { id: "content-block-object" },
      ),
    ).toThrow("content blocks must be objects");
    expect(() =>
      normalizeAnthropicMessageRequest(
        {
          ...base,
          messages: [{ role: "user", content: [{ type: "text", text: 123 }] }],
        },
        { id: "text-block" },
      ),
    ).toThrow('text blocks require a string "text" field');
    expect(() =>
      normalizeAnthropicMessageRequest(
        {
          ...base,
          messages: [{ role: "user", content: [{ type: "audio", data: "x" }] }],
        },
        { id: "unknown-block" },
      ),
    ).toThrow("only text and image content blocks are supported");
    expect(() =>
      normalizeAnthropicMessageRequest(
        {
          ...base,
          messages: [{ role: "user", content: [{ type: "image" }] }],
        },
        { id: "image-source" },
      ),
    ).toThrow("image content blocks require a source object");
    expect(() =>
      normalizeAnthropicMessageRequest(
        {
          ...base,
          messages: [
            {
              role: "user",
              content: [{ type: "image", source: { type: "base64", media_type: "", data: "abc" } }],
            },
          ],
        },
        { id: "image-media-type" },
      ),
    ).toThrow("expected a non-empty string");
    expect(() =>
      normalizeAnthropicMessageRequest(
        {
          ...base,
          messages: [
            {
              role: "user",
              content: [{ type: "image", source: { type: "base64", media_type: "image/png" } }],
            },
          ],
        },
        { id: "image-data" },
      ),
    ).toThrow("expected a non-empty string");
    expect(() =>
      normalizeAnthropicMessageRequest(
        {
          ...base,
          messages: [{ role: "user", content: [{ type: "image", source: { type: "url" } }] }],
        },
        { id: "image-url" },
      ),
    ).toThrow("expected a non-empty string");
    expect(() =>
      normalizeAnthropicMessageRequest(
        {
          ...base,
          messages: [
            {
              role: "user",
              content: [
                {
                  type: "image",
                  source: { type: "url", url: "data:image/png;base64,abcd" },
                },
              ],
            },
          ],
        },
        { id: "image-data-url" },
      ),
    ).toThrow('data payloads use source.type "base64"');
    expect(() =>
      normalizeAnthropicMessageRequest(
        {
          ...base,
          messages: [{ role: "user", content: [{ type: "image", source: { type: "file" } }] }],
        },
        { id: "image-file" },
      ),
    ).toThrow("expected a non-empty string");
    expect(() =>
      normalizeAnthropicMessageRequest(
        {
          ...base,
          messages: [{ role: "user", content: [{ type: "image", source: { type: "blob" } }] }],
        },
        { id: "image-type" },
      ),
    ).toThrow('image source type must be "base64", "url", or "file"');
    expect(() =>
      normalizeAnthropicMessageRequest(
        {
          ...base,
          messages: [
            {
              role: "user",
              content: [
                { type: "text", text: "late" },
                { type: "tool_result", tool_use_id: "toolu_1", content: "result" },
              ],
            },
          ],
        },
        { id: "tool-result-order" },
      ),
    ).toThrow("tool_result blocks must precede");
    expect(() =>
      normalizeAnthropicMessageRequest(
        {
          ...base,
          messages: [
            {
              role: "user",
              content: [{ type: "tool_result", content: "result" }],
            },
          ],
        },
        { id: "tool-result-id" },
      ),
    ).toThrow('tool_result blocks require a non-empty "tool_use_id" field');
    expect(() =>
      normalizeAnthropicMessageRequest(
        {
          ...base,
          messages: [
            {
              role: "user",
              content: [{ type: "tool_result", tool_use_id: "toolu_1", content: "result" }],
            },
          ],
        },
        { id: "tool-result-orphan" },
      ),
    ).toThrow("tool_result blocks must immediately follow");
    expect(() =>
      normalizeAnthropicMessageRequest(
        {
          ...base,
          messages: [
            {
              role: "assistant",
              content: [
                { type: "tool_use", id: "toolu_1", name: "read_file", input: { path: "a" } },
              ],
            },
            {
              role: "user",
              content: [{ type: "tool_result", tool_use_id: "toolu_2", content: "result" }],
            },
          ],
        },
        { id: "tool-result-mismatch" },
      ),
    ).toThrow("tool_result blocks must match");
    expect(() =>
      normalizeAnthropicMessageRequest(
        {
          ...base,
          messages: [
            {
              role: "assistant",
              content: [
                { type: "tool_use", id: "toolu_1", name: "read_file", input: { path: "a" } },
              ],
            },
            {
              role: "user",
              content: "I do not have the result.",
            },
          ],
        },
        { id: "tool-result-missing" },
      ),
    ).toThrow("tool_result blocks must immediately follow");
    expect(() =>
      normalizeAnthropicMessageRequest(
        {
          ...base,
          messages: [
            {
              role: "assistant",
              content: [
                { type: "tool_use", id: "toolu_1", name: "read_file", input: { path: "a" } },
                { type: "tool_use", id: "toolu_2", name: "read_file", input: { path: "b" } },
              ],
            },
            {
              role: "user",
              content: [{ type: "tool_result", tool_use_id: "toolu_1", content: "result" }],
            },
          ],
        },
        { id: "tool-result-partial" },
      ),
    ).toThrow("tool_result blocks must match");
    expect(() =>
      normalizeAnthropicMessageRequest(
        {
          ...base,
          messages: [
            {
              role: "assistant",
              content: [
                { type: "tool_use", id: "toolu_1", name: "read_file", input: { path: "a" } },
              ],
            },
            {
              role: "user",
              content: [
                { type: "tool_result", tool_use_id: "toolu_1", content: "one" },
                { type: "tool_result", tool_use_id: "toolu_1", content: "two" },
              ],
            },
          ],
        },
        { id: "tool-result-duplicate" },
      ),
    ).toThrow("tool_result blocks must not repeat");
    expect(() =>
      normalizeAnthropicMessageRequest(
        {
          ...base,
          messages: [
            {
              role: "assistant",
              content: [
                { type: "tool_use", id: "toolu_1", name: "read_file", input: { path: "a" } },
              ],
            },
            {
              role: "user",
              content: [
                { type: "tool_result", tool_use_id: "toolu_1", content: "boom", is_error: true },
              ],
            },
          ],
        },
        { id: "tool-result-error" },
      ),
    ).toThrow("tool_result is_error=true is not supported");
    expect(() =>
      normalizeAnthropicMessageRequest(
        {
          ...base,
          messages: [
            {
              role: "assistant",
              content: [
                { type: "tool_use", id: "toolu_1", name: "read_file", input: { path: "a" } },
              ],
            },
            {
              role: "user",
              content: [
                {
                  type: "tool_result",
                  tool_use_id: "toolu_1",
                  content: [
                    {
                      type: "image",
                      source: { type: "base64", media_type: "image/png", data: "x" },
                    },
                  ],
                },
              ],
            },
          ],
        },
        { id: "tool-result-image" },
      ),
    ).toThrow("tool_result content supports string or text blocks");
  });

  test("rejects malformed Anthropic assistant content fields", () => {
    const base = { model: "mlx-community/Qwen3.6-27B-4bit", max_tokens: 8 };

    expect(() =>
      normalizeAnthropicMessageRequest(
        { ...base, messages: [{ role: "assistant", content: 42 }] },
        { id: "assistant-content" },
      ),
    ).toThrow('assistant "content" must be a string');
    expect(() =>
      normalizeAnthropicMessageRequest(
        { ...base, messages: [{ role: "assistant", content: [123] }] },
        { id: "assistant-block" },
      ),
    ).toThrow("assistant content blocks must be objects");
    expect(() =>
      normalizeAnthropicMessageRequest(
        {
          ...base,
          messages: [{ role: "assistant", content: [{ type: "text", text: 123 }] }],
        },
        { id: "assistant-text" },
      ),
    ).toThrow('assistant text blocks require a string "text" field');
    expect(() =>
      normalizeAnthropicMessageRequest(
        {
          ...base,
          messages: [{ role: "assistant", content: [{ type: "thinking", thinking: 123 }] }],
        },
        { id: "assistant-thinking" },
      ),
    ).toThrow('thinking blocks require a string "thinking" field');
    expect(() =>
      normalizeAnthropicMessageRequest(
        {
          ...base,
          messages: [
            {
              role: "assistant",
              content: [
                { type: "image", source: { type: "base64", media_type: "image/png", data: "x" } },
              ],
            },
          ],
        },
        { id: "assistant-image" },
      ),
    ).toThrow("image content blocks are only supported in user messages");
    expect(() =>
      normalizeAnthropicMessageRequest(
        {
          ...base,
          messages: [{ role: "assistant", content: [{ type: "tool_use", id: "toolu" }] }],
        },
        { id: "assistant-tool" },
      ),
    ).toThrow('tool_use blocks require a non-empty "name" field');
    expect(() =>
      normalizeAnthropicMessageRequest(
        {
          ...base,
          messages: [{ role: "assistant", content: [{ type: "redacted_thinking" }] }],
        },
        { id: "assistant-unknown" },
      ),
    ).toThrow("assistant content supports text, thinking, and tool_use blocks");
  });
});
