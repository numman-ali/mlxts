import { describe, expect, test } from "bun:test";

import {
  formatAnthropicMessageResponse,
  normalizeAnthropicMessageRequest,
} from "./anthropic-messages";

describe("Anthropic messages adapter", () => {
  test("normalizes text-only Messages requests into protocol-neutral message input", () => {
    const normalized = normalizeAnthropicMessageRequest(
      {
        model: "qwen-local",
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
      model: "qwen-local",
      stream: true,
      maxTokens: 64,
      temperature: 0.7,
      topP: 0.95,
      topK: 20,
      request: {
        id: "msg-test",
        model: "qwen-local",
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

  test("formats visible and thinking output as Anthropic content blocks", () => {
    const normalized = normalizeAnthropicMessageRequest(
      {
        model: "qwen-local",
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
      model: "qwen-local",
      stop_reason: "max_tokens",
      stop_sequence: null,
      usage: { input_tokens: 5, output_tokens: 8 },
    });
  });

  test("rejects unsupported Anthropic shapes explicitly", () => {
    expect(() =>
      normalizeAnthropicMessageRequest(
        { model: "qwen-local", messages: [{ role: "user", content: "Hi" }] },
        { id: "missing-max" },
      ),
    ).toThrow('"max_tokens" is required');

    expect(() =>
      normalizeAnthropicMessageRequest(
        {
          model: "qwen-local",
          max_tokens: 8,
          messages: [{ role: "system", content: "Nope" }],
        },
        { id: "system-role" },
      ),
    ).toThrow('use top-level "system"');

    expect(() =>
      normalizeAnthropicMessageRequest(
        {
          model: "qwen-local",
          max_tokens: 8,
          messages: [
            {
              role: "user",
              content: [
                { type: "image", source: { type: "url", url: "https://example.com/a.png" } },
              ],
            },
          ],
        },
        { id: "image" },
      ),
    ).toThrow("image content blocks are not supported");

    expect(() =>
      normalizeAnthropicMessageRequest(
        {
          model: "qwen-local",
          max_tokens: 8,
          messages: [{ role: "user", content: "Hi" }],
          tools: [{ name: "read_file" }],
        },
        { id: "tools" },
      ),
    ).toThrow("tools are not supported");
  });

  test("normalizes optional Anthropic variants without sampling overrides", () => {
    const disabledThinking = normalizeAnthropicMessageRequest(
      {
        model: "qwen-local",
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

    const assistantText = normalizeAnthropicMessageRequest(
      {
        model: "qwen-local",
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
      model: "qwen-local",
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
  });

  test("rejects malformed Anthropic message content fields", () => {
    const base = { model: "qwen-local", max_tokens: 8 };

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
    ).toThrow('user "content" must be a string');
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
    ).toThrow("only text content blocks are supported");
  });

  test("rejects malformed Anthropic assistant content fields", () => {
    const base = { model: "qwen-local", max_tokens: 8 };

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
          messages: [{ role: "assistant", content: [{ type: "tool_use", id: "toolu" }] }],
        },
        { id: "assistant-tool" },
      ),
    ).toThrow("tool content blocks are not supported");
    expect(() =>
      normalizeAnthropicMessageRequest(
        {
          ...base,
          messages: [{ role: "assistant", content: [{ type: "redacted_thinking" }] }],
        },
        { id: "assistant-unknown" },
      ),
    ).toThrow("assistant content supports text and thinking blocks");
  });
});
