import { describe, expect, test } from "bun:test";

import { ServeError } from "../errors";
import { formatOpenAIResponse, normalizeOpenAIResponseRequest } from "./openai-responses";

describe("OpenAI Responses adapter", () => {
  test("normalizes a text-only response request into message input", () => {
    const response = normalizeOpenAIResponseRequest(
      {
        model: "tiny",
        instructions: "Be concise.",
        input: "Hello",
        max_output_tokens: 8,
        temperature: 0,
        top_p: 0.9,
        top_k: 20,
        seed: 123,
        metadata: { trace: "abc" },
        user: "user-1",
        prompt_cache_key: "cache-1",
        store: false,
        tool_choice: "none",
        parallel_tool_calls: false,
        text: { format: { type: "text" } },
        modalities: ["text"],
        truncation: "disabled",
        chat_template_kwargs: { enable_thinking: false },
      },
      { id: "resp-test" },
    );

    expect(response.model).toBe("tiny");
    expect(response.stream).toBe(false);
    expect(response.streamOptions).toEqual({ includeObfuscation: false });
    expect(response.instructions).toBe("Be concise.");
    expect(response.maxOutputTokens).toBe(8);
    expect(response.temperature).toBe(0);
    expect(response.topP).toBe(0.9);
    expect(response.toolChoice).toBe("none");
    expect(response.parallelToolCalls).toBe(false);
    expect(response.metadata).toEqual({ trace: "abc" });
    expect(response.user).toBe("user-1");
    expect(response.request).toMatchObject({
      id: "resp-test",
      model: "tiny",
      input: {
        kind: "messages",
        messages: [
          { role: "system", content: "Be concise." },
          { role: "user", content: "Hello" },
        ],
        chatTemplate: { enableThinking: false },
      },
      sampling: { maxTokens: 8, temperature: 0, topP: 0.9, topK: 20, seed: 123 },
      stream: false,
      protocol: "openai.responses",
      metadata: { trace: "abc", user: "user-1", promptCacheKey: "cache-1" },
    });
  });

  test("keeps model-native sampling defaults when optional fields are omitted", () => {
    const response = normalizeOpenAIResponseRequest(
      {
        model: "tiny",
        input: "Hello",
        max_output_tokens: null,
        temperature: null,
        top_p: null,
        store: null,
        tools: [],
      },
      { id: "resp-test" },
    );

    expect(response.maxOutputTokens).toBeNull();
    expect(response.temperature).toBeNull();
    expect(response.topP).toBeNull();
    expect(response.request.sampling.maxTokens).toBe(16);
    expect(response.request.sampling.temperature).toBeUndefined();
    expect(response.request.sampling.topP).toBeUndefined();
    expect(response.request.input).toEqual({
      kind: "messages",
      messages: [{ role: "user", content: "Hello" }],
    });
  });

  test("normalizes text-only response item arrays and stream options", () => {
    const response = normalizeOpenAIResponseRequest(
      {
        model: "tiny",
        input: [
          { role: "developer", content: [{ type: "input_text", text: "Be direct." }] },
          { role: "user", content: [{ type: "input_text", text: "Hello" }] },
          { role: "assistant", content: [{ type: "output_text", text: "Hi." }] },
          { type: "message", role: "user", content: "Continue" },
        ],
        stream: true,
        stream_options: { include_obfuscation: false },
        stop: ["END"],
        enable_thinking: false,
      },
      { id: "resp-test" },
    );

    expect(response.stream).toBe(true);
    expect(response.streamOptions).toEqual({ includeObfuscation: false });
    expect(response.request).toMatchObject({
      input: {
        kind: "messages",
        messages: [
          { role: "system", content: "Be direct." },
          { role: "user", content: "Hello" },
          { role: "assistant", content: "Hi." },
          { role: "user", content: "Continue" },
        ],
        chatTemplate: { enableThinking: false },
      },
      sampling: { maxTokens: 16, stop: ["END"] },
      stream: true,
    });
  });

  test("rejects unsupported response shapes explicitly", () => {
    const invalidBodies: Record<string, unknown>[] = [
      { input: [] },
      { input: "" },
      { input: [42] },
      { input: [{ role: "bad", content: "Hello" }] },
      { input: [{ role: "user", content: [{}] }] },
      { input: [{ role: "user", content: [42] }] },
      { input: [{ role: "user", content: [{ type: "input_text", text: 42 }] }] },
      { input: [{ role: "user", content: [{ type: "input_image", image_url: "data:" }] }] },
      { input: [{ role: "user", content: [{ type: "input_file", file_data: "" }] }] },
      { input: [{ role: "user", content: [{ type: "input_audio", input_audio: "" }] }] },
      { input: [{ role: "user", content: { text: "Hello" } }] },
      { input: [{ type: "function_call_output", call_id: "call", output: "ok" }] },
      { input: [{ type: "reasoning", content: [] }] },
      { input: [{ type: "function_call", name: "tool", arguments: "{}" }] },
      { tools: [{ type: "web_search_preview" }] },
      { tools: "bad" },
      { store: true },
      { background: true },
      { previous_response_id: "resp-old" },
      { conversation: "conv-1" },
      { prompt: { id: "prompt-1" } },
      { include: ["reasoning.encrypted_content"] },
      { max_tool_calls: 1 },
      { stream_options: {} },
      { stream: true, stream_options: "bad" },
      { stream: true, stream_options: { include_obfuscation: true } },
      { reasoning: {} },
      { text: "bad" },
      { text: { format: { type: "json_object" } } },
      { modalities: ["audio"] },
      { truncation: "auto" },
      { tool_choice: { type: "function" } },
      { metadata: { bad: 42 } },
      { metadata: "bad" },
      { instructions: 42 },
      { stream: "yes" },
      { chat_template_kwargs: "bad" },
      { temperature: 3 },
      { top_p: 0 },
      { top_k: 0 },
      { seed: -1 },
      { model: "" },
    ];

    for (const body of invalidBodies) {
      expect(() =>
        normalizeOpenAIResponseRequest(
          {
            model: "tiny",
            input: "Hello",
            ...body,
          },
          { id: "bad" },
        ),
      ).toThrow(ServeError);
    }
  });

  test("formats a generation result as an OpenAI response object", () => {
    const response = normalizeOpenAIResponseRequest(
      {
        model: "tiny",
        instructions: "Be helpful.",
        input: "Hello",
        max_output_tokens: 2,
        temperature: 0,
        top_p: 1,
        metadata: { trace: "abc" },
        safety_identifier: "safe-user",
      },
      { id: "resp-test" },
    );

    const body = formatOpenAIResponse(
      response,
      {
        text: "Hi there.",
        reasoningContent: "Greet the user.",
        finishReason: "eos",
        usage: { promptTokens: 3, completionTokens: 2, totalTokens: 5 },
      },
      { id: "resp-test", created: 123 },
    );

    expect(body).toMatchObject({
      id: "resp-test",
      object: "response",
      created_at: 123,
      status: "completed",
      completed_at: 123,
      error: null,
      incomplete_details: null,
      instructions: "Be helpful.",
      max_output_tokens: 2,
      model: "tiny",
      output_text: "Hi there.",
      parallel_tool_calls: true,
      previous_response_id: null,
      reasoning: { effort: null, summary: null },
      store: false,
      temperature: 0,
      text: { format: { type: "text" } },
      tool_choice: "auto",
      tools: [],
      top_p: 1,
      truncation: "disabled",
      usage: {
        input_tokens: 3,
        input_tokens_details: { cached_tokens: 0 },
        output_tokens: 2,
        output_tokens_details: { reasoning_tokens: 0 },
        total_tokens: 5,
      },
      user: "safe-user",
      metadata: { trace: "abc" },
    });
    expect(body.output).toEqual([
      {
        id: "resp-test-rsn",
        type: "reasoning",
        status: "completed",
        summary: [],
        content: [{ type: "reasoning_text", text: "Greet the user." }],
      },
      {
        id: "resp-test-msg",
        type: "message",
        status: "completed",
        role: "assistant",
        content: [{ type: "output_text", text: "Hi there.", annotations: [] }],
      },
    ]);
  });

  test("marks length-limited results as incomplete", () => {
    const response = normalizeOpenAIResponseRequest(
      { model: "tiny", input: "Hello", max_output_tokens: 1 },
      { id: "resp-test" },
    );
    const body = formatOpenAIResponse(
      response,
      { text: "Hi", finishReason: "length" },
      { id: "resp-test", created: 123 },
    );

    expect(body.status).toBe("incomplete");
    expect(body.completed_at).toBeNull();
    expect(body.incomplete_details).toEqual({ reason: "max_output_tokens" });
    expect(body.usage).toBeNull();
  });
});
