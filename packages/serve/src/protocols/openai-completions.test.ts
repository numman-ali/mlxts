import { describe, expect, test } from "bun:test";

import { ServeError } from "../errors";
import {
  formatOpenAICompletionResponse,
  formatOpenAICompletionStreamChunk,
  formatOpenAICompletionUsageStreamChunk,
  type NormalizedCompletionBatch,
  normalizeOpenAICompletionRequest,
} from "./openai-completions";

function firstRequest(batch: NormalizedCompletionBatch) {
  const request = batch.requests[0];
  if (request === undefined) {
    throw new Error("expected at least one normalized request");
  }
  return request;
}

describe("OpenAI completions adapter", () => {
  test("normalizes a completion request into protocol-neutral requests", () => {
    const batch = normalizeOpenAICompletionRequest(
      {
        model: "tiny",
        prompt: ["Hello", "World"],
        max_tokens: 8,
        temperature: 0,
        top_k: 20,
        top_p: 0.9,
        stop: ["</s>"],
      },
      { id: "cmpl-test" },
    );

    expect(batch.model).toBe("tiny");
    expect(batch.stream).toBe(false);
    expect(batch.requests).toHaveLength(2);
    expect(batch.requests[0]).toMatchObject({
      id: "cmpl-test-0",
      model: "tiny",
      input: { kind: "text", text: "Hello" },
      sampling: { maxTokens: 8, temperature: 0, topK: 20, topP: 0.9, stop: ["</s>"] },
      protocol: "openai.completions",
    });
    expect(batch.requests[1]?.input).toEqual({ kind: "text", text: "World" });

    const tokenBatch = normalizeOpenAICompletionRequest(
      {
        model: "tiny",
        prompt: [
          [1, 2],
          [3, 4],
        ],
      },
      { id: "cmpl-tokens" },
    );
    expect(tokenBatch.requests.map((request) => request.input)).toEqual([
      { kind: "tokens", tokenIds: [1, 2] },
      { kind: "tokens", tokenIds: [3, 4] },
    ]);
  });

  test("applies model-native sampling defaults and rejects unsupported options", () => {
    const batch = normalizeOpenAICompletionRequest(
      {
        model: "tiny",
        prompt: "Hello",
        max_tokens: null,
        temperature: null,
        top_p: null,
        stream: null,
        n: 1,
        best_of: 1,
        echo: false,
        frequency_penalty: 0,
        presence_penalty: 0,
        logit_bias: {},
        logprobs: null,
        suffix: null,
        seed: 123,
        user: "user-1",
      },
      { id: "cmpl-test" },
    );

    expect(batch.requests[0]?.sampling.maxTokens).toBe(16);
    expect(batch.requests[0]?.sampling.temperature).toBeUndefined();
    expect(batch.requests[0]?.sampling.topP).toBeUndefined();
    expect(batch.requests[0]?.sampling.seed).toBe(123);
    expect(batch.requests[0]?.metadata?.user).toBe("user-1");
    expect(() =>
      normalizeOpenAICompletionRequest({ model: "tiny", prompt: "Hello", n: 2 }, { id: "bad" }),
    ).toThrow(ServeError);
    expect(() =>
      normalizeOpenAICompletionRequest(
        { model: "tiny", prompt: "Hello", logprobs: 1 },
        { id: "bad" },
      ),
    ).toThrow(ServeError);
    expect(() =>
      normalizeOpenAICompletionRequest(
        { model: "tiny", prompt: "Hello", presence_penalty: 0.5 },
        { id: "bad" },
      ),
    ).toThrow(ServeError);
    expect(() =>
      normalizeOpenAICompletionRequest(
        { model: "tiny", prompt: "Hello", stream_options: { include_usage: true } },
        { id: "bad" },
      ),
    ).toThrow(ServeError);
    expect(() =>
      normalizeOpenAICompletionRequest({ model: "", prompt: "Hello" }, { id: "bad" }),
    ).toThrow(ServeError);
    expect(() =>
      normalizeOpenAICompletionRequest(
        { model: "tiny", prompt: "Hello", stream: "yes" },
        { id: "bad" },
      ),
    ).toThrow(ServeError);
    expect(() =>
      normalizeOpenAICompletionRequest(
        { model: "tiny", prompt: "Hello", max_tokens: -1 },
        { id: "bad" },
      ),
    ).toThrow(ServeError);
    expect(() =>
      normalizeOpenAICompletionRequest({ model: "tiny", prompt: 42 }, { id: "bad" }),
    ).toThrow(ServeError);
    expect(() =>
      normalizeOpenAICompletionRequest(
        { model: "tiny", prompt: "Hello", stop: ["a", "b", "c", "d", "e"] },
        { id: "bad" },
      ),
    ).toThrow(ServeError);
  });

  test("rejects unsupported non-default OpenAI completion semantics explicitly", () => {
    const invalidBodies: Record<string, unknown>[] = [
      { best_of: 2 },
      { echo: true },
      { frequency_penalty: -0.5 },
      { logit_bias: { "1": 10 } },
      { suffix: "after" },
      { suffix: 1 },
      { temperature: 3 },
      { top_p: 0 },
      { top_k: 0 },
      { seed: -1 },
      { stream: true, stream_options: { include_obfuscation: true } },
      { prompt: [] },
      { prompt: [1, "bad"] },
      { stop: [42] },
    ];

    for (const body of invalidBodies) {
      expect(() =>
        normalizeOpenAICompletionRequest(
          {
            model: "tiny",
            prompt: "Hello",
            ...body,
          },
          { id: "bad" },
        ),
      ).toThrow(ServeError);
    }
  });

  test("formats completion responses and stream chunks", () => {
    const batch = normalizeOpenAICompletionRequest(
      { model: "tiny", prompt: "Hello", max_tokens: 2 },
      { id: "cmpl-test" },
    );
    const response = formatOpenAICompletionResponse(
      batch,
      [
        {
          text: " there",
          finishReason: "eos",
          usage: { promptTokens: 1, completionTokens: 2, totalTokens: 3 },
        },
      ],
      { id: "cmpl-test", created: 123 },
    );

    expect(response).toEqual({
      id: "cmpl-test",
      object: "text_completion",
      created: 123,
      model: "tiny",
      choices: [{ text: " there", index: 0, logprobs: null, finish_reason: "stop" }],
      usage: { prompt_tokens: 1, completion_tokens: 2, total_tokens: 3 },
    });

    expect(
      formatOpenAICompletionStreamChunk(firstRequest(batch), "a", {
        id: "cmpl-test",
        created: 123,
        includeUsage: true,
      }).choices[0]?.finish_reason,
    ).toBeNull();
    expect(
      formatOpenAICompletionStreamChunk(firstRequest(batch), "a", {
        id: "cmpl-test",
        created: 123,
        includeUsage: true,
      }).usage,
    ).toBeNull();
    expect(
      formatOpenAICompletionStreamChunk(firstRequest(batch), "", {
        id: "cmpl-test",
        created: 123,
        finishReason: "length",
      }).choices[0]?.finish_reason,
    ).toBe("length");
    expect(
      formatOpenAICompletionStreamChunk(firstRequest(batch), "", {
        id: "cmpl-test",
        created: 123,
        finishReason: "cancelled",
      }).choices[0]?.finish_reason,
    ).toBeNull();
    expect(
      formatOpenAICompletionStreamChunk(firstRequest(batch), "", {
        id: "cmpl-test",
        created: 123,
        finishReason: "error",
      }).choices[0]?.finish_reason,
    ).toBe("content_filter");
    expect(
      formatOpenAICompletionUsageStreamChunk(
        { ...batch, streamOptions: { includeUsage: true } },
        undefined,
        { id: "cmpl-test", created: 123 },
      ).usage,
    ).toBeNull();
    expect(
      formatOpenAICompletionUsageStreamChunk(
        { ...batch, streamOptions: { includeUsage: true } },
        { promptTokens: 1, completionTokens: 2, totalTokens: 3 },
        { id: "cmpl-test", created: 123 },
      ),
    ).toMatchObject({
      choices: [],
      usage: { prompt_tokens: 1, completion_tokens: 2, total_tokens: 3 },
    });
  });
});
