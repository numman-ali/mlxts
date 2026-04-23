import { describe, expect, test } from "bun:test";

import { ServeError } from "./errors";
import { createFetchHandler, startServeServer } from "./server";
import type { GenerationEngine, NormalizedGenerationRequest, ServeEvent } from "./types";

function request(path: string, body: unknown): Request {
  return new Request(`http://localhost${path}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
}

describe("serve fetch handler", () => {
  test("responds to health checks", async () => {
    const fetch = createFetchHandler({
      engine: {
        generate() {
          return { text: "", finishReason: "stop" };
        },
      },
    });

    const response = await fetch(new Request("http://localhost/health"));

    expect(response.status).toBe(200);
    expect(await response.json()).toEqual({ status: "ok" });
  });

  test("lists and retrieves served models with OpenAI-compatible shape", async () => {
    const fetch = createFetchHandler({
      engine: {
        generate() {
          return { text: "", finishReason: "stop" };
        },
      },
      models: [
        { id: "tiny", ownedBy: "test-suite", created: 42 },
        { id: "org/model", ownedBy: "slash-suite", created: 43 },
      ],
      now: () => new Date(123_000),
    });

    const response = await fetch(new Request("http://localhost/v1/models"));
    const body = await response.json();
    const retrieve = await fetch(new Request("http://localhost/v1/models/tiny"));
    const retrieveBody = await retrieve.json();
    const encodedRetrieve = await fetch(new Request("http://localhost/v1/models/org%2Fmodel"));
    const encodedRetrieveBody = await encodedRetrieve.json();
    const missing = await fetch(new Request("http://localhost/v1/models/missing"));
    const missingBody = await missing.json();
    const malformed = await fetch(new Request("http://localhost/v1/models/%E0%A4%A"));
    const malformedBody = await malformed.json();

    expect(response.status).toBe(200);
    expect(body).toEqual({
      object: "list",
      data: [
        { id: "tiny", object: "model", created: 42, owned_by: "test-suite" },
        { id: "org/model", object: "model", created: 43, owned_by: "slash-suite" },
      ],
    });
    expect(retrieve.status).toBe(200);
    expect(retrieveBody).toEqual({
      id: "tiny",
      object: "model",
      created: 42,
      owned_by: "test-suite",
    });
    expect(encodedRetrieve.status).toBe(200);
    expect(encodedRetrieveBody).toEqual({
      id: "org/model",
      object: "model",
      created: 43,
      owned_by: "slash-suite",
    });
    expect(missing.status).toBe(404);
    expect(missingBody.error.param).toBe("model");
    expect(malformed.status).toBe(400);
    expect(malformedBody.error.param).toBe("model");
  });

  test("enforces optional bearer auth on OpenAI-compatible routes", async () => {
    const fetch = createFetchHandler({
      apiKey: "secret",
      engine: {
        generate() {
          return { text: "ok", finishReason: "stop" };
        },
      },
    });

    const missing = await fetch(new Request("http://localhost/v1/models"));
    const authorized = await fetch(
      new Request("http://localhost/v1/models", {
        headers: { authorization: "Bearer secret" },
      }),
    );
    const health = await fetch(new Request("http://localhost/health"));

    expect(missing.status).toBe(401);
    expect((await missing.json()).error.code).toBe("invalid_api_key");
    expect(authorized.status).toBe(200);
    expect(health.status).toBe(200);
  });

  test("routes OpenAI completions through the normalized generation engine", async () => {
    const events: ServeEvent[] = [];
    const seen: NormalizedGenerationRequest[] = [];
    const engine: GenerationEngine = {
      generate(normalized) {
        seen.push(normalized);
        return {
          text: `echo:${normalized.input.kind === "text" ? normalized.input.text : ""}`,
          finishReason: "stop",
          usage: { promptTokens: 1, completionTokens: 1, totalTokens: 2 },
        };
      },
    };
    const fetch = createFetchHandler({
      engine,
      idGenerator: () => "cmpl-test",
      now: () => new Date(123_000),
      onEvent: (event) => events.push(event),
    });

    const response = await fetch(
      request("/v1/completions", {
        model: "tiny",
        prompt: "Hello",
        max_tokens: 4,
        temperature: 0,
      }),
    );
    const body = await response.json();

    expect(response.status).toBe(200);
    expect(seen).toHaveLength(1);
    expect(seen[0]).toMatchObject({
      id: "cmpl-test",
      model: "tiny",
      input: { kind: "text", text: "Hello" },
      sampling: { maxTokens: 4, temperature: 0 },
      protocol: "openai.completions",
    });
    expect(body.choices[0].text).toBe("echo:Hello");
    expect(body.usage).toEqual({ prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 });
    expect(
      events.filter((event) => event.type.startsWith("generation_")).map((event) => event.type),
    ).toEqual(["generation_start", "generation_complete"]);
    expect(events.find((event) => event.type === "generation_start")).toMatchObject({
      model: "tiny",
      protocol: "openai.completions",
      maxTokens: 4,
    });
  });

  test("uses batch generation for multi-prompt completion requests", async () => {
    const events: ServeEvent[] = [];
    const batchSizes: number[] = [];
    const engine: GenerationEngine = {
      generate() {
        throw new Error("generate should not be used for multi-prompt requests");
      },
      generateBatch(requests) {
        batchSizes.push(requests.length);
        return requests.map((normalized) => ({
          text: `batch:${normalized.input.kind === "text" ? normalized.input.text : ""}`,
          finishReason: "stop",
        }));
      },
    };
    const fetch = createFetchHandler({
      engine,
      idGenerator: () => "cmpl-batch",
      now: () => new Date(123_000),
      onEvent: (event) => events.push(event),
    });

    const response = await fetch(
      request("/v1/completions", {
        model: "tiny",
        prompt: ["A", "B"],
        max_tokens: 1,
      }),
    );
    const body = await response.json();

    expect(response.status).toBe(200);
    expect(batchSizes).toEqual([2]);
    expect(body.choices.map((choice: { text: string }) => choice.text)).toEqual([
      "batch:A",
      "batch:B",
    ]);
    expect(
      events.filter((event) => event.type === "generation_start").map((event) => event.id),
    ).toEqual(["cmpl-batch-0", "cmpl-batch-1"]);
    expect(
      events.filter((event) => event.type === "generation_complete").map((event) => event.id),
    ).toEqual(["cmpl-batch-0", "cmpl-batch-1"]);
  });

  test("routes OpenAI chat completions through message input", async () => {
    const seen: NormalizedGenerationRequest[] = [];
    const engine: GenerationEngine = {
      generate(normalized) {
        seen.push(normalized);
        return {
          text: `chat:${normalized.input.kind}`,
          finishReason: "stop",
          usage: { promptTokens: 4, completionTokens: 2, totalTokens: 6 },
        };
      },
    };
    const fetch = createFetchHandler({
      engine,
      idGenerator: () => "chat-test",
      now: () => new Date(123_000),
    });

    const response = await fetch(
      request("/v1/chat/completions", {
        model: "tiny",
        messages: [{ role: "user", content: "Hello" }],
        max_tokens: 4,
        temperature: 0,
      }),
    );
    const body = await response.json();

    expect(response.status).toBe(200);
    expect(seen[0]).toMatchObject({
      id: "chat-test",
      model: "tiny",
      input: { kind: "messages", messages: [{ role: "user", content: "Hello" }] },
      sampling: { maxTokens: 4, temperature: 0 },
      protocol: "openai.chat_completions",
    });
    expect(body.choices[0].message.content).toBe("chat:messages");
    expect(body.usage).toEqual({ prompt_tokens: 4, completion_tokens: 2, total_tokens: 6 });
  });

  test("routes OpenAI responses through message input", async () => {
    const events: ServeEvent[] = [];
    const seen: NormalizedGenerationRequest[] = [];
    const engine: GenerationEngine = {
      generate(normalized) {
        seen.push(normalized);
        return {
          text: `response:${normalized.input.kind}`,
          reasoningContent: "Reason briefly.",
          finishReason: "stop",
          usage: { promptTokens: 5, completionTokens: 3, totalTokens: 8 },
        };
      },
    };
    const fetch = createFetchHandler({
      engine,
      idGenerator: () => "resp-test",
      now: () => new Date(123_000),
      onEvent: (event) => events.push(event),
    });

    const response = await fetch(
      request("/v1/responses", {
        model: "tiny",
        instructions: "Be concise.",
        input: "Hello",
        max_output_tokens: 4,
        temperature: 0,
      }),
    );
    const body = await response.json();

    expect(response.status).toBe(200);
    expect(seen[0]).toMatchObject({
      id: "resp-test",
      model: "tiny",
      input: {
        kind: "messages",
        messages: [
          { role: "system", content: "Be concise." },
          { role: "user", content: "Hello" },
        ],
      },
      sampling: { maxTokens: 4, temperature: 0 },
      protocol: "openai.responses",
    });
    expect(body.object).toBe("response");
    expect(body.output_text).toBe("response:messages");
    expect(body.output[0]).toMatchObject({
      type: "reasoning",
      content: [{ type: "reasoning_text", text: "Reason briefly." }],
    });
    expect(body.output[1]).toMatchObject({
      type: "message",
      role: "assistant",
      content: [{ type: "output_text", text: "response:messages", annotations: [] }],
    });
    expect(body.usage).toMatchObject({
      input_tokens: 5,
      output_tokens: 3,
      total_tokens: 8,
    });
    expect(
      events.filter((event) => event.type.startsWith("generation_")).map((event) => event.type),
    ).toEqual(["generation_start", "generation_complete"]);
    expect(events.find((event) => event.type === "generation_start")).toMatchObject({
      model: "tiny",
      protocol: "openai.responses",
      maxTokens: 4,
    });
  });

  test("streams OpenAI chat completions as SSE chunks with reasoning separation", async () => {
    const engine: GenerationEngine = {
      generate() {
        throw new Error("generate should not be used");
      },
      async *stream() {
        yield { type: "text", text: "<think>I should greet.</think>\n\nHel" };
        yield { type: "text", text: "lo" };
        yield {
          type: "done",
          finishReason: "stop",
          usage: { promptTokens: 2, completionTokens: 4, totalTokens: 6 },
        };
      },
    };
    const fetch = createFetchHandler({
      engine,
      idGenerator: () => "chat-stream",
      now: () => new Date(123_000),
    });

    const response = await fetch(
      request("/v1/chat/completions", {
        model: "tiny",
        messages: [{ role: "user", content: "Hello" }],
        stream: true,
        stream_options: { include_usage: true },
      }),
    );
    const text = await response.text();

    expect(response.headers.get("content-type")).toContain("text/event-stream");
    expect(text).toContain('"role":"assistant"');
    expect(text).toContain('"reasoning_content":"I should greet."');
    expect(text).toContain('"content":"\\n\\nHello"');
    expect(text).toContain('"choices":[]');
    expect(text).toContain('"prompt_tokens":2');
    expect(text).toContain("data: [DONE]");
  });

  test("returns OpenAI-shaped errors for invalid OpenAI requests", async () => {
    const fetch = createFetchHandler({
      engine: {
        generate() {
          return { text: "", finishReason: "stop" };
        },
      },
    });

    const response = await fetch(
      request("/v1/completions", {
        model: "tiny",
        prompt: "Hello",
        n: 2,
      }),
    );
    const body = await response.json();

    expect(response.status).toBe(400);
    expect(body.error.param).toBe("n");

    const chatResponse = await fetch(
      request("/v1/chat/completions", {
        model: "tiny",
        messages: [{ role: "user", content: "Hello" }],
        n: 2,
      }),
    );
    const chatBody = await chatResponse.json();

    expect(chatResponse.status).toBe(400);
    expect(chatBody.error.param).toBe("n");

    const responseResponse = await fetch(
      request("/v1/responses", {
        model: "tiny",
        input: [{ role: "user", content: "Hello" }],
      }),
    );
    const responseBody = await responseResponse.json();

    expect(responseResponse.status).toBe(400);
    expect(responseBody.error.param).toBe("input");
  });

  test("emits generation errors with normalized request context", async () => {
    const events: ServeEvent[] = [];
    const fetch = createFetchHandler({
      engine: {
        generate(normalized) {
          throw new ServeError(`Model "${normalized.model}" is not served by this endpoint.`, {
            code: "model_not_found",
            param: "model",
            status: 404,
          });
        },
      },
      idGenerator: () => "chat-error",
      onEvent: (event) => events.push(event),
    });

    const response = await fetch(
      request("/v1/chat/completions", {
        model: "missing-local",
        messages: [{ role: "user", content: "Hello" }],
      }),
    );
    const body = await response.json();

    expect(response.status).toBe(404);
    expect(body.error.code).toBe("model_not_found");
    expect(events.filter((event) => event.type.startsWith("generation_"))).toMatchObject([
      {
        type: "generation_start",
        id: "chat-error",
        model: "missing-local",
        protocol: "openai.chat_completions",
      },
      {
        type: "generation_error",
        id: "chat-error",
        model: "missing-local",
        protocol: "openai.chat_completions",
        code: "model_not_found",
      },
    ]);
    expect(events.find((event) => event.type === "request_error")).toMatchObject({
      code: "model_not_found",
      status: 404,
    });
  });

  test("returns OpenAI-shaped server errors for malformed batch engine results", async () => {
    const events: ServeEvent[] = [];
    const fetch = createFetchHandler({
      engine: {
        generate() {
          throw new Error("generate should not be called");
        },
        generateBatch() {
          return [];
        },
      },
      idGenerator: () => "cmpl-batch",
      onEvent: (event) => events.push(event),
    });

    const response = await fetch(
      request("/v1/completions", {
        model: "tiny",
        prompt: ["A", "B"],
      }),
    );
    const body = await response.json();

    expect(response.status).toBe(500);
    expect(body.error.code).toBe("invalid_engine_result");
    expect(
      events.filter((event) => event.type === "generation_error").map((event) => event.id),
    ).toEqual(["cmpl-batch-0", "cmpl-batch-1"]);
  });

  test("streams OpenAI completions as SSE chunks", async () => {
    const events: ServeEvent[] = [];
    const engine: GenerationEngine = {
      generate() {
        return { text: "", finishReason: "stop" };
      },
      async *stream() {
        yield { type: "text", text: "Hel" };
        yield { type: "text", text: "lo" };
        yield {
          type: "done",
          finishReason: "stop",
          usage: { promptTokens: 1, completionTokens: 2, totalTokens: 3 },
        };
      },
    };
    const fetch = createFetchHandler({
      engine,
      idGenerator: () => "cmpl-stream",
      now: () => new Date(123_000),
      onEvent: (event) => events.push(event),
    });

    const response = await fetch(
      request("/v1/completions", {
        model: "tiny",
        prompt: "Hello",
        stream: true,
        stream_options: { include_usage: true },
      }),
    );
    const text = await response.text();

    expect(response.headers.get("content-type")).toContain("text/event-stream");
    expect(text).toContain('"text":"Hel"');
    expect(text).toContain('"text":"lo"');
    expect(text).toContain('"choices":[]');
    expect(text).toContain('"prompt_tokens":1');
    expect(text).toContain("data: [DONE]");
    expect(
      events.filter((event) => event.type.startsWith("generation_")).map((event) => event.type),
    ).toEqual(["generation_start", "generation_complete"]);
  });

  test("keeps streaming requests alive with Bun timeout override and completes them after the body ends", async () => {
    const events: ServeEvent[] = [];
    const timeouts: number[] = [];
    const engine: GenerationEngine = {
      generate() {
        throw new Error("generate should not be used");
      },
      async *stream() {
        yield { type: "text", text: "Hello" };
        yield {
          type: "done",
          finishReason: "stop",
          usage: { promptTokens: 1, completionTokens: 1, totalTokens: 2 },
        };
      },
    };
    const fetch = createFetchHandler({
      engine,
      idGenerator: () => "cmpl-timeout",
      now: () => new Date(123_000),
      onEvent: (event) => events.push(event),
    });

    const response = await fetch(
      request("/v1/completions", {
        model: "tiny",
        prompt: "Hello",
        stream: true,
      }),
      {
        timeout(_request, seconds) {
          timeouts.push(seconds);
        },
      },
    );
    const text = await response.text();

    expect(timeouts).toEqual([0]);
    expect(response.headers.get("content-type")).toContain("text/event-stream");
    expect(text).toContain("data: [DONE]");
    expect(events.filter((event) => event.type === "request_complete")).toHaveLength(1);
  });

  test("cancels streaming chat responses when the client disconnects", async () => {
    const events: ServeEvent[] = [];
    let streamClosed = false;
    const engine: GenerationEngine = {
      generate() {
        throw new Error("generate should not be used");
      },
      async *stream() {
        try {
          yield { type: "text", text: "Hello" };
          await new Promise((resolve) => setTimeout(resolve, 10));
          yield {
            type: "done",
            finishReason: "stop",
            usage: { promptTokens: 2, completionTokens: 1, totalTokens: 3 },
          };
        } finally {
          streamClosed = true;
        }
      },
    };
    const fetch = createFetchHandler({
      engine,
      idGenerator: () => "chat-cancel",
      now: () => new Date(123_000),
      onEvent: (event) => events.push(event),
    });

    const response = await fetch(
      request("/v1/chat/completions", {
        model: "tiny",
        messages: [{ role: "user", content: "Hello" }],
        stream: true,
      }),
    );

    expect(events.map((event) => event.type)).toEqual(["request_start", "generation_start"]);

    const reader = response.body?.getReader();
    expect(reader).toBeDefined();
    if (reader === undefined) {
      throw new Error("expected a response body reader");
    }

    const firstChunk = await reader.read();
    expect(firstChunk.done).toBe(false);
    await reader.cancel();
    await new Promise((resolve) => setTimeout(resolve, 25));

    expect(streamClosed).toBe(true);
    expect(events.find((event) => event.type === "generation_complete")).toMatchObject({
      finishReason: "cancelled",
    });
    expect(events.find((event) => event.type === "request_error")).toMatchObject({
      code: "client_cancelled",
      status: 499,
    });
    expect(events.find((event) => event.type === "request_complete")).toBeUndefined();
  });

  test("rejects unsupported streaming shapes and unknown routes", async () => {
    const fetch = createFetchHandler({
      engine: {
        generate() {
          return { text: "", finishReason: "stop" };
        },
      },
    });

    const missingStream = await fetch(
      request("/v1/completions", {
        model: "tiny",
        prompt: "Hello",
        stream: true,
      }),
    );
    const multiPrompt = await fetch(
      request("/v1/completions", {
        model: "tiny",
        prompt: ["a", "b"],
        stream: true,
      }),
    );
    const missingChatStream = await fetch(
      request("/v1/chat/completions", {
        model: "tiny",
        messages: [{ role: "user", content: "Hello" }],
        stream: true,
      }),
    );
    const responseStream = await fetch(
      request("/v1/responses", {
        model: "tiny",
        input: "Hello",
        stream: true,
      }),
    );
    const notFound = await fetch(new Request("http://localhost/v1/chat/completions"));

    expect(missingStream.status).toBe(400);
    expect((await missingStream.json()).error.code).toBe("stream_not_supported");
    expect(multiPrompt.status).toBe(400);
    expect((await multiPrompt.json()).error.param).toBe("prompt");
    expect(missingChatStream.status).toBe(400);
    expect((await missingChatStream.json()).error.code).toBe("stream_not_supported");
    expect(responseStream.status).toBe(400);
    expect((await responseStream.json()).error.param).toBe("stream");
    expect(notFound.status).toBe(404);
  });

  test("rejects malformed JSON before it reaches the engine", async () => {
    const fetch = createFetchHandler({
      engine: {
        generate() {
          throw new Error("should not be called");
        },
      },
    });

    const response = await fetch(
      new Request("http://localhost/v1/completions", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: "{",
      }),
    );
    const body = await response.json();

    expect(response.status).toBe(400);
    expect(body.error.code).toBe("invalid_json");
  });

  test("starts a Bun server with the shared fetch handler", () => {
    const server = startServeServer({
      hostname: "127.0.0.1",
      port: 0,
      engine: {
        generate() {
          return { text: "", finishReason: "stop" };
        },
      },
    });

    try {
      expect(server.port).toBeGreaterThan(0);
    } finally {
      server.stop(true);
    }
  });
});
