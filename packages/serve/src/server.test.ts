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

function endpointFor(server: ReturnType<typeof Bun.serve>): string {
  return `http://${server.hostname}:${server.port}`;
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

  test("reports lightweight server info without invoking the engine", async () => {
    let invoked = false;
    const fetch = createFetchHandler({
      engine: {
        generate() {
          invoked = true;
          return { text: "", finishReason: "stop" };
        },
        generateBatch() {
          invoked = true;
          return [];
        },
        async *stream() {
          invoked = true;
          yield { type: "text", text: "" };
        },
      },
      models: [{ id: "tiny" }, { id: "qwen-local" }],
      limits: {
        maxGeneratedTokens: 2048,
        maxPromptTokens: 4096,
        maxTotalTokens: 4096,
        maxBatchSize: 32,
        batchWindowMs: 1,
        streamDecodeInterval: 1,
        maxConcurrentRequests: 1,
        gpuMemoryUtilization: 0.9,
      },
    });

    const response = await fetch(new Request("http://localhost/info"));
    const body = await response.json();

    expect(response.status).toBe(200);
    expect(invoked).toBe(false);
    expect(body).toMatchObject({
      status: "ok",
      router: "@mlxts/serve",
      version: null,
      model_id: "tiny",
      model_ids: ["tiny", "qwen-local"],
      model_count: 2,
      limits: {
        max_generated_tokens: 2048,
        max_prompt_tokens: 4096,
        max_total_tokens: 4096,
        max_client_batch_size: 32,
        batch_window_ms: 1,
        stream_decode_interval: 1,
        max_concurrent_requests: 1,
        gpu_memory_utilization: 0.9,
      },
      models: [
        {
          id: "tiny",
          context_window: null,
          max_prompt_tokens: 4096,
          max_total_tokens: 4096,
          effective_total_tokens: null,
        },
        {
          id: "qwen-local",
          context_window: null,
          max_prompt_tokens: 4096,
          max_total_tokens: 4096,
          effective_total_tokens: null,
        },
      ],
      capabilities: {
        completions: true,
        chat_completions: true,
        responses: "text_only",
        sse_streaming: true,
        batch_generation: true,
        reasoning_content: true,
        tool_calls: true,
      },
      runtime_strategy: {
        scheduler: {
          mode: "auto",
          max_batch_size: 32,
          batch_window_ms: 1,
          max_concurrent_requests: 1,
        },
        cache: {
          backend: "managed",
          precision: "model",
        },
        attention: {
          backend: "auto",
        },
        decoding: {
          backend: "model",
        },
        streaming: {
          stream_decode_interval: 1,
        },
        memory: {
          policy: "admit_only",
          gpu_memory_utilization: 0.9,
        },
      },
    });
    expect(body.endpoints).toContain("/v1/responses");
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
    const infoMissing = await fetch(new Request("http://localhost/info"));
    const authorized = await fetch(
      new Request("http://localhost/v1/models", {
        headers: { authorization: "Bearer secret" },
      }),
    );
    const infoAuthorized = await fetch(
      new Request("http://localhost/info", {
        headers: { authorization: "Bearer secret" },
      }),
    );
    const health = await fetch(new Request("http://localhost/health"));

    expect(missing.status).toBe(401);
    expect((await missing.json()).error.code).toBe("invalid_api_key");
    expect(infoMissing.status).toBe(401);
    expect((await infoMissing.json()).error.code).toBe("invalid_api_key");
    expect(authorized.status).toBe(200);
    expect(infoAuthorized.status).toBe(200);
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
    expect(seen[0]?.abortSignal).toBeDefined();
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

  test("disables Bun request timeout before buffered generation starts", async () => {
    const timeouts: number[] = [];
    let timeoutDisabledBeforeGenerate = false;
    const engine: GenerationEngine = {
      generate() {
        timeoutDisabledBeforeGenerate = timeouts.includes(0);
        return {
          text: "ok",
          finishReason: "stop",
          usage: { promptTokens: 1, completionTokens: 1, totalTokens: 2 },
        };
      },
    };
    const fetch = createFetchHandler({
      engine,
      idGenerator: () => "cmpl-buffered-timeout",
      now: () => new Date(123_000),
    });

    const response = await fetch(
      request("/v1/completions", {
        model: "tiny",
        prompt: "Hello",
        max_tokens: 1,
      }),
      {
        timeout(_request, seconds) {
          timeouts.push(seconds);
        },
      },
    );
    const body = await response.json();

    expect(response.status).toBe(200);
    expect(timeouts).toEqual([0]);
    expect(timeoutDisabledBeforeGenerate).toBe(true);
    expect(body.choices[0].text).toBe("ok");
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
    expect(seen[0]?.abortSignal).toBeDefined();
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
        input: [{ role: "user", content: [{ type: "input_text", text: "Hello" }] }],
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
    expect(seen[0]?.abortSignal).toBeDefined();
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

  test("streams OpenAI responses as semantic SSE events with reasoning separation", async () => {
    const events: ServeEvent[] = [];
    const seen: NormalizedGenerationRequest[] = [];
    const engine: GenerationEngine = {
      generate() {
        throw new Error("generate should not be used");
      },
      async *stream(normalized) {
        seen.push(normalized);
        yield { type: "text", text: "<think>Reason briefly.</think>Hel" };
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
      idGenerator: () => "resp-stream",
      now: () => new Date(123_000),
      onEvent: (event) => events.push(event),
    });

    const response = await fetch(
      request("/v1/responses", {
        model: "tiny",
        input: [{ role: "user", content: "Hello" }],
        stream: true,
        stream_options: { include_obfuscation: false },
      }),
    );
    const text = await response.text();

    expect(response.headers.get("content-type")).toContain("text/event-stream");
    expect(seen[0]).toMatchObject({
      id: "resp-stream",
      model: "tiny",
      input: { kind: "messages", messages: [{ role: "user", content: "Hello" }] },
      stream: true,
      protocol: "openai.responses",
    });
    expect(text).toContain("event: response.created");
    expect(text).toContain("event: response.in_progress");
    expect(text).toContain("event: response.reasoning_text.delta");
    expect(text).toContain('"delta":"Reason briefly."');
    expect(text).toContain("event: response.output_text.delta");
    expect(text).toContain('"delta":"Hello"');
    expect(text).toContain("event: response.output_text.done");
    expect(text).toContain("event: response.completed");
    expect(text).toContain('"output_text":"Hello"');
    expect(text).toContain("data: [DONE]");
    expect(
      events.filter((event) => event.type.startsWith("generation_")).map((event) => event.type),
    ).toEqual(["generation_start", "generation_complete"]);
  });

  test("streams OpenAI responses length stops as incomplete terminal events", async () => {
    const engine: GenerationEngine = {
      generate() {
        throw new Error("generate should not be used");
      },
      async *stream() {
        yield { type: "text", text: "Partial" };
        yield {
          type: "done",
          finishReason: "length",
          usage: { promptTokens: 2, completionTokens: 1, totalTokens: 3 },
        };
      },
    };
    const fetch = createFetchHandler({
      engine,
      idGenerator: () => "resp-length",
      now: () => new Date(123_000),
    });

    const response = await fetch(
      request("/v1/responses", {
        model: "tiny",
        input: "Hello",
        stream: true,
      }),
    );
    const text = await response.text();

    expect(text).toContain("event: response.incomplete");
    expect(text).toContain('"status":"incomplete"');
    expect(text).toContain('"incomplete_details":{"reason":"max_output_tokens"}');
  });

  test("closes OpenAI response streams when stop sequences end generation early", async () => {
    let streamClosed = false;
    const engine: GenerationEngine = {
      generate() {
        throw new Error("generate should not be used");
      },
      async *stream() {
        try {
          yield { type: "text", text: "Hello stop there" };
          await new Promise((resolve) => setTimeout(resolve, 10));
          yield {
            type: "done",
            finishReason: "stop",
            usage: { promptTokens: 2, completionTokens: 4, totalTokens: 6 },
          };
        } finally {
          streamClosed = true;
        }
      },
    };
    const fetch = createFetchHandler({
      engine,
      idGenerator: () => "resp-stop",
      now: () => new Date(123_000),
    });

    const response = await fetch(
      request("/v1/responses", {
        model: "tiny",
        input: "Hello",
        stream: true,
        stop: ["stop"],
      }),
    );
    const text = await response.text();

    expect(text).toContain("event: response.completed");
    expect(text).toContain('"output_text":"Hello "');
    expect(text).not.toContain("there");
    expect(streamClosed).toBe(true);
  });

  test("flushes Responses stream chunks before a microtask-heavy generator drains", async () => {
    let drained = false;
    const engine: GenerationEngine = {
      generate() {
        return { text: "", finishReason: "stop" };
      },
      async *stream() {
        for (let index = 0; index < 50; index += 1) {
          await Promise.resolve();
          yield { type: "text", text: `${index} ` };
        }
        drained = true;
        yield {
          type: "done",
          finishReason: "stop",
          usage: { promptTokens: 1, completionTokens: 50, totalTokens: 51 },
        };
      },
    };
    const server = Bun.serve({
      port: 0,
      fetch: createFetchHandler({ engine }),
    });

    try {
      const response = await fetch(`${endpointFor(server)}/v1/responses`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: "tiny",
          input: "Hello",
          stream: true,
        }),
      });
      const reader = response.body?.getReader();
      expect(reader).toBeDefined();
      if (reader === undefined) {
        throw new Error("expected a response body reader");
      }

      const firstChunk = await Promise.race([
        reader.read(),
        new Promise<never>((_, reject) =>
          setTimeout(() => reject(new Error("timed out waiting for streamed bytes")), 1000),
        ),
      ]);
      expect(firstChunk.done).toBe(false);
      expect(drained).toBe(false);
      await reader.cancel();
    } finally {
      server.stop(true);
    }
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
    expect(text).toContain('"content":"Hello"');
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
        input: [{ role: "user", content: [{ type: "input_image", image_url: "data:" }] }],
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

  test("emits generation errors when response streaming startup fails", async () => {
    const events: ServeEvent[] = [];
    const fetch = createFetchHandler({
      engine: {
        generate() {
          throw new Error("generate should not be called");
        },
        stream(normalized) {
          throw new ServeError(`Model "${normalized.model}" could not stream.`, {
            code: "stream_failed",
            status: 503,
          });
        },
      },
      idGenerator: () => "resp-stream-error",
      onEvent: (event) => events.push(event),
    });

    const response = await fetch(
      request("/v1/responses", {
        model: "broken-local",
        input: "Hello",
        stream: true,
      }),
    );
    const body = await response.json();

    expect(response.status).toBe(503);
    expect(body.error.code).toBe("stream_failed");
    expect(events.filter((event) => event.type.startsWith("generation_"))).toMatchObject([
      {
        type: "generation_start",
        id: "resp-stream-error",
        model: "broken-local",
        protocol: "openai.responses",
      },
      {
        type: "generation_error",
        id: "resp-stream-error",
        model: "broken-local",
        protocol: "openai.responses",
        code: "stream_failed",
      },
    ]);
  });

  test("returns client_cancelled when a non-streaming request aborts during generation", async () => {
    const events: ServeEvent[] = [];
    const controller = new AbortController();
    let enteredGeneration!: () => void;
    const entered = new Promise<void>((resolve) => {
      enteredGeneration = resolve;
    });
    const fetch = createFetchHandler({
      engine: {
        generate(normalized) {
          enteredGeneration();
          return new Promise<never>((_resolve, reject) => {
            if (normalized.abortSignal?.aborted) {
              const error = new Error("aborted");
              error.name = "AbortError";
              reject(error);
              return;
            }
            normalized.abortSignal?.addEventListener(
              "abort",
              () => {
                const error = new Error("aborted");
                error.name = "AbortError";
                reject(error);
              },
              { once: true },
            );
          });
        },
      },
      idGenerator: () => "cmpl-cancel",
      onEvent: (event) => events.push(event),
    });

    const responsePromise = fetch(
      new Request("http://localhost/v1/completions", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model: "tiny", prompt: "Hello" }),
        signal: controller.signal,
      }),
    );
    await entered;
    controller.abort();
    const response = await responsePromise;
    const body = await response.json();

    expect(response.status).toBe(499);
    expect(body.error.code).toBe("client_cancelled");
    expect(events.find((event) => event.type === "generation_error")).toMatchObject({
      code: "client_cancelled",
    });
    expect(events.find((event) => event.type === "request_error")).toMatchObject({
      code: "client_cancelled",
      status: 499,
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

  test("flushes completion stream chunks before a microtask-heavy generator drains", async () => {
    let drained = false;
    const engine: GenerationEngine = {
      generate() {
        return { text: "", finishReason: "stop" };
      },
      async *stream() {
        for (let index = 0; index < 50; index += 1) {
          await Promise.resolve();
          yield { type: "text", text: `${index} ` };
        }
        drained = true;
        yield {
          type: "done",
          finishReason: "stop",
          usage: { promptTokens: 1, completionTokens: 50, totalTokens: 51 },
        };
      },
    };
    const server = Bun.serve({
      port: 0,
      fetch: createFetchHandler({ engine }),
    });

    try {
      const response = await fetch(`${endpointFor(server)}/v1/completions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: "tiny",
          prompt: "Hello",
          stream: true,
        }),
      });
      const reader = response.body?.getReader();
      expect(reader).toBeDefined();
      if (reader === undefined) {
        throw new Error("expected a response body reader");
      }

      const firstChunk = await Promise.race([
        reader.read(),
        new Promise<never>((_, reject) =>
          setTimeout(() => reject(new Error("timed out waiting for streamed bytes")), 1000),
        ),
      ]);
      expect(firstChunk.done).toBe(false);
      expect(drained).toBe(false);
      await reader.cancel();
    } finally {
      server.stop(true);
    }
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
    let seenSignal: AbortSignal | undefined;
    const engine: GenerationEngine = {
      generate() {
        throw new Error("generate should not be used");
      },
      async *stream(normalized) {
        seenSignal = normalized.abortSignal;
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
    expect(seenSignal?.aborted).toBe(true);
    expect(events.find((event) => event.type === "generation_complete")).toMatchObject({
      finishReason: "cancelled",
    });
    expect(events.find((event) => event.type === "request_error")).toMatchObject({
      code: "client_cancelled",
      status: 499,
    });
    expect(events.find((event) => event.type === "request_complete")).toBeUndefined();
  });

  test("cancels streaming responses when the client disconnects", async () => {
    const events: ServeEvent[] = [];
    let streamClosed = false;
    let seenSignal: AbortSignal | undefined;
    const engine: GenerationEngine = {
      generate() {
        throw new Error("generate should not be used");
      },
      async *stream(normalized) {
        seenSignal = normalized.abortSignal;
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
      idGenerator: () => "resp-cancel",
      now: () => new Date(123_000),
      onEvent: (event) => events.push(event),
    });

    const response = await fetch(
      request("/v1/responses", {
        model: "tiny",
        input: "Hello",
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
    expect(seenSignal?.aborted).toBe(true);
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
    const missingResponseStream = await fetch(
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
    expect(missingResponseStream.status).toBe(400);
    expect((await missingResponseStream.json()).error.code).toBe("stream_not_supported");
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

    const responseBody = await fetch(
      new Request("http://localhost/v1/responses", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: "{",
      }),
    );
    const parsedResponseBody = await responseBody.json();

    expect(responseBody.status).toBe(400);
    expect(parsedResponseBody.error.code).toBe("invalid_json");
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
