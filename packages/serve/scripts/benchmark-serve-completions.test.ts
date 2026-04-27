import { describe, expect, test } from "bun:test";

import { runCompletionRequest } from "./benchmark-serve-completions";
import { parseServeBenchmarkArgs, type ServeBenchmarkOptions } from "./benchmark-serve-options";

function endpointFor(server: ReturnType<typeof Bun.serve>): string {
  return `http://${server.hostname}:${server.port}`;
}

function options(args: readonly string[] = []): ServeBenchmarkOptions {
  return parseServeBenchmarkArgs(["model", "--model-id", "tiny", "--greedy", ...args]);
}

const prompt = { tokenIds: [1, 2], text: "Hello benchmark" };

function textSseFrame(payload: unknown): string {
  return `data: ${JSON.stringify(payload)}\n\n`;
}

function eventSseFrame(event: string, payload: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(payload)}\n\n`;
}

describe("serve benchmark completion requests", () => {
  test("measures buffered completion responses", async () => {
    const server = Bun.serve({
      port: 0,
      async fetch(request) {
        const body = (await request.json()) as Record<string, unknown>;
        expect(body.stream).toBeUndefined();
        expect(body.temperature).toBe(0);
        expect(body.ignore_eos).toBe(true);
        return Response.json({
          id: "cmpl-buffered",
          choices: [{ text: "ok", finish_reason: "length" }],
          usage: { prompt_tokens: 2, completion_tokens: 3, total_tokens: 5 },
        });
      },
    });

    try {
      const metrics = await runCompletionRequest(
        endpointFor(server),
        "tiny",
        prompt,
        { promptTokens: 2, generationTokens: 3, concurrency: 1 },
        options(["--ignore-eos"]),
      );

      expect(metrics).toMatchObject({
        id: "cmpl-buffered",
        ttftMs: null,
        promptToFirstTokenTps: null,
        postTtftCompletionTps: null,
        meanStreamChunkGapMs: null,
        maxStreamChunkGapMs: null,
        promptTokens: 2,
        completionTokens: 3,
        totalTokens: 5,
        finishReason: "length",
        streamChunks: 0,
        streamBytes: 0,
      });
      expect(metrics.durationMs).toBeGreaterThanOrEqual(0);
    } finally {
      server.stop(true);
    }
  });

  test("measures streaming completion responses with TTFT and usage chunks", async () => {
    const encoder = new TextEncoder();
    const frames = [
      textSseFrame({
        id: "cmpl-stream",
        choices: [{ text: "a", finish_reason: null }],
        usage: null,
      }),
      textSseFrame({
        id: "cmpl-stream",
        choices: [{ text: "b", finish_reason: null }],
        usage: null,
      }),
      textSseFrame({
        id: "cmpl-stream",
        choices: [{ text: "", finish_reason: "length" }],
        usage: null,
      }),
      textSseFrame({
        id: "cmpl-stream",
        choices: [],
        usage: { prompt_tokens: 2, completion_tokens: 3, total_tokens: 5 },
      }),
      "data: [DONE]\n\n",
    ].join("");

    const server = Bun.serve({
      port: 0,
      async fetch(request) {
        const body = (await request.json()) as Record<string, unknown>;
        expect(body.stream).toBe(true);
        expect(body.stream_options).toEqual({ include_usage: true });

        return new Response(
          new ReadableStream<Uint8Array>({
            start(controller) {
              controller.enqueue(encoder.encode(frames.slice(0, 17)));
              controller.enqueue(encoder.encode(frames.slice(17)));
              controller.close();
            },
          }),
          { headers: { "content-type": "text/event-stream" } },
        );
      },
    });

    try {
      const metrics = await runCompletionRequest(
        endpointFor(server),
        "tiny",
        prompt,
        { promptTokens: 2, generationTokens: 3, concurrency: 1 },
        options(["--stream"]),
      );

      expect(metrics).toMatchObject({
        id: "cmpl-stream",
        promptTokens: 2,
        completionTokens: 3,
        totalTokens: 5,
        finishReason: "length",
        streamChunks: 2,
      });
      expect(metrics.ttftMs).not.toBeNull();
      expect(metrics.promptToFirstTokenTps).not.toBeNull();
      expect(metrics.postTtftCompletionTps).not.toBeNull();
      expect(metrics.meanStreamChunkGapMs).not.toBeNull();
      expect(metrics.maxStreamChunkGapMs).not.toBeNull();
      expect(metrics.streamBytes).toBeGreaterThan(0);
    } finally {
      server.stop(true);
    }
  });

  test("measures chat completion responses", async () => {
    const server = Bun.serve({
      port: 0,
      async fetch(request) {
        expect(new URL(request.url).pathname).toBe("/v1/chat/completions");
        const body = (await request.json()) as Record<string, unknown>;
        expect(body.messages).toEqual([{ role: "user", content: "Hello benchmark" }]);
        return Response.json({
          choices: [{ message: { content: "ok" }, finish_reason: "length" }],
          usage: { prompt_tokens: 4, completion_tokens: 3, total_tokens: 7 },
        });
      },
    });

    try {
      const metrics = await runCompletionRequest(
        endpointFor(server),
        "tiny",
        prompt,
        { promptTokens: 2, generationTokens: 3, concurrency: 1 },
        options(["--protocol", "chat", "--ignore-eos"]),
      );

      expect(metrics).toMatchObject({
        promptTokens: 4,
        completionTokens: 3,
        totalTokens: 7,
        finishReason: "length",
      });
    } finally {
      server.stop(true);
    }
  });

  test("measures buffered Anthropic messages responses", async () => {
    const server = Bun.serve({
      port: 0,
      async fetch(request) {
        expect(new URL(request.url).pathname).toBe("/v1/messages");
        const body = (await request.json()) as Record<string, unknown>;
        expect(body.messages).toEqual([{ role: "user", content: "Hello benchmark" }]);
        expect(body.max_tokens).toBe(3);
        expect(body.temperature).toBe(0);
        return Response.json({
          id: "msg-buffered",
          type: "message",
          content: [{ type: "text", text: "ok" }],
          stop_reason: "max_tokens",
          usage: { input_tokens: 4, output_tokens: 3 },
        });
      },
    });

    try {
      const metrics = await runCompletionRequest(
        endpointFor(server),
        "tiny",
        prompt,
        { promptTokens: 2, generationTokens: 3, concurrency: 1 },
        options(["--protocol", "anthropic"]),
      );

      expect(metrics).toMatchObject({
        id: "msg-buffered",
        promptTokens: 4,
        completionTokens: 3,
        totalTokens: 7,
        finishReason: "max_tokens",
      });
    } finally {
      server.stop(true);
    }
  });

  test("measures streaming responses API events", async () => {
    const encoder = new TextEncoder();
    const frames = [
      eventSseFrame("response.output_text.delta", { delta: "a" }),
      eventSseFrame("response.reasoning_text.delta", { delta: "b" }),
      eventSseFrame("response.incomplete", {
        response: {
          status: "incomplete",
          incomplete_details: { reason: "max_output_tokens" },
          usage: { input_tokens: 5, output_tokens: 3, total_tokens: 8 },
        },
      }),
      "data: [DONE]\n\n",
    ].join("");

    const server = Bun.serve({
      port: 0,
      async fetch(request) {
        expect(new URL(request.url).pathname).toBe("/v1/responses");
        const body = (await request.json()) as Record<string, unknown>;
        expect(body.input).toEqual([{ role: "user", content: "Hello benchmark" }]);
        expect(body.stream_options).toEqual({ include_obfuscation: false });
        return new Response(
          new ReadableStream<Uint8Array>({
            start(controller) {
              controller.enqueue(encoder.encode(frames));
              controller.close();
            },
          }),
          { headers: { "content-type": "text/event-stream" } },
        );
      },
    });

    try {
      const metrics = await runCompletionRequest(
        endpointFor(server),
        "tiny",
        prompt,
        { promptTokens: 2, generationTokens: 3, concurrency: 1 },
        options(["--protocol", "responses", "--stream"]),
      );

      expect(metrics).toMatchObject({
        promptTokens: 5,
        completionTokens: 3,
        totalTokens: 8,
        finishReason: "length",
        streamChunks: 2,
      });
      expect(metrics.ttftMs).not.toBeNull();
    } finally {
      server.stop(true);
    }
  });

  test("measures streaming Anthropic Messages events", async () => {
    const encoder = new TextEncoder();
    const frames = [
      eventSseFrame("message_start", {
        type: "message_start",
        message: {
          id: "msg-stream",
          type: "message",
          usage: { input_tokens: 5, output_tokens: 0 },
        },
      }),
      eventSseFrame("content_block_delta", {
        type: "content_block_delta",
        index: 0,
        delta: { type: "thinking_delta", thinking: "a" },
      }),
      eventSseFrame("content_block_delta", {
        type: "content_block_delta",
        index: 1,
        delta: { type: "text_delta", text: "b" },
      }),
      eventSseFrame("message_delta", {
        type: "message_delta",
        delta: { stop_reason: "max_tokens", stop_sequence: null },
        usage: { output_tokens: 3 },
      }),
      eventSseFrame("message_stop", { type: "message_stop" }),
    ].join("");

    const server = Bun.serve({
      port: 0,
      async fetch(request) {
        expect(new URL(request.url).pathname).toBe("/v1/messages");
        const body = (await request.json()) as Record<string, unknown>;
        expect(body.messages).toEqual([{ role: "user", content: "Hello benchmark" }]);
        expect(body.stream).toBe(true);
        return new Response(
          new ReadableStream<Uint8Array>({
            start(controller) {
              controller.enqueue(encoder.encode(frames));
              controller.close();
            },
          }),
          { headers: { "content-type": "text/event-stream" } },
        );
      },
    });

    try {
      const metrics = await runCompletionRequest(
        endpointFor(server),
        "tiny",
        prompt,
        { promptTokens: 2, generationTokens: 3, concurrency: 1 },
        options(["--protocol", "anthropic", "--stream"]),
      );

      expect(metrics).toMatchObject({
        id: "msg-stream",
        promptTokens: 5,
        completionTokens: 3,
        totalTokens: 8,
        finishReason: "max_tokens",
        streamChunks: 2,
      });
      expect(metrics.ttftMs).not.toBeNull();
    } finally {
      server.stop(true);
    }
  });

  test("rejects streaming responses that end without usage", async () => {
    const server = Bun.serve({
      port: 0,
      fetch() {
        return new Response("data: [DONE]\n\n", {
          headers: { "content-type": "text/event-stream" },
        });
      },
    });

    try {
      await expect(
        runCompletionRequest(
          endpointFor(server),
          "tiny",
          prompt,
          { promptTokens: 2, generationTokens: 3, concurrency: 1 },
          options(["--stream"]),
        ),
      ).rejects.toThrow("streaming request ended without usage");
    } finally {
      server.stop(true);
    }
  });

  test("rejects streaming terminal chunks without usage", async () => {
    const server = Bun.serve({
      port: 0,
      fetch() {
        return new Response(
          textSseFrame({ choices: [{ text: "", finish_reason: "length" }], usage: null }) +
            "data: [DONE]\n\n",
          { headers: { "content-type": "text/event-stream" } },
        );
      },
    });

    try {
      await expect(
        runCompletionRequest(
          endpointFor(server),
          "tiny",
          prompt,
          { promptTokens: 2, generationTokens: 3, concurrency: 1 },
          options(["--stream"]),
        ),
      ).rejects.toThrow("streaming request ended without usage");
    } finally {
      server.stop(true);
    }
  });

  test("rejects streaming text chunks without usage", async () => {
    const server = Bun.serve({
      port: 0,
      fetch() {
        return new Response(
          textSseFrame({ choices: [{ text: "a", finish_reason: null }], usage: null }) +
            textSseFrame({ choices: [{ text: "", finish_reason: "length" }], usage: null }) +
            "data: [DONE]\n\n",
          { headers: { "content-type": "text/event-stream" } },
        );
      },
    });

    try {
      await expect(
        runCompletionRequest(
          endpointFor(server),
          "tiny",
          prompt,
          { promptTokens: 2, generationTokens: 3, concurrency: 1 },
          options(["--stream"]),
        ),
      ).rejects.toThrow("streaming request ended without usage");
    } finally {
      server.stop(true);
    }
  });
});
