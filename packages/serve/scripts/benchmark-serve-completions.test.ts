import { describe, expect, test } from "bun:test";

import { runCompletionRequest } from "./benchmark-serve-completions";
import { parseServeBenchmarkArgs, type ServeBenchmarkOptions } from "./benchmark-serve-options";

function endpointFor(server: ReturnType<typeof Bun.serve>): string {
  return `http://${server.hostname}:${server.port}`;
}

function options(args: readonly string[] = []): ServeBenchmarkOptions {
  return parseServeBenchmarkArgs(["model", "--model-id", "tiny", "--greedy", ...args]);
}

function textSseFrame(payload: unknown): string {
  return `data: ${JSON.stringify(payload)}\n\n`;
}

describe("serve benchmark completion requests", () => {
  test("measures buffered completion responses", async () => {
    const server = Bun.serve({
      port: 0,
      async fetch(request) {
        const body = (await request.json()) as Record<string, unknown>;
        expect(body.stream).toBeUndefined();
        expect(body.temperature).toBe(0);
        return Response.json({
          choices: [{ text: "ok", finish_reason: "length" }],
          usage: { prompt_tokens: 2, completion_tokens: 3, total_tokens: 5 },
        });
      },
    });

    try {
      const metrics = await runCompletionRequest(
        endpointFor(server),
        "tiny",
        [1, 2],
        { promptTokens: 2, generationTokens: 3, concurrency: 1 },
        options(),
      );

      expect(metrics).toMatchObject({
        ttftMs: null,
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
        choices: [{ text: "a", finish_reason: null }],
        usage: null,
      }),
      textSseFrame({
        choices: [{ text: "b", finish_reason: null }],
        usage: null,
      }),
      textSseFrame({
        choices: [{ text: "", finish_reason: "length" }],
        usage: null,
      }),
      textSseFrame({
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
        [1, 2],
        { promptTokens: 2, generationTokens: 3, concurrency: 1 },
        options(["--stream"]),
      );

      expect(metrics).toMatchObject({
        promptTokens: 2,
        completionTokens: 3,
        totalTokens: 5,
        finishReason: "length",
        streamChunks: 2,
      });
      expect(metrics.ttftMs).not.toBeNull();
      expect(metrics.streamBytes).toBeGreaterThan(0);
    } finally {
      server.stop(true);
    }
  });
});
