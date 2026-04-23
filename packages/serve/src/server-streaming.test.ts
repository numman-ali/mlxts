import { describe, expect, test } from "bun:test";

import { normalizeOpenAIChatCompletionRequest } from "./protocols/openai-chat-completions";
import { normalizeOpenAICompletionRequest } from "./protocols/openai-completions";
import { sseHeaders, writeChatStreamEvents, writeStreamEvents } from "./server-streaming";
import type { GenerationStreamEvent } from "./types";

function parseSsePayloads(text: string): unknown[] {
  return text
    .trim()
    .split("\n\n")
    .filter((chunk) => chunk.startsWith("data: "))
    .map((chunk) => chunk.slice("data: ".length))
    .filter((payload) => payload !== "[DONE]")
    .map((payload) => JSON.parse(payload) as unknown);
}

async function collectSse<T>(
  write: (controller: ReadableStreamDefaultController<Uint8Array>) => Promise<T>,
): Promise<{ text: string; summary: T }> {
  let summary: T | undefined;
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      return write(controller).then(
        (result) => {
          summary = result;
          controller.close();
        },
        (error: unknown) => {
          controller.error(error);
        },
      );
    },
  });

  return { text: await new Response(stream).text(), summary: summary as T };
}

async function* streamEvents(
  ...events: readonly GenerationStreamEvent[]
): AsyncIterable<GenerationStreamEvent> {
  for (const event of events) {
    yield event;
  }
}

describe("server streaming helpers", () => {
  test("returns SSE headers for Bun responses", () => {
    expect(sseHeaders()).toEqual({
      "content-type": "text/event-stream; charset=utf-8",
      "cache-control": "no-cache",
      connection: "keep-alive",
    });
  });

  test("completion SSE stops on stop sequences even when the stream ends without a done event", async () => {
    const batch = normalizeOpenAICompletionRequest(
      {
        model: "tiny",
        prompt: "Hello",
        stream: true,
        stop: "stop",
      },
      { id: "cmpl-stop" },
    );
    const request = batch.requests[0];

    expect(request).toBeDefined();
    if (request === undefined) {
      throw new Error("expected a normalized completion request");
    }

    const { text, summary } = await collectSse((controller) =>
      writeStreamEvents(
        controller,
        streamEvents({ type: "text", text: "He sto" }, { type: "text", text: "p there" }),
        batch,
        request,
        { id: "cmpl-stop", created: 123 },
      ),
    );
    const payloads = parseSsePayloads(text) as Array<{
      choices: Array<{ text: string; finish_reason: string | null }>;
    }>;

    expect(summary).toEqual({ finishReason: "stop" });
    expect(payloads.map((payload) => payload.choices[0]?.text).filter(Boolean)).toEqual(["He "]);
    expect(payloads.at(-1)?.choices[0]?.finish_reason).toBe("stop");
    expect(text).toContain("data: [DONE]");
    expect(text).not.toContain('"usage"');
    expect(text).not.toContain("stop there");
  });

  test("chat SSE preserves reasoning, stops visible content, and emits null usage when the stream ends early", async () => {
    const chat = normalizeOpenAIChatCompletionRequest(
      {
        model: "tiny",
        messages: [{ role: "user", content: "Hi" }],
        stream: true,
        stream_options: { include_usage: true },
        stop: "done",
      },
      { id: "chat-stop" },
    );

    const { text, summary } = await collectSse((controller) =>
      writeChatStreamEvents(
        controller,
        streamEvents(
          { type: "text", text: "<think>plan " },
          { type: "text", text: "more</think>Hello do" },
          { type: "text", text: "ne extra" },
        ),
        chat,
        { id: "chat-stop", created: 123 },
      ),
    );
    const payloads = parseSsePayloads(text) as Array<{
      choices: Array<{
        delta?: { role?: string; content?: string; reasoning_content?: string };
        finish_reason?: string | null;
      }>;
      usage?: unknown;
    }>;
    const visibleContent = payloads
      .flatMap((payload) => payload.choices)
      .map((choice) => choice.delta?.content)
      .filter((value): value is string => value !== undefined)
      .join("");
    const reasoning = payloads
      .flatMap((payload) => payload.choices)
      .map((choice) => choice.delta?.reasoning_content)
      .filter((value): value is string => value !== undefined)
      .join("");

    expect(summary).toEqual({ finishReason: "stop" });
    expect(payloads[0]?.choices[0]?.delta?.role).toBe("assistant");
    expect(visibleContent).toBe("Hello ");
    expect(reasoning).toBe("plan more");
    expect(payloads.at(-1)).toMatchObject({ choices: [], usage: null });
    expect(payloads.at(-2)?.choices[0]?.finish_reason).toBe("stop");
    expect(text).toContain("data: [DONE]");
    expect(text).not.toContain("done extra");
  });
});
