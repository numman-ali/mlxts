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

async function* closableStreamEvents(
  onClose: () => void,
  ...events: readonly GenerationStreamEvent[]
): AsyncIterable<GenerationStreamEvent> {
  try {
    yield* streamEvents(...events);
  } finally {
    onClose();
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
    let closed = false;

    expect(request).toBeDefined();
    if (request === undefined) {
      throw new Error("expected a normalized completion request");
    }

    const { text, summary } = await collectSse((controller) =>
      writeStreamEvents(
        controller,
        closableStreamEvents(
          () => {
            closed = true;
          },
          { type: "text", text: "He sto" },
          { type: "text", text: "p there" },
        ),
        batch,
        request,
        { id: "cmpl-stop", created: 123 },
      ),
    );
    const payloads = parseSsePayloads(text) as Array<{
      choices: Array<{ text: string; finish_reason: string | null }>;
    }>;

    expect(summary).toEqual({ finishReason: "stop" });
    expect(text.startsWith(": mlxts-serve stream started\n\n")).toBe(true);
    expect(payloads.map((payload) => payload.choices[0]?.text).filter(Boolean)).toEqual(["He "]);
    expect(payloads.at(-1)?.choices[0]?.finish_reason).toBe("stop");
    expect(text).toContain("data: [DONE]");
    expect(text).not.toContain('"usage"');
    expect(text).not.toContain("stop there");
    expect(closed).toBe(true);
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
    let closed = false;

    const { text, summary } = await collectSse((controller) =>
      writeChatStreamEvents(
        controller,
        closableStreamEvents(
          () => {
            closed = true;
          },
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
    expect(
      payloads
        .filter((payload) => payload.choices.length > 0)
        .every((payload) => payload.usage === null),
    ).toBe(true);
    expect(payloads.at(-1)).toMatchObject({ choices: [], usage: null });
    expect(payloads.at(-2)?.choices[0]?.finish_reason).toBe("stop");
    expect(text).toContain("data: [DONE]");
    expect(text).not.toContain("done extra");
    expect(closed).toBe(true);
  });

  test("chat SSE preserves stop finish when the stop sequence is found during final flush", async () => {
    const chat = normalizeOpenAIChatCompletionRequest(
      {
        model: "tiny",
        messages: [{ role: "user", content: "Hi" }],
        stream: true,
        stop: "stop",
      },
      { id: "chat-final-stop" },
    );

    const { text, summary } = await collectSse((controller) =>
      writeChatStreamEvents(
        controller,
        streamEvents(
          { type: "text", text: "<think>plan</think>Hello stop" },
          { type: "done", finishReason: "length" },
        ),
        chat,
        { id: "chat-final-stop", created: 123 },
      ),
    );
    const payloads = parseSsePayloads(text) as Array<{
      choices: Array<{ delta?: { content?: string }; finish_reason?: string | null }>;
    }>;
    const visibleContent = payloads
      .flatMap((payload) => payload.choices)
      .map((choice) => choice.delta?.content)
      .filter((value): value is string => value !== undefined)
      .join("");

    expect(summary).toEqual({ finishReason: "stop" });
    expect(visibleContent).toBe("Hello ");
    expect(payloads.at(-1)?.choices[0]?.finish_reason).toBe("stop");
    expect(text).not.toContain("Hello stop");
  });

  test("chat SSE emits structured Qwen tool-call deltas without XML leakage", async () => {
    const chat = normalizeOpenAIChatCompletionRequest(
      {
        model: "tiny",
        messages: [{ role: "user", content: "List files" }],
        tools: [{ type: "function", function: { name: "list_files" } }],
        stream: true,
      },
      { id: "chat-tools" },
    );

    const { text } = await collectSse((controller) =>
      writeChatStreamEvents(
        controller,
        streamEvents(
          { type: "text", text: "<tool" },
          {
            type: "text",
            text: "_call><function=list_files><parameter=path>.</parameter></function></tool_call>",
          },
          { type: "done", finishReason: "stop" },
        ),
        chat,
        { id: "chat-tools", created: 123 },
      ),
    );
    const payloads = parseSsePayloads(text) as Array<{
      choices: Array<{
        delta?: {
          role?: string;
          content?: string;
          tool_calls?: Array<{
            index: number;
            id?: string;
            type?: string;
            function?: { name?: string; arguments?: string };
          }>;
        };
        finish_reason?: string | null;
      }>;
    }>;
    const deltas = payloads.flatMap((payload) => payload.choices).map((choice) => choice.delta);
    const toolCalls = deltas.flatMap((delta) => delta?.tool_calls ?? []);

    expect(deltas[0]?.role).toBe("assistant");
    expect(toolCalls).toEqual([
      {
        index: 0,
        id: "call_1",
        type: "function",
        function: { name: "list_files", arguments: '{"path":"."}' },
      },
    ]);
    expect(payloads.at(-1)?.choices[0]?.finish_reason).toBe("tool_calls");
    expect(text).not.toContain("<tool_call>");
    expect(text).toContain("data: [DONE]");
  });

  test("chat SSE strips Gemma tool-response sentinel after structured tool calls", async () => {
    const chat = normalizeOpenAIChatCompletionRequest(
      {
        model: "tiny",
        messages: [{ role: "user", content: "List files" }],
        tools: [{ type: "function", function: { name: "list_files" } }],
        stream: true,
      },
      { id: "chat-gemma-tools" },
    );

    const { text } = await collectSse((controller) =>
      writeChatStreamEvents(
        controller,
        streamEvents(
          {
            type: "text",
            text: '<|tool_call>call:list_files{path:<|"|>.<|"|>}<tool_call|>',
          },
          { type: "text", text: "<|tool_response>" },
          { type: "done", finishReason: "eos" },
        ),
        chat,
        { id: "chat-gemma-tools", created: 123 },
      ),
    );
    const payloads = parseSsePayloads(text) as Array<{
      choices: Array<{
        delta?: {
          content?: string;
          tool_calls?: Array<{
            index: number;
            id?: string;
            type?: string;
            function?: { name?: string; arguments?: string };
          }>;
        };
        finish_reason?: string | null;
      }>;
    }>;
    const content = payloads
      .flatMap((payload) => payload.choices)
      .map((choice) => choice.delta?.content)
      .filter((value): value is string => value !== undefined)
      .join("");
    const toolCalls = payloads.flatMap((payload) =>
      payload.choices.flatMap((choice) => choice.delta?.tool_calls ?? []),
    );

    expect(content).toBe("");
    expect(toolCalls).toEqual([
      {
        index: 0,
        id: "call_1",
        type: "function",
        function: { name: "list_files", arguments: '{"path":"."}' },
      },
    ]);
    expect(payloads.at(-1)?.choices[0]?.finish_reason).toBe("tool_calls");
    expect(text).not.toContain("<|tool_response>");
  });

  test("chat SSE keeps malformed tool-call envelopes visible", async () => {
    const chat = normalizeOpenAIChatCompletionRequest(
      {
        model: "tiny",
        messages: [{ role: "user", content: "Read file" }],
        tools: [{ type: "function", function: { name: "read_file" } }],
        stream: true,
      },
      { id: "chat-bad-tool" },
    );

    const { text } = await collectSse((controller) =>
      writeChatStreamEvents(
        controller,
        streamEvents(
          { type: "text", text: "<tool_call>{bad" },
          { type: "done", finishReason: "stop" },
        ),
        chat,
        { id: "chat-bad-tool", created: 123 },
      ),
    );

    expect(text).toContain("<tool_call>{bad");
    expect(text).not.toContain("tool_calls");
    expect(text).toContain('"finish_reason":"stop"');
  });

  test("chat SSE separates reasoning and multiple generated tool calls", async () => {
    const chat = normalizeOpenAIChatCompletionRequest(
      {
        model: "tiny",
        messages: [{ role: "user", content: "Read and list" }],
        tools: [
          { type: "function", function: { name: "read_file" } },
          { type: "function", function: { name: "list_files" } },
        ],
        stream: true,
      },
      { id: "chat-many-tools" },
    );

    const { text } = await collectSse((controller) =>
      writeChatStreamEvents(
        controller,
        streamEvents(
          { type: "text", text: "<think>Need tools.</think>\n" },
          {
            type: "text",
            text: '<tool_call>{"id":"call-read","name":"read_file","arguments":{"path":"README.md"}}</tool_call>',
          },
          {
            type: "text",
            text: '<tool_call>{"name":"list_files","arguments":{"path":"."}}</tool_call>',
          },
          { type: "done", finishReason: "eos" },
        ),
        chat,
        { id: "chat-many-tools", created: 123 },
      ),
    );
    const payloads = parseSsePayloads(text) as Array<{
      choices: Array<{
        delta?: {
          reasoning_content?: string;
          content?: string;
          tool_calls?: Array<{ index: number; id?: string; function?: { name?: string } }>;
        };
        finish_reason?: string | null;
      }>;
    }>;
    const reasoning = payloads
      .flatMap((payload) => payload.choices)
      .map((choice) => choice.delta?.reasoning_content)
      .filter((value): value is string => value !== undefined)
      .join("");
    const toolCalls = payloads.flatMap((payload) =>
      payload.choices.flatMap((choice) => choice.delta?.tool_calls ?? []),
    );

    expect(reasoning).toBe("Need tools.");
    expect(toolCalls.map((call) => [call.index, call.id, call.function?.name])).toEqual([
      [0, "call-read", "read_file"],
      [1, "call_2", "list_files"],
    ]);
    expect(payloads.at(-1)?.choices[0]?.finish_reason).toBe("tool_calls");
    expect(text).not.toContain("<tool_call>");
  });
});
