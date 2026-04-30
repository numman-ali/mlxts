import { describe, expect, test } from "bun:test";

import { normalizeAnthropicMessageRequest } from "../protocols/anthropic-messages";
import type { GenerationStreamEvent } from "../types";
import { writeAnthropicMessageStreamEvents } from "./writer-anthropic-messages";

type ParsedSseEvent = {
  event: string;
  data: Record<string, unknown>;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function parseSseEvents(text: string): ParsedSseEvent[] {
  const events: ParsedSseEvent[] = [];
  for (const block of text.trim().split("\n\n")) {
    const lines = block.split("\n");
    const eventLine = lines.find((line) => line.startsWith("event: "));
    const dataLine = lines.find((line) => line.startsWith("data: "));
    if (eventLine === undefined || dataLine === undefined) {
      continue;
    }
    const parsed = JSON.parse(dataLine.slice("data: ".length)) as unknown;
    if (isRecord(parsed)) {
      events.push({ event: eventLine.slice("event: ".length), data: parsed });
    }
  }
  return events;
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

describe("Anthropic Messages stream writer", () => {
  test("streams generated tool_use blocks when tools are active", async () => {
    const message = normalizeAnthropicMessageRequest(
      {
        model: "tiny",
        stream: true,
        tools: [{ name: "read_file", input_schema: { type: "object" } }],
        messages: [{ role: "user", content: "Read README.md" }],
        max_tokens: 8,
      },
      { id: "msg-tool-stream" },
    );

    const { text, summary } = await collectSse((controller) =>
      writeAnthropicMessageStreamEvents(
        controller,
        streamEvents(
          { type: "text", text: "<think>Need a file read.</think>I will check.<tool" },
          {
            type: "text",
            text: '_call>{"id":"toolu_read","name":"read_file","arguments":{"path":"README.md"}}</tool_call>',
          },
          {
            type: "done",
            finishReason: "stop",
            usage: { promptTokens: 4, completionTokens: 6, totalTokens: 10 },
          },
        ),
        message,
        { id: "msg-tool-stream", created: 123 },
      ),
    );
    const events = parseSseEvents(text);
    const starts = events.filter((event) => event.event === "content_block_start");
    const inputDelta = events.find(
      (event) =>
        event.event === "content_block_delta" &&
        isRecord(event.data.delta) &&
        event.data.delta.type === "input_json_delta",
    );
    const messageDelta = events.find((event) => event.event === "message_delta");

    expect(summary).toEqual({
      finishReason: "stop",
      usage: { promptTokens: 4, completionTokens: 6, totalTokens: 10 },
    });
    expect(starts.map((event) => event.data.content_block)).toMatchObject([
      { type: "thinking", thinking: "", signature: "" },
      { type: "text", text: "" },
      { type: "tool_use", id: "toolu_read", name: "read_file", input: {} },
    ]);
    expect(inputDelta?.data).toMatchObject({
      index: 2,
      delta: { type: "input_json_delta", partial_json: '{"path":"README.md"}' },
    });
    expect(messageDelta?.data).toMatchObject({
      delta: { stop_reason: "tool_use", stop_sequence: null },
      usage: { output_tokens: 6 },
    });
    expect(text).toContain('"delta":{"type":"thinking_delta","thinking":"Need a file read."}');
    expect(text).toContain('"delta":{"type":"text_delta","text":"I will check."}');
    expect(text).not.toContain("<tool_call>");
    expect(text).not.toContain("[DONE]");
  });

  test("keeps tool-looking text visible when tools are inactive", async () => {
    const message = normalizeAnthropicMessageRequest(
      {
        model: "tiny",
        stream: true,
        messages: [{ role: "user", content: "Echo this." }],
        max_tokens: 8,
      },
      { id: "msg-tool-text" },
    );

    const { text } = await collectSse((controller) =>
      writeAnthropicMessageStreamEvents(
        controller,
        streamEvents(
          { type: "text", text: "<tool_call>{bad" },
          { type: "done", finishReason: "stop" },
        ),
        message,
        { id: "msg-tool-text", created: 123 },
      ),
    );

    expect(text).toContain("<tool_call>{bad");
    expect(text).not.toContain("input_json_delta");
    expect(text).toContain('"stop_reason":"end_turn"');
  });
});
