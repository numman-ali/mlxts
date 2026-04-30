import { describe, expect, test } from "bun:test";

import { normalizeOpenAIResponseRequest } from "../protocols/openai-responses";
import type { GenerationStreamEvent } from "../types";
import { writeOpenAIResponseStreamEvents } from "./writer-openai-responses";

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
    const data = dataLine.slice("data: ".length);
    if (data === "[DONE]") {
      continue;
    }
    const parsed = JSON.parse(data) as unknown;
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

describe("OpenAI Responses stream writer", () => {
  test("streams generated function-call events when tools are active", async () => {
    const response = normalizeOpenAIResponseRequest(
      {
        model: "tiny",
        input: "Read README.md",
        stream: true,
        tools: [{ type: "function", name: "read_file" }],
      },
      { id: "resp-tool-stream" },
    );

    const { text, summary } = await collectSse((controller) =>
      writeOpenAIResponseStreamEvents(
        controller,
        streamEvents(
          { type: "text", text: "<think>Need a file read.</think><tool" },
          {
            type: "text",
            text: '_call>{"id":"call-read","name":"read_file","arguments":{"path":"README.md"}}</tool_call>',
          },
          {
            type: "done",
            finishReason: "stop",
            usage: { promptTokens: 4, completionTokens: 6, totalTokens: 10 },
          },
        ),
        response,
        { id: "resp-tool-stream", created: 123 },
      ),
    );
    const events = parseSseEvents(text);
    const outputAdded = events.filter((event) => event.event === "response.output_item.added");
    const argumentDelta = events.find(
      (event) => event.event === "response.function_call_arguments.delta",
    );
    const argumentDone = events.find(
      (event) => event.event === "response.function_call_arguments.done",
    );
    const terminal = events.find((event) => event.event === "response.completed");

    expect(summary).toEqual({
      finishReason: "stop",
      usage: { promptTokens: 4, completionTokens: 6, totalTokens: 10 },
    });
    expect(outputAdded.map((event) => event.data.item)).toMatchObject([
      { id: "resp-tool-stream-rsn", type: "reasoning", status: "in_progress" },
      {
        id: "resp-tool-stream-fc-1",
        type: "function_call",
        status: "in_progress",
        call_id: "call-read",
        name: "read_file",
        arguments: "",
      },
    ]);
    expect(outputAdded[1]?.data.response_id).toBe("resp-tool-stream");
    expect(argumentDelta?.data).toMatchObject({
      response_id: "resp-tool-stream",
      item_id: "resp-tool-stream-fc-1",
      output_index: 1,
      delta: '{"path":"README.md"}',
    });
    expect(argumentDone?.data).toMatchObject({
      response_id: "resp-tool-stream",
      item_id: "resp-tool-stream-fc-1",
      output_index: 1,
      call_id: "call-read",
      name: "read_file",
      arguments: '{"path":"README.md"}',
    });
    expect(terminal?.data.response).toMatchObject({
      output_text: "",
      output: [
        {
          id: "resp-tool-stream-rsn",
          type: "reasoning",
          status: "completed",
          content: [{ type: "reasoning_text", text: "Need a file read." }],
        },
        {
          id: "resp-tool-stream-fc-1",
          type: "function_call",
          status: "completed",
          call_id: "call-read",
          name: "read_file",
          arguments: '{"path":"README.md"}',
        },
      ],
    });
    expect(text).not.toContain("<tool_call>");
    expect(text).toContain("data: [DONE]");
  });

  test("keeps tool-looking text visible when tools are inactive", async () => {
    const response = normalizeOpenAIResponseRequest(
      {
        model: "tiny",
        input: "Echo this.",
        stream: true,
      },
      { id: "resp-tool-text" },
    );

    const { text } = await collectSse((controller) =>
      writeOpenAIResponseStreamEvents(
        controller,
        streamEvents(
          { type: "text", text: "<tool_call>{bad" },
          { type: "done", finishReason: "stop" },
        ),
        response,
        { id: "resp-tool-text", created: 123 },
      ),
    );
    const terminal = parseSseEvents(text).find((event) => event.event === "response.completed");

    expect(text).toContain("<tool_call>{bad");
    expect(text).not.toContain("response.function_call_arguments.delta");
    expect(terminal?.data.response).toMatchObject({ output_text: "<tool_call>{bad" });
  });
});
