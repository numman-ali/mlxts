import { describe, expect, test } from "bun:test";

import type { GenerationStreamEvent } from "../types";
import { runSseGenerationStream, sseHeaders } from "./writer-base";

function createController(): ReadableStreamDefaultController<Uint8Array> {
  let controller: ReadableStreamDefaultController<Uint8Array> | undefined;
  new ReadableStream<Uint8Array>({
    start(streamController) {
      controller = streamController;
    },
  });
  if (controller === undefined) {
    throw new Error("Expected readable stream controller.");
  }
  return controller;
}

describe("server SSE scaffolding", () => {
  test("returns shared SSE headers", () => {
    expect(sseHeaders()).toEqual({
      "content-type": "text/event-stream; charset=utf-8",
      "cache-control": "no-cache",
      connection: "keep-alive",
    });
  });

  test("returns the model iterator when the protocol handler stops early", async () => {
    const first: GenerationStreamEvent = { type: "text", text: "hello" };
    const second: GenerationStreamEvent = { type: "text", text: "ignored" };
    const events = [first, second];
    let index = 0;
    let returned = false;
    const seen: GenerationStreamEvent[] = [];
    const iterator: AsyncIterator<GenerationStreamEvent> = {
      next() {
        const value = events[index];
        index += 1;
        return Promise.resolve(
          value === undefined ? { done: true, value: undefined } : { done: false, value },
        );
      },
      return() {
        returned = true;
        return Promise.resolve({ done: true, value: undefined });
      },
    };

    await runSseGenerationStream(
      createController(),
      iterator,
      { id: "stream-test", created: 0 },
      (event) => {
        seen.push(event);
        return true;
      },
    );

    expect(seen).toEqual([first]);
    expect(returned).toBe(true);
  });
});
