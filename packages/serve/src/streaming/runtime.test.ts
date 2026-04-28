import { describe, expect, test } from "bun:test";
import type { GenerationStreamEvent } from "../types";
import { closeStreamEvents, enqueueObservedSse, readStreamEvent } from "./runtime";

describe("server stream runtime", () => {
  test("returns cancellation immediately when the signal is already aborted", async () => {
    const controller = new AbortController();
    controller.abort();
    let nextCalled = false;
    let returned = false;
    const iterator: AsyncIterator<GenerationStreamEvent> = {
      next() {
        nextCalled = true;
        return Promise.resolve({ done: true, value: undefined });
      },
      return() {
        returned = true;
        return Promise.resolve({ done: true, value: undefined });
      },
    };

    await expect(readStreamEvent(iterator, controller.signal)).resolves.toEqual({
      type: "cancelled",
    });
    expect(nextCalled).toBe(false);
    expect(returned).toBe(true);
  });

  test("cancels a pending iterator read when the request signal aborts", async () => {
    const controller = new AbortController();
    let returned = false;
    const iterator: AsyncIterator<GenerationStreamEvent> = {
      next() {
        return new Promise<IteratorResult<GenerationStreamEvent>>(() => {});
      },
      return() {
        returned = true;
        return Promise.resolve({ done: true, value: undefined });
      },
    };

    const read = readStreamEvent(iterator, controller.signal);
    controller.abort();

    await expect(read).resolves.toEqual({ type: "cancelled" });
    expect(returned).toBe(true);
  });

  test("returns events and finished results from the iterator", async () => {
    const event: GenerationStreamEvent = { type: "text", text: "hello" };
    const iterator = [event][Symbol.iterator]();
    const streamIterator: AsyncIterator<GenerationStreamEvent> = {
      next() {
        return Promise.resolve(iterator.next());
      },
    };

    await expect(readStreamEvent(streamIterator, undefined)).resolves.toEqual({
      type: "event",
      event,
    });
    await expect(readStreamEvent(streamIterator, undefined)).resolves.toEqual({
      type: "finished",
    });
  });

  test("closes stream iterators through return", async () => {
    let returned = false;
    const iterator: AsyncIterator<GenerationStreamEvent> = {
      next() {
        return Promise.resolve({ done: true, value: undefined });
      },
      return() {
        returned = true;
        return Promise.resolve({ done: true, value: undefined });
      },
    };

    await closeStreamEvents(iterator);

    expect(returned).toBe(true);
  });

  test("aborts and rethrows when enqueueing into a closed stream fails", () => {
    let controller: ReadableStreamDefaultController<Uint8Array> | undefined;
    new ReadableStream<Uint8Array>({
      start(streamController) {
        controller = streamController;
        streamController.close();
      },
    });
    let aborted = false;
    if (controller === undefined) {
      throw new Error("Expected stream controller to be created.");
    }
    const streamController = controller;

    expect(() =>
      enqueueObservedSse(
        streamController,
        "data: test\n\n",
        { abort: () => (aborted = true) },
        "protocol",
      ),
    ).toThrow();
    expect(aborted).toBe(true);
  });
});
