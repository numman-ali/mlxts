import { describe, expect, test } from "bun:test";

import { createConcurrencyLimitGenerationEngine } from "./concurrency-engine";
import type { GenerationEngine, NormalizedGenerationRequest } from "./types";

function request(id: string): NormalizedGenerationRequest {
  return {
    id,
    model: "tiny",
    input: { kind: "text", text: id },
    sampling: { maxTokens: 1 },
    stream: false,
    protocol: "openai.completions",
  };
}

describe("concurrency limit generation engine", () => {
  test("serializes generate calls through one shared permit", async () => {
    let inFlight = 0;
    let maxInFlight = 0;
    const engine = createConcurrencyLimitGenerationEngine({
      maxConcurrentRequests: 1,
      engine: {
        async generate(normalized) {
          inFlight += 1;
          maxInFlight = Math.max(maxInFlight, inFlight);
          await Bun.sleep(5);
          inFlight -= 1;
          return { text: normalized.id, finishReason: "stop" };
        },
      },
    });

    const results = await Promise.all([
      engine.generate(request("a")),
      engine.generate(request("b")),
    ]);

    expect(results.map((result) => result.text)).toEqual(["a", "b"]);
    expect(maxInFlight).toBe(1);
  });

  test("preserves inner batch execution under one permit", async () => {
    const batchSizes: number[] = [];
    const engine = createConcurrencyLimitGenerationEngine({
      maxConcurrentRequests: 1,
      engine: {
        async generate() {
          throw new Error("generate should not be used");
        },
        async generateBatch(requests) {
          batchSizes.push(requests.length);
          await Bun.sleep(5);
          return requests.map((normalized) => ({ text: normalized.id, finishReason: "stop" }));
        },
      },
    });

    const results = await engine.generateBatch?.([request("a"), request("b")]);

    expect(batchSizes).toEqual([2]);
    expect(results?.map((result) => result.text)).toEqual(["a", "b"]);
  });

  test("holds the permit for the lifetime of a stream", async () => {
    const events: string[] = [];
    let streamEntered = false;
    const engine: GenerationEngine = createConcurrencyLimitGenerationEngine({
      maxConcurrentRequests: 1,
      engine: {
        async generate(normalized) {
          events.push(`generate:${normalized.id}`);
          return { text: normalized.id, finishReason: "stop" };
        },
        async *stream() {
          streamEntered = true;
          events.push("stream:start");
          yield { type: "text", text: "hello" };
          await Bun.sleep(5);
          events.push("stream:done");
          yield { type: "done", finishReason: "stop" };
        },
      },
    });

    const stream = await engine.stream?.(request("stream"));
    const generated = engine.generate(request("queued"));
    const seen: string[] = [];

    for await (const event of stream ?? []) {
      if (event.type === "text") {
        seen.push(event.text);
      }
    }
    const result = await generated;

    expect(streamEntered).toBe(true);
    expect(seen).toEqual(["hello"]);
    expect(result.text).toBe("queued");
    expect(events).toEqual(["stream:start", "stream:done", "generate:queued"]);
  });

  test("releases the permit when stream startup throws before yielding", async () => {
    let attempts = 0;
    const engine: GenerationEngine = createConcurrencyLimitGenerationEngine({
      maxConcurrentRequests: 1,
      engine: {
        generate(normalized) {
          return { text: normalized.id, finishReason: "stop" };
        },
        async stream() {
          attempts += 1;
          throw new Error("stream setup failed");
        },
      },
    });

    await expect(engine.stream?.(request("broken"))).rejects.toThrow("stream setup failed");
    const result = await engine.generate(request("queued"));

    expect(attempts).toBe(1);
    expect(result.text).toBe("queued");
  });

  test("rejects invalid maxConcurrentRequests values", () => {
    expect(() =>
      createConcurrencyLimitGenerationEngine({
        maxConcurrentRequests: 0,
        engine: { generate: () => ({ text: "", finishReason: "stop" }) },
      }),
    ).toThrow("maxConcurrentRequests");
  });
});
