import { describe, expect, test } from "bun:test";

import { createMicroBatchingGenerationEngine } from "./batching-engine";
import type {
  GenerationEngine,
  GenerationStreamEvent,
  NormalizedGenerationRequest,
  ServeEvent,
} from "./types";

function textRequest(id: string): NormalizedGenerationRequest {
  return {
    id,
    model: "tiny",
    input: { kind: "text", text: id },
    sampling: { maxTokens: 1 },
    stream: false,
    protocol: "openai.completions",
  };
}

describe("micro-batching generation engine", () => {
  test("validates batching options", () => {
    const inner: GenerationEngine = {
      generate(request) {
        return { text: request.id, finishReason: "stop" };
      },
    };

    expect(() => createMicroBatchingGenerationEngine({ engine: inner, maxBatchSize: 0 })).toThrow(
      "maxBatchSize",
    );
    expect(() => createMicroBatchingGenerationEngine({ engine: inner, batchWindowMs: -1 })).toThrow(
      "batchWindowMs",
    );
  });

  test("coalesces nearby generate calls into one batch-capable engine call", async () => {
    const batchIds: string[][] = [];
    const events: ServeEvent[] = [];
    const inner: GenerationEngine = {
      generate() {
        throw new Error("generate should not be called when generateBatch is available");
      },
      generateBatch(requests) {
        batchIds.push(requests.map((request) => request.id));
        return requests.map((request) => ({ text: `ok:${request.id}`, finishReason: "stop" }));
      },
    };
    const engine = createMicroBatchingGenerationEngine({
      engine: inner,
      batchWindowMs: 0,
      maxBatchSize: 8,
      onEvent: (event) => events.push(event),
    });

    const results = await Promise.all([
      engine.generate(textRequest("agent-1")),
      engine.generate(textRequest("agent-2")),
      engine.generate(textRequest("agent-3")),
      engine.generate(textRequest("agent-4")),
    ]);

    expect(batchIds).toEqual([["agent-1", "agent-2", "agent-3", "agent-4"]]);
    expect(events).toEqual([
      {
        type: "generation_admission_batch",
        mode: "micro",
        engineMode: "batch",
        model: "tiny",
        ids: ["agent-1", "agent-2", "agent-3", "agent-4"],
        batchSize: 4,
        maxTokens: 1,
        maxTokensByRequest: [1, 1, 1, 1],
      },
    ]);
    expect(results.map((result) => result.text)).toEqual([
      "ok:agent-1",
      "ok:agent-2",
      "ok:agent-3",
      "ok:agent-4",
    ]);
  });

  test("splits batches at the configured maximum and delegates streaming", async () => {
    const batchSizes: number[] = [];
    const inner: GenerationEngine = {
      generateBatch(requests) {
        batchSizes.push(requests.length);
        return requests.map((request) => ({ text: request.id, finishReason: "stop" }));
      },
      generate(request) {
        return { text: request.id, finishReason: "stop" };
      },
      async *stream() {
        yield { type: "done", finishReason: "stop" };
      },
    };
    const engine = createMicroBatchingGenerationEngine({
      engine: inner,
      batchWindowMs: 0,
      maxBatchSize: 2,
    });

    await Promise.all([
      engine.generate(textRequest("a")),
      engine.generate(textRequest("b")),
      engine.generate(textRequest("c")),
    ]);
    const stream = await engine.stream?.(textRequest("stream"));
    const events: GenerationStreamEvent[] = [];
    if (stream !== undefined) {
      for await (const event of stream) {
        events.push(event);
      }
    }

    expect(batchSizes).toEqual([2, 1]);
    expect(events).toEqual([{ type: "done", finishReason: "stop" }]);
  });

  test("uses a timed flush window when configured", async () => {
    const batchIds: string[][] = [];
    const inner: GenerationEngine = {
      generate() {
        throw new Error("generate should not be called when generateBatch is available");
      },
      generateBatch(requests) {
        batchIds.push(requests.map((request) => request.id));
        return requests.map((request) => ({ text: request.id, finishReason: "stop" }));
      },
    };
    const engine = createMicroBatchingGenerationEngine({
      engine: inner,
      batchWindowMs: 1,
      maxBatchSize: 4,
    });

    await expect(engine.generate(textRequest("timer"))).resolves.toEqual({
      text: "timer",
      finishReason: "stop",
    });
    expect(batchIds).toEqual([["timer"]]);
  });

  test("rejects incomplete batch result slots", async () => {
    const engine = createMicroBatchingGenerationEngine({
      engine: {
        generate() {
          throw new Error("generate should not be called when generateBatch is available");
        },
        generateBatch() {
          return [undefined as unknown as { text: string; finishReason: "stop" }];
        },
      },
      batchWindowMs: 0,
    });

    await expect(engine.generate(textRequest("hole"))).rejects.toThrow("incomplete batch result");
  });

  test("falls back to sequential generation when the inner engine is not batch-capable", async () => {
    const seen: string[] = [];
    const events: ServeEvent[] = [];
    const inner: GenerationEngine = {
      generate(request) {
        seen.push(request.id);
        return { text: request.id, finishReason: "stop" };
      },
    };
    const engine = createMicroBatchingGenerationEngine({
      engine: inner,
      batchWindowMs: 0,
      onEvent: (event) => events.push(event),
    });

    const results = await engine.generateBatch?.([textRequest("a"), textRequest("b")]);

    expect(seen).toEqual(["a", "b"]);
    expect(events).toEqual([
      {
        type: "generation_admission_batch",
        mode: "micro",
        engineMode: "sequential",
        model: "tiny",
        ids: ["a", "b"],
        batchSize: 2,
        maxTokens: 1,
        maxTokensByRequest: [1, 1],
      },
    ]);
    expect(results?.map((result) => result.text)).toEqual(["a", "b"]);
  });

  test("keeps fallback sequential generation single-flight across queued batches", async () => {
    let inFlight = 0;
    let maxInFlight = 0;
    const seen: string[] = [];
    const inner: GenerationEngine = {
      async generate(request) {
        inFlight += 1;
        maxInFlight = Math.max(maxInFlight, inFlight);
        seen.push(`start:${request.id}`);
        await Bun.sleep(5);
        seen.push(`end:${request.id}`);
        inFlight -= 1;
        return { text: request.id, finishReason: "stop" };
      },
    };
    const engine = createMicroBatchingGenerationEngine({
      engine: inner,
      batchWindowMs: 0,
      maxBatchSize: 1,
    });

    const results = await Promise.all([
      engine.generate(textRequest("a")),
      engine.generate(textRequest("b")),
      engine.generate(textRequest("c")),
    ]);

    expect(maxInFlight).toBe(1);
    expect(seen).toEqual(["start:a", "end:a", "start:b", "end:b", "start:c", "end:c"]);
    expect(results.map((result) => result.text)).toEqual(["a", "b", "c"]);
  });

  test("rejects queued requests when the inner batch call fails or returns the wrong size", async () => {
    const failing = createMicroBatchingGenerationEngine({
      engine: {
        generate() {
          throw new Error("generate should not be called");
        },
        generateBatch() {
          throw new Error("batch failed");
        },
      },
      batchWindowMs: 0,
    });
    await expect(failing.generate(textRequest("fail"))).rejects.toThrow("batch failed");

    const wrongSize = createMicroBatchingGenerationEngine({
      engine: {
        generate() {
          throw new Error("generate should not be called");
        },
        generateBatch() {
          return [];
        },
      },
      batchWindowMs: 0,
    });
    await expect(wrongSize.generate(textRequest("wrong"))).rejects.toThrow(
      "wrong number of batch results",
    );
  });
});
