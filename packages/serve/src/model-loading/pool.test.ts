import { describe, expect, test } from "bun:test";
import { createFetchHandler } from "../http/server";
import type {
  GenerationEngine,
  GenerationStreamEvent,
  NormalizedGenerationRequest,
} from "../types";
import { createSourceModelPoolGenerationEngine, type LoadedSourceModelPoolEntry } from "./pool";

function request(id: string, model: string): NormalizedGenerationRequest {
  return {
    id,
    model,
    input: { kind: "text", text: id },
    sampling: { maxTokens: 1 },
    stream: false,
    protocol: "openai.completions",
  };
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

function httpRequest(path: string, body: unknown): Request {
  return new Request(`http://localhost${path}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
}

function deferred<T>() {
  let resolveValue: (value: T) => void = () => {};
  const promise = new Promise<T>((resolve) => {
    resolveValue = resolve;
  });
  return { promise, resolve: resolveValue };
}

async function readWithTimeout<T>(promise: Promise<T>): Promise<T> {
  return Promise.race([
    promise,
    new Promise<never>((_, reject) => {
      setTimeout(() => reject(new Error("timed out waiting for streamed bytes")), 1000);
    }),
  ]);
}

async function drainReader(reader: ReadableStreamDefaultReader<Uint8Array>): Promise<void> {
  while (true) {
    const chunk = await readWithTimeout(reader.read());
    if (chunk.done) {
      return;
    }
  }
}

async function expectRejectionMessage(
  result: unknown | Promise<unknown>,
  message: string,
): Promise<void> {
  let rejection: unknown;
  try {
    await result;
  } catch (error) {
    rejection = error;
  }
  expect(rejection).toBeInstanceOf(Error);
  if (!(rejection instanceof Error)) {
    throw new Error("Expected promise to reject with an Error.");
  }
  expect(rejection.message).toContain(message);
}

function loadedEntry(
  tag: string,
  dispose: () => void,
  streamEvents: readonly GenerationStreamEvent[] = [
    { type: "text", text: tag },
    { type: "done", finishReason: "stop" },
  ],
): LoadedSourceModelPoolEntry {
  const engine: GenerationEngine = {
    generate(normalized) {
      return { text: `${tag}:${normalized.id}`, finishReason: "stop" };
    },
    generateBatch(requests) {
      return requests.map((normalized) => ({
        text: `${tag}:${normalized.id}`,
        finishReason: "stop",
      }));
    },
    async *stream() {
      for (const event of streamEvents) {
        yield event;
      }
    },
  };
  return { engine, dispose };
}

describe("source model pool generation engine", () => {
  test("validates model entries and stopped pools", async () => {
    expect(() =>
      createSourceModelPoolGenerationEngine({
        entries: [],
        async load() {
          return loadedEntry("unused", () => {});
        },
      }),
    ).toThrow("requires at least one model source");
    expect(() =>
      createSourceModelPoolGenerationEngine({
        entries: [{ modelId: "alpha" }, { modelId: "alpha" }],
        async load() {
          return loadedEntry("unused", () => {});
        },
      }),
    ).toThrow("duplicate model id");

    const engine = createSourceModelPoolGenerationEngine({
      entries: [{ modelId: "alpha" }],
      async load() {
        return loadedEntry("a", () => {});
      },
    });
    engine[Symbol.dispose]?.();
    engine[Symbol.dispose]?.();

    await expect(engine.generate(request("one", "alpha"))).rejects.toThrow("pool is stopped");
  });

  test("loads a model once and shares concurrent first requests", async () => {
    const loadGate = deferred<LoadedSourceModelPoolEntry>();
    const loads: string[] = [];
    const engine = createSourceModelPoolGenerationEngine({
      entries: [{ modelId: "alpha" }],
      async load(entry) {
        loads.push(entry.modelId);
        return loadGate.promise;
      },
    });

    const first = engine.generate(request("one", "alpha"));
    const second = engine.generate(request("two", "alpha"));
    await sleep(0);
    expect(loads).toEqual(["alpha"]);

    loadGate.resolve(loadedEntry("a", () => {}));

    await expect(first).resolves.toMatchObject({ text: "a:one" });
    await expect(second).resolves.toMatchObject({ text: "a:two" });
  });

  test("reuses already loaded models without reloading", async () => {
    const loads: string[] = [];
    const engine = createSourceModelPoolGenerationEngine({
      entries: [{ modelId: "alpha" }],
      async load(entry) {
        loads.push(entry.modelId);
        return loadedEntry("a", () => {});
      },
    });

    await expect(engine.generate(request("one", "alpha"))).resolves.toMatchObject({
      text: "a:one",
    });
    await expect(engine.generate(request("two", "alpha"))).resolves.toMatchObject({
      text: "a:two",
    });
    expect(loads).toEqual(["alpha"]);
  });

  test("rejects malformed inner batch results", async () => {
    const engine = createSourceModelPoolGenerationEngine({
      entries: [{ modelId: "alpha" }],
      async load() {
        return {
          engine: {
            generate() {
              return { text: "a", finishReason: "stop" };
            },
            generateBatch() {
              return [{ text: "a:one", finishReason: "stop" }];
            },
          },
          dispose() {},
        };
      },
    });

    await expect(
      engine.generateBatch?.([request("one", "alpha"), request("two", "alpha")]),
    ).rejects.toThrow("wrong number of batch results");
  });

  test("serializes cold loads for different models", async () => {
    const alphaGate = deferred<LoadedSourceModelPoolEntry>();
    const loads: string[] = [];
    const engine = createSourceModelPoolGenerationEngine({
      entries: [{ modelId: "alpha" }, { modelId: "beta" }],
      async load(entry) {
        loads.push(entry.modelId);
        if (entry.modelId === "alpha") {
          return alphaGate.promise;
        }
        return loadedEntry("b", () => {});
      },
    });

    const first = engine.generate(request("one", "alpha"));
    const second = engine.generate(request("two", "beta"));
    await sleep(0);
    expect(loads).toEqual(["alpha"]);

    alphaGate.resolve(loadedEntry("a", () => {}));

    await expect(first).resolves.toMatchObject({ text: "a:one" });
    await expect(second).resolves.toMatchObject({ text: "b:two" });
    expect(loads).toEqual(["alpha", "beta"]);
  });

  test("routes batches by loaded model while preserving result order", async () => {
    const engine = createSourceModelPoolGenerationEngine({
      entries: [{ modelId: "alpha" }, { modelId: "beta" }],
      async load(entry) {
        return loadedEntry(entry.modelId.slice(0, 1), () => {});
      },
    });

    const results = await engine.generateBatch?.([
      request("one", "beta"),
      request("two", "alpha"),
      request("three", "beta"),
    ]);

    expect(results?.map((result) => result.text)).toEqual(["b:one", "a:two", "b:three"]);
  });

  test("evicts idle non-pinned models and reloads them on later requests", async () => {
    const loads: string[] = [];
    let disposeCount = 0;
    const engine = createSourceModelPoolGenerationEngine({
      entries: [{ modelId: "alpha" }],
      idleTtlMs: 1,
      async load(entry) {
        loads.push(entry.modelId);
        return loadedEntry(`a${loads.length}`, () => {
          disposeCount += 1;
        });
      },
    });

    await expect(engine.generate(request("one", "alpha"))).resolves.toMatchObject({
      text: "a1:one",
    });
    await sleep(10);
    expect(disposeCount).toBe(1);

    await expect(engine.generate(request("two", "alpha"))).resolves.toMatchObject({
      text: "a2:two",
    });
    engine[Symbol.dispose]?.();

    expect(loads).toEqual(["alpha", "alpha"]);
    expect(disposeCount).toBe(2);
  });

  test("keeps pinned models loaded until pool disposal", async () => {
    let disposeCount = 0;
    const engine = createSourceModelPoolGenerationEngine({
      entries: [{ modelId: "alpha", pinned: true }],
      idleTtlMs: 1,
      async load() {
        return loadedEntry("a", () => {
          disposeCount += 1;
        });
      },
    });

    await engine.generate(request("one", "alpha"));
    await sleep(10);
    expect(disposeCount).toBe(0);

    engine[Symbol.dispose]?.();
    expect(disposeCount).toBe(1);
  });

  test("holds active stream leases until the stream closes", async () => {
    let disposeCount = 0;
    const engine = createSourceModelPoolGenerationEngine({
      entries: [{ modelId: "alpha" }],
      idleTtlMs: 1,
      async load() {
        return loadedEntry("a", () => {
          disposeCount += 1;
        });
      },
    });

    const stream = await engine.stream?.(request("stream", "alpha"));
    const iterator = stream?.[Symbol.asyncIterator]();
    await iterator?.next();
    await sleep(10);
    expect(disposeCount).toBe(0);

    await iterator?.return?.();
    await sleep(10);
    expect(disposeCount).toBe(1);
  });

  test("keeps HTTP SSE streams leased until the response body closes", async () => {
    const continueStream = deferred<void>();
    const loads: string[] = [];
    let disposeCount = 0;
    const engine = createSourceModelPoolGenerationEngine({
      entries: [{ modelId: "alpha" }],
      idleTtlMs: 1,
      async load(entry) {
        loads.push(entry.modelId);
        const tag = `a${loads.length}`;
        return {
          engine: {
            generate(normalized) {
              return { text: `${tag}:${normalized.id}`, finishReason: "stop" };
            },
            async *stream() {
              yield { type: "text", text: tag } satisfies GenerationStreamEvent;
              await continueStream.promise;
              yield { type: "done", finishReason: "stop" } satisfies GenerationStreamEvent;
            },
          },
          dispose() {
            disposeCount += 1;
          },
        };
      },
    });
    const fetch = createFetchHandler({
      engine,
      models: [{ id: "alpha" }],
      idGenerator: () => "cmpl-lazy",
    });

    const response = await fetch(
      httpRequest("/v1/completions", {
        model: "alpha",
        prompt: "hello",
        stream: true,
      }),
    );
    const reader = response.body?.getReader();
    expect(response.status).toBe(200);
    expect(response.headers.get("content-type")).toContain("text/event-stream");
    expect(reader).toBeDefined();
    if (reader === undefined) {
      throw new Error("expected a response body reader");
    }

    const firstChunk = await readWithTimeout(reader.read());
    expect(firstChunk.done).toBe(false);
    await sleep(10);
    expect(disposeCount).toBe(0);

    continueStream.resolve();
    await drainReader(reader);
    await sleep(10);
    expect(disposeCount).toBe(1);

    const followup = await fetch(
      httpRequest("/v1/completions", {
        model: "alpha",
        prompt: "again",
      }),
    );
    const body = await followup.json();
    expect(followup.status).toBe(200);
    expect(body.choices[0].text).toBe("a2:cmpl-lazy");
    expect(loads).toEqual(["alpha", "alpha"]);
    engine[Symbol.dispose]?.();
    expect(disposeCount).toBe(2);
  });

  test("releases HTTP SSE stream leases when the reader cancels", async () => {
    let streamClosed = false;
    let seenSignal: AbortSignal | undefined;
    let disposeCount = 0;
    const engine = createSourceModelPoolGenerationEngine({
      entries: [{ modelId: "alpha" }],
      idleTtlMs: 1,
      async load() {
        return {
          engine: {
            generate() {
              return { text: "unused", finishReason: "stop" };
            },
            async *stream(normalized) {
              seenSignal = normalized.abortSignal;
              try {
                yield { type: "text", text: "a" } satisfies GenerationStreamEvent;
                await new Promise<void>((resolve) => {
                  if (normalized.abortSignal?.aborted === true) {
                    resolve();
                    return;
                  }
                  normalized.abortSignal?.addEventListener("abort", () => resolve(), {
                    once: true,
                  });
                });
              } finally {
                streamClosed = true;
              }
            },
          },
          dispose() {
            disposeCount += 1;
          },
        };
      },
    });
    const fetch = createFetchHandler({ engine, models: [{ id: "alpha" }] });
    const response = await fetch(
      httpRequest("/v1/completions", {
        model: "alpha",
        prompt: "hello",
        stream: true,
      }),
    );
    const reader = response.body?.getReader();
    expect(reader).toBeDefined();
    if (reader === undefined) {
      throw new Error("expected a response body reader");
    }

    const firstChunk = await readWithTimeout(reader.read());
    expect(firstChunk.done).toBe(false);
    await sleep(10);
    expect(disposeCount).toBe(0);

    await reader.cancel();
    await sleep(25);
    expect(seenSignal?.aborted).toBe(true);
    expect(streamClosed).toBe(true);
    expect(disposeCount).toBe(1);
  });

  test("disposes a model that finishes loading after the pool stops", async () => {
    const loadGate = deferred<LoadedSourceModelPoolEntry>();
    let disposeCount = 0;
    const engine = createSourceModelPoolGenerationEngine({
      entries: [{ modelId: "alpha" }],
      async load() {
        return loadGate.promise;
      },
    });

    const pending = engine.generate(request("one", "alpha"));
    await sleep(0);
    const rejected = expectRejectionMessage(pending, "stopped while loading");
    engine[Symbol.dispose]?.();
    loadGate.resolve(
      loadedEntry("a", () => {
        disposeCount += 1;
      }),
    );

    await rejected;
    expect(disposeCount).toBe(1);
  });

  test("does not start queued cold loads after the pool stops", async () => {
    const alphaGate = deferred<LoadedSourceModelPoolEntry>();
    const loads: string[] = [];
    const disposals: string[] = [];
    const engine = createSourceModelPoolGenerationEngine({
      entries: [{ modelId: "alpha" }, { modelId: "beta" }],
      async load(entry) {
        loads.push(entry.modelId);
        if (entry.modelId === "alpha") {
          return alphaGate.promise;
        }
        return loadedEntry("b", () => {
          disposals.push("beta");
        });
      },
    });

    const alpha = engine.generate(request("one", "alpha"));
    const beta = engine.generate(request("two", "beta"));
    await sleep(0);
    expect(loads).toEqual(["alpha"]);

    engine[Symbol.dispose]?.();
    const alphaRejected = expectRejectionMessage(alpha, "stopped while loading");
    const betaRejected = expectRejectionMessage(beta, "pool is stopped");
    alphaGate.resolve(
      loadedEntry("a", () => {
        disposals.push("alpha");
      }),
    );

    await Promise.all([alphaRejected, betaRejected]);
    expect(loads).toEqual(["alpha"]);
    expect(disposals).toEqual(["alpha"]);
  });

  test("reports unknown models and unsupported streaming", async () => {
    const engine = createSourceModelPoolGenerationEngine({
      entries: [{ modelId: "alpha" }],
      async load() {
        return {
          engine: {
            generate() {
              return { text: "a", finishReason: "stop" };
            },
          },
          dispose() {},
        };
      },
    });

    await expect(engine.generate(request("one", "missing"))).rejects.toThrow(
      "not served by this endpoint",
    );
    await expect(engine.stream?.(request("stream", "alpha"))).rejects.toThrow(
      "does not support streaming",
    );
  });

  test("releases stream leases when stream startup throws", async () => {
    let disposeCount = 0;
    const engine = createSourceModelPoolGenerationEngine({
      entries: [{ modelId: "alpha" }],
      idleTtlMs: 1,
      async load() {
        return {
          engine: {
            generate() {
              return { text: "a", finishReason: "stop" };
            },
            stream() {
              throw new Error("stream startup failed");
            },
          },
          dispose() {
            disposeCount += 1;
          },
        };
      },
    });

    await expect(engine.stream?.(request("stream", "alpha"))).rejects.toThrow(
      "stream startup failed",
    );
    await sleep(10);
    expect(disposeCount).toBe(1);
  });
});
