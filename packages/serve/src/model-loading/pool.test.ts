import { describe, expect, test } from "bun:test";
import { ServeError } from "../errors";
import { createFetchHandler } from "../http/server";
import type {
  GenerationEngine,
  GenerationStreamEvent,
  NormalizedGenerationRequest,
  ServeEvent,
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

async function expectServeErrorCode(
  result: unknown | Promise<unknown>,
  code: string,
): Promise<void> {
  let rejection: unknown;
  try {
    await result;
  } catch (error) {
    rejection = error;
  }
  expect(rejection).toBeInstanceOf(ServeError);
  if (!(rejection instanceof ServeError)) {
    throw new Error("Expected promise to reject with a ServeError.");
  }
  expect(rejection.code).toBe(code);
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
    promptPrefixCacheInfo() {
      return {
        models: [
          {
            id: tag,
            retainedSnapshots: 1,
            retainedSnapshotBytes: 1024,
            indexedBlockHashes: 2,
            tokenBlocks: {
              blockSize: 64,
              blockCount: 2,
              blockReferences: 2,
              uniqueTokenCount: 128,
              referencedTokenCount: 128,
            },
          },
        ],
        totalRetainedSnapshots: 1,
        totalRetainedSnapshotBytes: 1024,
        totalIndexedBlockHashes: 2,
        totalTokenBlocks: 2,
        totalTokenBlockReferences: 2,
        totalUniqueTokenCount: 128,
        totalReferencedTokenCount: 128,
      };
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

  test("reports lazy model-pool status without loading extra models", async () => {
    const engine = createSourceModelPoolGenerationEngine({
      entries: [{ modelId: "alpha" }, { modelId: "beta", pinned: true }],
      idleTtlMs: 10_000,
      pressurePolicy: "shed_non_pinned",
      pressureReleaseTimeoutMs: 500,
      async load(entry) {
        return loadedEntry(entry.modelId, () => {});
      },
    });

    expect(engine.modelPoolInfo?.()).toMatchObject({
      load_policy: "lazy",
      pressure_policy: "shed_non_pinned",
      pressure_release_timeout_ms: 500,
      idle_ttl_ms: 10_000,
      models: [
        {
          id: "alpha",
          pinned: false,
          state: "not_loaded",
          active_requests: 0,
          pressure_aborted_requests: 0,
        },
        {
          id: "beta",
          pinned: true,
          state: "not_loaded",
          active_requests: 0,
          pressure_aborted_requests: 0,
        },
      ],
    });
    expect(engine.promptPrefixCacheInfo?.()).toMatchObject({
      totalRetainedSnapshots: 0,
      models: [],
    });

    const stream = await engine.stream?.({ ...request("stream", "alpha"), stream: true });
    if (stream === undefined) {
      throw new Error("Expected lazy pool engine to expose streaming.");
    }

    expect(engine.modelPoolInfo?.().models).toEqual([
      {
        id: "alpha",
        pinned: false,
        state: "loaded",
        active_requests: 1,
        pressure_aborted_requests: 0,
      },
      {
        id: "beta",
        pinned: true,
        state: "not_loaded",
        active_requests: 0,
        pressure_aborted_requests: 0,
      },
    ]);
    expect(engine.promptPrefixCacheInfo?.()).toMatchObject({
      totalRetainedSnapshots: 1,
      totalRetainedSnapshotBytes: 1024,
      totalTokenBlocks: 2,
      models: [{ id: "alpha" }],
    });

    let streamed = 0;
    for await (const _event of stream) {
      streamed += 1;
    }
    expect(streamed).toBeGreaterThan(0);
    expect(engine.modelPoolInfo?.().models[0]).toMatchObject({
      id: "alpha",
      state: "loaded",
      active_requests: 0,
    });
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

  test("evicts idle non-pinned models to retry cold loads under memory pressure", async () => {
    const loads: string[] = [];
    const events: ServeEvent[] = [];
    let alphaDisposals = 0;
    let betaAttempts = 0;
    const engine = createSourceModelPoolGenerationEngine({
      entries: [{ modelId: "alpha" }, { modelId: "beta" }],
      pressurePolicy: "shed_non_pinned",
      onEvent(event) {
        events.push(event);
      },
      async load(entry) {
        loads.push(entry.modelId);
        if (entry.modelId === "alpha") {
          return loadedEntry("a", () => {
            alphaDisposals += 1;
          });
        }
        betaAttempts += 1;
        if (betaAttempts === 1) {
          throw new ServeError("beta needs more model memory", {
            code: "model_load_memory_exceeded",
            status: 503,
          });
        }
        return loadedEntry("b", () => {});
      },
    });

    await expect(engine.generate(request("one", "alpha"))).resolves.toMatchObject({
      text: "a:one",
    });
    await expect(engine.generate(request("two", "beta"))).resolves.toMatchObject({
      text: "b:two",
    });

    expect(loads).toEqual(["alpha", "beta", "beta"]);
    expect(betaAttempts).toBe(2);
    expect(alphaDisposals).toBe(1);
    expect(events).toContainEqual({
      type: "model_pool_pressure",
      targetModel: "beta",
      action: "evict_idle",
      reason: "model_load_memory_exceeded",
      evictedModels: ["alpha"],
      abortedRequestIds: [],
      activeRequests: 0,
    });
  });

  test("keeps pinned idle models loaded when cold-load pressure cannot be relieved", async () => {
    let alphaDisposals = 0;
    let betaAttempts = 0;
    const events: ServeEvent[] = [];
    const engine = createSourceModelPoolGenerationEngine({
      entries: [{ modelId: "alpha", pinned: true }, { modelId: "beta" }],
      pressurePolicy: "shed_non_pinned",
      onEvent(event) {
        events.push(event);
      },
      async load(entry) {
        if (entry.modelId === "alpha") {
          return loadedEntry("a", () => {
            alphaDisposals += 1;
          });
        }
        betaAttempts += 1;
        throw new ServeError("beta needs more model memory", {
          code: "model_load_memory_exceeded",
          status: 503,
        });
      },
    });

    await engine.generate(request("one", "alpha"));
    await expectServeErrorCode(
      engine.generate(request("two", "beta")),
      "model_load_memory_exceeded",
    );

    expect(betaAttempts).toBe(1);
    expect(alphaDisposals).toBe(0);
    expect(events).toEqual([]);
    engine[Symbol.dispose]?.();
    expect(alphaDisposals).toBe(1);
  });

  test("aborts active non-pinned streams before retrying cold loads", async () => {
    const events: ServeEvent[] = [];
    let betaAttempts = 0;
    const engine = createSourceModelPoolGenerationEngine({
      entries: [{ modelId: "alpha" }, { modelId: "beta" }],
      pressurePolicy: "shed_non_pinned",
      onEvent(event) {
        events.push(event);
      },
      async load(entry) {
        if (entry.modelId === "alpha") {
          return {
            engine: {
              generate(normalized) {
                return { text: `a:${normalized.id}`, finishReason: "stop" };
              },
              async *stream(normalized) {
                yield { type: "text", text: "a" } satisfies GenerationStreamEvent;
                await new Promise<void>((resolve) => {
                  normalized.abortSignal?.addEventListener("abort", () => resolve(), {
                    once: true,
                  });
                });
                yield { type: "done", finishReason: "cancelled" } satisfies GenerationStreamEvent;
              },
            },
            dispose() {},
          };
        }
        betaAttempts += 1;
        if (betaAttempts === 1) {
          throw new ServeError("beta needs more model memory", {
            code: "model_load_memory_exceeded",
            status: 503,
          });
        }
        return loadedEntry("b", () => {});
      },
    });

    const stream = await engine.stream?.(request("stream", "alpha"));
    const iterator = stream?.[Symbol.asyncIterator]();
    if (iterator === undefined) {
      throw new Error("expected stream iterator");
    }
    await expect(iterator.next()).resolves.toMatchObject({
      value: { type: "text", text: "a" },
    });
    const pressureCancelled = iterator.next();
    const beta = engine.generate(request("cold", "beta"));

    await expectServeErrorCode(pressureCancelled, "model_pool_memory_pressure");
    await expect(beta).resolves.toMatchObject({ text: "b:cold" });

    expect(betaAttempts).toBe(2);
    expect(events).toContainEqual({
      type: "model_pool_pressure",
      targetModel: "beta",
      action: "abort_active",
      reason: "model_load_memory_exceeded",
      evictedModels: [],
      abortedRequestIds: ["stream"],
      activeRequests: 1,
    });
  });

  test("aborts competing active leases before retrying request memory preflight", async () => {
    const events: ServeEvent[] = [];
    let shortAttempts = 0;
    const engine = createSourceModelPoolGenerationEngine({
      entries: [{ modelId: "alpha" }],
      pressurePolicy: "shed_non_pinned",
      onEvent(event) {
        events.push(event);
      },
      async load() {
        return {
          engine: {
            async generate(normalized) {
              if (normalized.id === "long") {
                await new Promise<void>((_, reject) => {
                  normalized.abortSignal?.addEventListener(
                    "abort",
                    () => {
                      const error = new Error("aborted");
                      error.name = "AbortError";
                      reject(error);
                    },
                    { once: true },
                  );
                });
              }
              shortAttempts += 1;
              if (shortAttempts === 1) {
                throw new ServeError("request exceeds memory budget", {
                  code: "memory_budget_exceeded",
                  status: 429,
                });
              }
              return { text: `a:${normalized.id}`, finishReason: "stop" };
            },
          },
          dispose() {},
        };
      },
    });

    const long = engine.generate(request("long", "alpha"));
    await sleep(0);
    const short = engine.generate(request("short", "alpha"));

    await expectServeErrorCode(long, "model_pool_memory_pressure");
    await expect(short).resolves.toMatchObject({ text: "a:short" });
    expect(shortAttempts).toBe(2);
    expect(events).toContainEqual({
      type: "model_pool_pressure",
      targetModel: "alpha",
      action: "abort_active",
      reason: "memory_budget_exceeded",
      evictedModels: [],
      abortedRequestIds: ["long"],
      activeRequests: 2,
    });
  });

  test("aborts active leases one at a time while request pressure persists", async () => {
    const events: ServeEvent[] = [];
    let shortAttempts = 0;
    const engine = createSourceModelPoolGenerationEngine({
      entries: [{ modelId: "alpha" }],
      pressurePolicy: "shed_non_pinned",
      onEvent(event) {
        events.push(event);
      },
      async load() {
        return {
          engine: {
            async generate(normalized) {
              if (normalized.id.startsWith("long")) {
                await new Promise<void>((_, reject) => {
                  normalized.abortSignal?.addEventListener(
                    "abort",
                    () => {
                      const error = new Error("aborted");
                      error.name = "AbortError";
                      reject(error);
                    },
                    { once: true },
                  );
                });
              }
              shortAttempts += 1;
              if (shortAttempts <= 2) {
                throw new ServeError("request exceeds memory budget", {
                  code: "memory_budget_exceeded",
                  status: 429,
                });
              }
              return { text: `a:${normalized.id}`, finishReason: "stop" };
            },
          },
          dispose() {},
        };
      },
    });

    const longA = engine.generate(request("long-a", "alpha"));
    await sleep(0);
    const longB = engine.generate(request("long-b", "alpha"));
    await sleep(0);
    const short = engine.generate(request("short", "alpha"));

    await expectServeErrorCode(longA, "model_pool_memory_pressure");
    await expectServeErrorCode(longB, "model_pool_memory_pressure");
    await expect(short).resolves.toMatchObject({ text: "a:short" });
    expect(shortAttempts).toBe(3);
    expect(
      events
        .filter((event) => event.type === "model_pool_pressure")
        .map((event) => event.abortedRequestIds),
    ).toEqual([["long-a"], ["long-b"]]);
  });

  test("retries concurrent pressure waiters before shedding another active lease", async () => {
    const events: ServeEvent[] = [];
    const releaseLong = new Map<string, () => void>();
    const bothShortsAttempted = deferred<void>();
    let shortFirstAttempts = 0;
    const shortAttempts = new Map<string, number>();
    const engine = createSourceModelPoolGenerationEngine({
      entries: [{ modelId: "alpha" }],
      pressurePolicy: "shed_non_pinned",
      onEvent(event) {
        events.push(event);
      },
      async load() {
        return {
          engine: {
            async generate(normalized) {
              if (normalized.id.startsWith("long")) {
                await new Promise<void>((resolve, reject) => {
                  releaseLong.set(normalized.id, resolve);
                  normalized.abortSignal?.addEventListener(
                    "abort",
                    () => {
                      const error = new Error("aborted");
                      error.name = "AbortError";
                      reject(error);
                    },
                    { once: true },
                  );
                });
                return { text: `a:${normalized.id}`, finishReason: "stop" };
              }
              const attempts = (shortAttempts.get(normalized.id) ?? 0) + 1;
              shortAttempts.set(normalized.id, attempts);
              if (attempts === 1) {
                shortFirstAttempts += 1;
                if (shortFirstAttempts === 2) {
                  bothShortsAttempted.resolve();
                }
                await bothShortsAttempted.promise;
                throw new ServeError("request exceeds memory budget", {
                  code: "memory_budget_exceeded",
                  status: 429,
                });
              }
              return { text: `a:${normalized.id}`, finishReason: "stop" };
            },
          },
          dispose() {},
        };
      },
    });

    const longA = engine.generate(request("long-a", "alpha"));
    await sleep(0);
    const longB = engine.generate(request("long-b", "alpha"));
    await sleep(0);
    const shortA = engine.generate(request("short-a", "alpha"));
    const shortB = engine.generate(request("short-b", "alpha"));

    await expectServeErrorCode(longA, "model_pool_memory_pressure");
    await expect(shortA).resolves.toMatchObject({ text: "a:short-a" });
    await expect(shortB).resolves.toMatchObject({ text: "a:short-b" });
    releaseLong.get("long-b")?.();
    await expect(longB).resolves.toMatchObject({ text: "a:long-b" });
    expect(
      events
        .filter((event) => event.type === "model_pool_pressure")
        .map((event) => event.abortedRequestIds),
    ).toEqual([["long-a"]]);
  });

  test("times out when pressure-aborted active leases do not release", async () => {
    const releaseLong = deferred<void>();
    const engine = createSourceModelPoolGenerationEngine({
      entries: [{ modelId: "alpha" }],
      pressurePolicy: "shed_non_pinned",
      pressureReleaseTimeoutMs: 5,
      async load() {
        return {
          engine: {
            async generate(normalized) {
              if (normalized.id === "long") {
                await releaseLong.promise;
                return { text: "", finishReason: "cancelled" };
              }
              throw new ServeError("request exceeds memory budget", {
                code: "memory_budget_exceeded",
                status: 429,
              });
            },
          },
          dispose() {},
        };
      },
    });

    const long = engine.generate(request("long", "alpha"));
    await sleep(0);
    await expectServeErrorCode(
      engine.generate(request("short", "alpha")),
      "model_pool_pressure_timeout",
    );
    releaseLong.resolve();
    await expectServeErrorCode(long, "model_pool_memory_pressure");
  });

  test("retries stream factory pressure before emitting bytes", async () => {
    const events: ServeEvent[] = [];
    let streamAttempts = 0;
    const engine = createSourceModelPoolGenerationEngine({
      entries: [{ modelId: "alpha" }],
      pressurePolicy: "shed_non_pinned",
      onEvent(event) {
        events.push(event);
      },
      async load() {
        return {
          engine: {
            async generate(normalized) {
              if (normalized.id.startsWith("long")) {
                await new Promise<void>((_, reject) => {
                  normalized.abortSignal?.addEventListener(
                    "abort",
                    () => {
                      const error = new Error("aborted");
                      error.name = "AbortError";
                      reject(error);
                    },
                    { once: true },
                  );
                });
              }
              return { text: `a:${normalized.id}`, finishReason: "stop" };
            },
            stream() {
              streamAttempts += 1;
              const attempt = streamAttempts;
              if (attempt === 2) {
                throw new ServeError("request exceeds memory budget", {
                  code: "memory_budget_exceeded",
                  status: 429,
                });
              }
              return (async function* () {
                if (attempt === 1) {
                  throw new ServeError("request exceeds memory budget", {
                    code: "memory_budget_exceeded",
                    status: 429,
                  });
                }
                yield { type: "text", text: "ok" } satisfies GenerationStreamEvent;
                yield { type: "done", finishReason: "stop" } satisfies GenerationStreamEvent;
              })();
            },
          },
          dispose() {},
        };
      },
    });

    const longA = engine.generate(request("long-a", "alpha"));
    const longAResult = expectServeErrorCode(longA, "model_pool_memory_pressure");
    await sleep(0);
    const longB = engine.generate(request("long-b", "alpha"));
    const longBResult = expectServeErrorCode(longB, "model_pool_memory_pressure");
    await sleep(0);
    const stream = await engine.stream?.(request("short", "alpha"));
    if (stream === undefined) {
      throw new Error("expected stream");
    }
    const iterator = stream[Symbol.asyncIterator]();

    await expect(iterator.next()).resolves.toEqual({
      done: false,
      value: { type: "text", text: "ok" },
    });
    await expect(iterator.next()).resolves.toEqual({
      done: false,
      value: { type: "done", finishReason: "stop" },
    });
    await expect(iterator.next()).resolves.toMatchObject({ done: true });
    await longAResult;
    await longBResult;
    expect(streamAttempts).toBe(3);
    expect(
      events
        .filter((event) => event.type === "model_pool_pressure")
        .map((event) => event.abortedRequestIds),
    ).toEqual([["long-a"], ["long-b"]]);
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
