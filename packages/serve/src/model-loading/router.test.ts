import { describe, expect, test } from "bun:test";
import type {
  GenerationEngine,
  GenerationStreamEvent,
  NormalizedGenerationRequest,
} from "../types";
import { createModelRouterGenerationEngine } from "./router";

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

function taggedEngine(tag: string, batchIds: string[][]): GenerationEngine {
  return {
    generate(normalized) {
      return { text: `${tag}:${normalized.id}`, finishReason: "stop" };
    },
    generateBatch(requests) {
      batchIds.push(requests.map((entry) => entry.id));
      return requests.map((entry) => ({ text: `${tag}:${entry.id}`, finishReason: "stop" }));
    },
  };
}

describe("model router generation engine", () => {
  test("requires at least one model engine and routes single requests", async () => {
    expect(() => createModelRouterGenerationEngine({ engines: {} })).toThrow("at least one engine");

    const engine = createModelRouterGenerationEngine({
      engines: { alpha: taggedEngine("a", []) },
    });
    const result = await engine.generate(request("1", "alpha"));

    expect(result.text).toBe("a:1");
  });

  test("routes batched requests by model while preserving original order", async () => {
    const alphaBatches: string[][] = [];
    const betaBatches: string[][] = [];
    const engine = createModelRouterGenerationEngine({
      engines: {
        alpha: taggedEngine("a", alphaBatches),
        beta: taggedEngine("b", betaBatches),
      },
    });

    const results = await engine.generateBatch?.([
      request("1", "beta"),
      request("2", "alpha"),
      request("3", "beta"),
    ]);

    expect(results?.map((result) => result.text)).toEqual(["b:1", "a:2", "b:3"]);
    expect(alphaBatches).toEqual([]);
    expect(betaBatches).toEqual([["1", "3"]]);
  });

  test("disposes routed model engines", () => {
    let alphaDisposeCount = 0;
    let betaDisposeCount = 0;
    const engine = createModelRouterGenerationEngine({
      engines: {
        alpha: {
          generate() {
            return { text: "a", finishReason: "stop" };
          },
          [Symbol.dispose]() {
            alphaDisposeCount += 1;
          },
        },
        beta: {
          generate() {
            return { text: "b", finishReason: "stop" };
          },
          [Symbol.dispose]() {
            betaDisposeCount += 1;
          },
        },
      },
    });

    engine[Symbol.dispose]?.();

    expect(alphaDisposeCount).toBe(1);
    expect(betaDisposeCount).toBe(1);
  });

  test("aggregates prompt-prefix cache retention info from routed engines", () => {
    const engine = createModelRouterGenerationEngine({
      engines: {
        alpha: {
          generate() {
            return { text: "a", finishReason: "stop" };
          },
          promptPrefixCacheInfo() {
            return {
              models: [
                {
                  id: "alpha",
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
        },
        beta: {
          generate() {
            return { text: "b", finishReason: "stop" };
          },
          promptPrefixCacheInfo() {
            return {
              models: [
                {
                  id: "beta",
                  retainedSnapshots: 2,
                  retainedSnapshotBytes: 2048,
                  indexedBlockHashes: 3,
                  tokenBlocks: {
                    blockSize: 32,
                    blockCount: 3,
                    blockReferences: 5,
                    uniqueTokenCount: 96,
                    referencedTokenCount: 160,
                  },
                },
              ],
              totalRetainedSnapshots: 2,
              totalRetainedSnapshotBytes: 2048,
              totalIndexedBlockHashes: 3,
              totalTokenBlocks: 3,
              totalTokenBlockReferences: 5,
              totalUniqueTokenCount: 96,
              totalReferencedTokenCount: 160,
            };
          },
        },
      },
    });

    expect(engine.promptPrefixCacheInfo?.()).toMatchObject({
      totalRetainedSnapshots: 3,
      totalRetainedSnapshotBytes: 3072,
      totalIndexedBlockHashes: 5,
      totalTokenBlocks: 5,
      totalTokenBlockReferences: 7,
      totalUniqueTokenCount: 224,
      totalReferencedTokenCount: 288,
      models: [{ id: "alpha" }, { id: "beta" }],
    });
  });

  test("reports unknown model ids and delegates streaming per model", async () => {
    const engine = createModelRouterGenerationEngine({
      engines: new Map<string, GenerationEngine>([
        [
          "streaming",
          {
            generate() {
              return { text: "", finishReason: "stop" };
            },
            async *stream() {
              yield { type: "text", text: "hi" };
              yield { type: "done", finishReason: "stop" };
            },
          },
        ],
      ]),
    });

    expect(() => engine.generate(request("missing", "missing"))).toThrow(
      "not served by this endpoint",
    );

    const stream = await engine.stream?.(request("stream", "streaming"));
    const events: GenerationStreamEvent[] = [];
    if (stream !== undefined) {
      for await (const event of stream) {
        events.push(event);
      }
    }
    expect(events).toEqual([
      { type: "text", text: "hi" },
      { type: "done", finishReason: "stop" },
    ]);

    expect(() => {
      void engine.stream?.(request("stream", "missing"));
    }).toThrow("not served by this endpoint");
  });

  test("reports unsupported streaming and malformed inner batch results", async () => {
    const noStream = createModelRouterGenerationEngine({
      engines: {
        alpha: {
          generate() {
            return { text: "", finishReason: "stop" };
          },
        },
      },
    });

    expect(() => {
      void noStream.stream?.(request("stream", "alpha"));
    }).toThrow("does not support streaming");

    const wrongSize = createModelRouterGenerationEngine({
      engines: {
        alpha: {
          generate() {
            return { text: "", finishReason: "stop" };
          },
          generateBatch() {
            return [];
          },
        },
      },
    });
    const generateBatch = wrongSize.generateBatch;
    if (generateBatch === undefined) {
      throw new Error("router should expose generateBatch");
    }

    await expect(generateBatch([request("1", "alpha"), request("2", "alpha")])).rejects.toThrow(
      "wrong number of batch results",
    );

    const incomplete = createModelRouterGenerationEngine({
      engines: {
        alpha: {
          generate() {
            return { text: "", finishReason: "stop" };
          },
          generateBatch() {
            return [
              { text: "ok", finishReason: "stop" },
              undefined as unknown as { text: string; finishReason: "stop" },
            ];
          },
        },
      },
    });
    const incompleteBatch = incomplete.generateBatch;
    if (incompleteBatch === undefined) {
      throw new Error("router should expose generateBatch");
    }
    await expect(incompleteBatch([request("1", "alpha"), request("2", "alpha")])).rejects.toThrow(
      "did not receive a result",
    );
  });
});
