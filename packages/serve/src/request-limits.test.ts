import { describe, expect, test } from "bun:test";

import { createRequestLimitGenerationEngine } from "./request-limits";
import type { GenerationEngine, NormalizedGenerationRequest } from "./types";

function normalized(maxTokens: number): NormalizedGenerationRequest {
  return {
    id: `request-${maxTokens}`,
    model: "tiny",
    input: { kind: "text", text: "hello" },
    sampling: { maxTokens },
    stream: false,
    protocol: "openai.completions",
  };
}

function expectMaxTokensError(fn: () => unknown): void {
  try {
    fn();
  } catch (error) {
    expect(error).toMatchObject({
      code: "max_tokens_exceeded",
      param: "max_tokens",
    });
    return;
  }
  throw new Error("Expected request limit error.");
}

describe("request limit generation engine", () => {
  test("rejects invalid limit options", () => {
    expect(() =>
      createRequestLimitGenerationEngine({
        engine: {
          generate() {
            return { text: "", finishReason: "stop" };
          },
        },
        maxGeneratedTokens: 0,
      }),
    ).toThrow("positive integer");
  });

  test("rejects single requests over the generated token limit", () => {
    const engine = createRequestLimitGenerationEngine({
      engine: {
        generate() {
          throw new Error("inner engine should not run");
        },
      },
      maxGeneratedTokens: 4,
    });

    expectMaxTokensError(() => {
      void engine.generate(normalized(5));
    });
  });

  test("disposes the wrapped engine when present", () => {
    let disposeCount = 0;
    const engine = createRequestLimitGenerationEngine({
      engine: {
        generate() {
          return { text: "", finishReason: "stop" };
        },
        [Symbol.dispose]() {
          disposeCount += 1;
        },
      },
      maxGeneratedTokens: 4,
    });

    engine[Symbol.dispose]?.();

    expect(disposeCount).toBe(1);
  });

  test("validates a full batch before generation starts", () => {
    const seen: string[] = [];
    const inner: GenerationEngine = {
      generate(request) {
        seen.push(request.id);
        return { text: request.id, finishReason: "stop" };
      },
    };
    const engine = createRequestLimitGenerationEngine({
      engine: inner,
      maxGeneratedTokens: 3,
    });

    const generateBatch = engine.generateBatch;
    if (generateBatch === undefined) {
      throw new Error("request limit wrapper should expose generateBatch");
    }
    expectMaxTokensError(() => {
      void generateBatch([normalized(2), normalized(4)]);
    });
    expect(seen).toEqual([]);
  });

  test("delegates accepted batched and streaming requests", async () => {
    const inner: GenerationEngine = {
      generate() {
        throw new Error("single generate should not run");
      },
      generateBatch(requests) {
        return requests.map((request) => ({
          text: `${request.id}:${request.sampling.maxTokens}`,
          finishReason: "stop",
        }));
      },
      async *stream(request) {
        yield { type: "text", text: request.id };
        yield { type: "done", finishReason: "stop" };
      },
    };
    const engine = createRequestLimitGenerationEngine({
      engine: inner,
      maxGeneratedTokens: 4,
    });

    const batch = await engine.generateBatch?.([normalized(1), normalized(4)]);
    expect(batch?.map((result) => result.text)).toEqual(["request-1:1", "request-4:4"]);

    const stream = await engine.stream?.(normalized(2));
    const chunks: string[] = [];
    if (stream !== undefined) {
      for await (const event of stream) {
        if (event.type === "text") {
          chunks.push(event.text);
        }
      }
    }
    expect(chunks).toEqual(["request-2"]);
  });

  test("falls back to sequential batch generation and checks batch result sizes", async () => {
    const seen: string[] = [];
    const fallback = createRequestLimitGenerationEngine({
      engine: {
        generate(request) {
          seen.push(request.id);
          return { text: request.id, finishReason: "stop" };
        },
      },
      maxGeneratedTokens: 4,
    });

    const fallbackBatch = fallback.generateBatch;
    if (fallbackBatch === undefined) {
      throw new Error("request limit wrapper should expose generateBatch");
    }
    await expect(fallbackBatch([normalized(1), normalized(2)])).resolves.toEqual([
      { text: "request-1", finishReason: "stop" },
      { text: "request-2", finishReason: "stop" },
    ]);
    expect(seen).toEqual(["request-1", "request-2"]);

    const wrongSize = createRequestLimitGenerationEngine({
      engine: {
        generate() {
          throw new Error("single generate should not run");
        },
        generateBatch() {
          return [];
        },
      },
      maxGeneratedTokens: 4,
    });
    const wrongSizeBatch = wrongSize.generateBatch;
    if (wrongSizeBatch === undefined) {
      throw new Error("request limit wrapper should expose generateBatch");
    }
    await expect(wrongSizeBatch([normalized(1)])).rejects.toThrow("wrong number");
  });

  test("rejects streaming requests over the generated token limit", () => {
    const engine = createRequestLimitGenerationEngine({
      engine: {
        generate() {
          return { text: "", finishReason: "stop" };
        },
        async *stream() {
          yield { type: "done" as const, finishReason: "stop" as const };
        },
      },
      maxGeneratedTokens: 1,
    });

    expectMaxTokensError(() => {
      void engine.stream?.(normalized(2));
    });
  });
});
