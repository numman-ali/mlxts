import { describe, expect, test } from "bun:test";

import {
  formatPretrainedLoadProgress,
  formatServeEvent,
  formatServeReady,
  formatServeUsage,
  parseServeArgs,
  publicBindWarning,
  runServeCli,
  shouldLogServeEvent,
} from "./cli";
import type { RunningModelServer } from "./model-server";

describe("serve CLI args", () => {
  test("parses defaults from a model source", () => {
    const parsed = parseServeArgs(["mlx-community/Qwen3.6-27B-4bit"]);

    expect(parsed).toEqual({
      kind: "serve",
      options: {
        source: "mlx-community/Qwen3.6-27B-4bit",
        modelId: "mlx-community/Qwen3.6-27B-4bit",
        models: [
          {
            source: "mlx-community/Qwen3.6-27B-4bit",
            modelId: "mlx-community/Qwen3.6-27B-4bit",
          },
        ],
        hostname: "127.0.0.1",
        port: 8000,
        maxGeneratedTokens: 2048,
        maxTotalTokens: 4096,
        maxBatchSize: 32,
        batchWindowMs: 1,
        maxConcurrentRequests: 1,
        localFilesOnly: false,
        verbose: false,
      },
    });
  });

  test("parses serving, cache, auth, and safety options", () => {
    const parsed = parseServeArgs([
      "local-model",
      "--served-model-name",
      "qwen-local",
      "--host",
      "0.0.0.0",
      "--port",
      "8080",
      "--max-generated-tokens",
      "512",
      "--max-total-tokens",
      "1536",
      "--max-batch-size",
      "16",
      "--batch-window-ms",
      "4",
      "--max-concurrent-requests",
      "2",
      "--revision",
      "main",
      "--access-token",
      "hf_secret",
      "--cache-dir",
      ".cache/hf",
      "--api-key",
      "secret",
      "--local-files-only",
      "--verbose",
    ]);

    expect(parsed).toEqual({
      kind: "serve",
      options: {
        source: "local-model",
        modelId: "qwen-local",
        models: [{ source: "local-model", modelId: "qwen-local" }],
        hostname: "0.0.0.0",
        port: 8080,
        maxGeneratedTokens: 512,
        maxTotalTokens: 1536,
        maxBatchSize: 16,
        batchWindowMs: 4,
        maxConcurrentRequests: 2,
        revision: "main",
        accessToken: "hf_secret",
        cacheDir: ".cache/hf",
        apiKey: "secret",
        localFilesOnly: true,
        verbose: true,
      },
    });
  });

  test("parses repeatable model entries for multi-model serving", () => {
    const parsed = parseServeArgs([
      "--model",
      "gemma=google/gemma-4-E2B-it",
      "--model",
      "mlx-community/Qwen3.6-27B-4bit",
      "--local-files-only",
    ]);

    expect(parsed).toMatchObject({
      kind: "serve",
      options: {
        source: "google/gemma-4-E2B-it",
        modelId: "gemma",
        models: [
          { source: "google/gemma-4-E2B-it", modelId: "gemma" },
          {
            source: "mlx-community/Qwen3.6-27B-4bit",
            modelId: "mlx-community/Qwen3.6-27B-4bit",
          },
        ],
        localFilesOnly: true,
      },
    });
  });

  test("returns help for invalid or missing inputs", () => {
    expect(parseServeArgs([])).toMatchObject({
      kind: "help",
      exitCode: 1,
      message: "Missing model path or Hugging Face repo id.",
    });
    expect(parseServeArgs(["model", "--max-generated-tokens", "0"])).toMatchObject({
      kind: "help",
      exitCode: 1,
      message: 'Expected --max-generated-tokens to be a positive integer, got "0".',
    });
    expect(parseServeArgs(["model", "--max-batch-size", "0"])).toMatchObject({
      kind: "help",
      exitCode: 1,
      message: 'Expected --max-batch-size to be a positive integer, got "0".',
    });
    expect(parseServeArgs(["model", "--batch-window-ms", "-1"])).toMatchObject({
      kind: "help",
      exitCode: 1,
      message: 'Expected --batch-window-ms to be a non-negative integer, got "-1".',
    });
    expect(parseServeArgs(["model", "--max-concurrent-requests", "0"])).toMatchObject({
      kind: "help",
      exitCode: 1,
      message: 'Expected --max-concurrent-requests to be a positive integer, got "0".',
    });
    expect(parseServeArgs(["model", "--model-id"])).toMatchObject({
      kind: "help",
      message: "Missing value for --model-id.",
    });
    expect(parseServeArgs(["model", "--unknown"])).toMatchObject({
      kind: "help",
      message: "Unknown argument: --unknown",
    });
    expect(parseServeArgs(["model", "--model", "repo/model"])).toMatchObject({
      kind: "help",
      message: "Cannot mix a positional model source with --model entries.",
    });
    expect(parseServeArgs(["--model", "repo/model", "--model-id", "alias"])).toMatchObject({
      kind: "help",
      message: "Cannot use --model-id with --model; use --model <model-id=source>.",
    });
    expect(parseServeArgs(["--model", "alias="])).toMatchObject({
      kind: "help",
      message: "Expected --model to include a non-empty source.",
    });
    expect(parseServeArgs(["--model", "=repo/model"])).toMatchObject({
      kind: "help",
      message: "Expected --model to include a non-empty model id.",
    });
    expect(parseServeArgs(["--model", "same=one", "--model", "same=two"])).toMatchObject({
      kind: "help",
      message: 'model id "same" is duplicated.',
    });
    expect(parseServeArgs(["one", "two"])).toMatchObject({
      kind: "help",
      message: "Unexpected extra positional argument: two",
    });
    expect(parseServeArgs(["--help"])).toEqual({ kind: "help", exitCode: 0 });
    expect(formatServeUsage()).toContain("bunx @mlxts/serve");
  });

  test("formats progress, ready output, and public bind warnings", () => {
    expect(
      formatPretrainedLoadProgress({ stage: "resolve", status: "start", source: "repo/model" }),
    ).toBe("[resolve] resolving repo/model");
    expect(
      formatPretrainedLoadProgress({
        stage: "resolve",
        status: "complete",
        sourceKind: "hub",
        directory: "/tmp/model",
        fileCount: 2,
        totalBytes: 512,
      }),
    ).toContain("512 B");
    expect(
      formatPretrainedLoadProgress({
        stage: "resolve",
        status: "complete",
        sourceKind: "hub",
        directory: "/tmp/model",
        fileCount: 2,
        totalBytes: 1_500,
      }),
    ).toContain("1.5 KB");
    expect(
      formatPretrainedLoadProgress({
        stage: "download",
        status: "cached",
        repoId: "repo/model",
        relativePath: "config.json",
        size: 10,
        index: 1,
        totalFiles: 2,
        completedFiles: 1,
        completedBytes: 1_000_000,
        totalBytes: 1_000_000_000,
      }),
    ).toContain("1.0 MB / 1.0 GB");
    expect(
      formatPretrainedLoadProgress({ stage: "model", status: "weights-start", shardCount: 3 }),
    ).toContain("loading 3");
    expect(
      formatPretrainedLoadProgress({ stage: "model", status: "weights-complete", shardCount: 3 }),
    ).toContain("loaded 3");
    expect(
      formatPretrainedLoadProgress({ stage: "tokenizer", status: "start", directory: "/tmp" }),
    ).toBe("[tokenizer] loading from /tmp");
    expect(
      formatPretrainedLoadProgress({ stage: "tokenizer", status: "complete", directory: "/tmp" }),
    ).toBe("[tokenizer] ready");

    const options = {
      source: "repo/model",
      modelId: "qwen-local",
      models: [{ source: "repo/model", modelId: "qwen-local" }],
      hostname: "0.0.0.0",
      port: 8000,
      maxGeneratedTokens: 64,
      maxTotalTokens: 256,
      maxBatchSize: 8,
      batchWindowMs: 2,
      maxConcurrentRequests: 1,
      localFilesOnly: false,
      verbose: false,
    };
    expect(formatServeReady("http://127.0.0.1:8000", options)).toContain("/v1/completions");
    expect(formatServeReady("http://127.0.0.1:8000", options)).toContain("Models: qwen-local");
    expect(formatServeReady("http://127.0.0.1:8000", options)).not.toContain("temperature");
    expect(formatServeReady("http://127.0.0.1:8000", options)).toContain(
      "Micro-batching: max_batch=8 window_ms=2",
    );
    expect(formatServeReady("http://127.0.0.1:8000", options)).toContain(
      "Admission concurrency: max_in_flight=1",
    );
    expect(publicBindWarning(options)).toContain("exposes the endpoint");
    expect(publicBindWarning({ ...options, apiKey: "secret" })).toBeNull();
    expect(
      formatServeReady("http://127.0.0.1:8000", {
        ...options,
        models: [
          { source: "repo/gemma", modelId: "gemma" },
          { source: "repo/qwen", modelId: "qwen" },
        ],
      }),
    ).toContain("Serving 2 models");
    expect(
      formatServeEvent({
        type: "generation_start",
        id: "cmpl-test",
        protocol: "openai.chat_completions",
        model: "qwen-local",
        inputKind: "messages",
        maxTokens: 32,
      }),
    ).toBe(
      "[generation] cmpl-test openai.chat_completions model=qwen-local input=messages max_tokens=32 started",
    );
    expect(
      formatServeEvent({
        type: "generation_progress",
        id: "cmpl-test",
        protocol: "openai.chat_completions",
        model: "qwen-local",
        promptTokens: 12,
        completionTokens: 64,
        maxTokens: 128,
        memory: {
          activeBytes: 1_000_000,
          cacheBytes: 2_000_000,
          peakBytes: 3_000_000,
          limitBytes: 4_000_000,
        },
      }),
    ).toBe(
      "[generation] cmpl-test progress prompt_tokens=12 completion_tokens=64/128 active=1.0 MB cache=2.0 MB peak=3.0 MB",
    );
    expect(
      formatServeEvent({
        type: "generation_batch_start",
        mode: "static",
        model: "qwen-local",
        ids: ["cmpl-a", "cmpl-b"],
        batchSize: 2,
        maxTokens: 64,
        maxTokensByRequest: [32, 64],
      }),
    ).toBe(
      "[batch] static model=qwen-local size=2 max_tokens=64 per_request=32,64 ids=cmpl-a,cmpl-b started",
    );
    expect(
      formatServeEvent({
        type: "generation_complete",
        id: "cmpl-test",
        protocol: "openai.chat_completions",
        model: "qwen-local",
        finishReason: "eos",
        promptTokens: 12,
        completionTokens: 47,
        totalTokens: 59,
        durationMs: 1_234,
      }),
    ).toBe("[generation] cmpl-test eos prompt_tokens=12 tokens=47 total_tokens=59 in 1.2s");
    expect(
      formatServeEvent({
        type: "generation_error",
        id: "cmpl-test",
        protocol: "openai.chat_completions",
        model: "qwen-local",
        code: "model_not_found",
        message: 'Model "missing" is not served by this endpoint.',
        durationMs: 321,
      }),
    ).toBe(
      '[generation:error] cmpl-test openai.chat_completions model=qwen-local model_not_found: Model "missing" is not served by this endpoint. in 321.0ms',
    );
    expect(
      shouldLogServeEvent(
        { type: "request_start", method: "POST", path: "/v1/chat/completions" },
        false,
      ),
    ).toBe(false);
    expect(
      shouldLogServeEvent(
        {
          type: "generation_batch_start",
          mode: "static",
          model: "qwen-local",
          ids: ["cmpl-a", "cmpl-b"],
          batchSize: 2,
          maxTokens: 64,
          maxTokensByRequest: [64, 64],
        },
        false,
      ),
    ).toBe(true);
    expect(
      shouldLogServeEvent(
        {
          type: "generation_error",
          id: "cmpl-test",
          protocol: "openai.chat_completions",
          model: "qwen-local",
          code: "internal_error",
          message: "boom",
          durationMs: 1,
        },
        false,
      ),
    ).toBe(true);
    expect(
      shouldLogServeEvent(
        { type: "request_start", method: "POST", path: "/v1/chat/completions" },
        true,
      ),
    ).toBe(true);
  });

  test("runs the serve CLI with generation logs by default", async () => {
    const logs: string[] = [];
    const errors: string[] = [];
    const running = fakeRunningServer();
    await runServeCli(["repo/model"], {
      async serveModel(options) {
        options.onEvent?.({ type: "request_start", method: "POST", path: "/v1/chat/completions" });
        options.onEvent?.({
          type: "generation_start",
          id: "cmpl-test",
          protocol: "openai.chat_completions",
          model: "repo/model",
          inputKind: "messages",
          maxTokens: 32,
        });
        return running;
      },
      log(message) {
        logs.push(message);
      },
      error(message) {
        errors.push(message);
      },
      async waitForShutdown(server) {
        server.stop();
      },
    });

    const output = logs.join("\n");
    expect(errors).toEqual([]);
    expect(output).not.toContain("[request]");
    expect(output).toContain("[generation] cmpl-test");
    expect(running.stopped).toBe(true);
  });

  test("runs the serve CLI with injectable runtime", async () => {
    const logs: string[] = [];
    const errors: string[] = [];
    const running = fakeRunningServer();
    await runServeCli(["repo/model", "--host", "0.0.0.0", "--verbose"], {
      async serveModel(options) {
        options.onProgress?.({ stage: "tokenizer", status: "complete", directory: "/tmp" });
        options.onEvent?.({ type: "request_start", method: "POST", path: "/v1/chat/completions" });
        expect(options.source).toBe("repo/model");
        expect(options.hostname).toBe("0.0.0.0");
        expect(options.maxBatchSize).toBe(32);
        expect(options.batchWindowMs).toBe(1);
        expect(options.maxConcurrentRequests).toBe(1);
        return running;
      },
      log(message) {
        logs.push(message);
      },
      error(message) {
        errors.push(message);
      },
      async waitForShutdown(server) {
        server.stop();
      },
    });

    expect(errors).toEqual([]);
    expect(logs.join("\n")).toContain("[tokenizer] ready");
    expect(logs.join("\n")).toContain("[request] POST /v1/chat/completions started");
    expect(logs.join("\n")).toContain("without --api-key");
    expect(running.stopped).toBe(true);
  });

  test("runs the serve CLI with repeatable model entries", async () => {
    const logs: string[] = [];
    const errors: string[] = [];
    const running = fakeRunningServer(["gemma", "qwen"]);
    await runServeCli(["--model", "gemma=repo/gemma", "--model", "qwen=repo/qwen"], {
      async serveModel() {
        throw new Error("single-model server should not be used for repeatable --model entries");
      },
      async serveModels(options) {
        options.onProgress?.(
          { stage: "resolve", status: "start", source: "repo/gemma" },
          { index: 0, source: "repo/gemma", modelId: "gemma" },
        );
        expect(options.models).toEqual([
          { source: "repo/gemma", modelId: "gemma" },
          { source: "repo/qwen", modelId: "qwen" },
        ]);
        return running;
      },
      log(message) {
        logs.push(message);
      },
      error(message) {
        errors.push(message);
      },
      async waitForShutdown(server) {
        server.stop();
      },
    });

    expect(errors).toEqual([]);
    expect(logs.join("\n")).toContain("[load 1/2 gemma] [resolve] resolving repo/gemma");
    expect(logs.join("\n")).toContain("Models: gemma, qwen");
    expect(running.stopped).toBe(true);
  });

  test("runs help through injectable exit", async () => {
    const errors: string[] = [];
    let exitCode = -1;

    await runServeCli([], {
      error(message) {
        errors.push(message);
      },
      exit(code) {
        exitCode = code;
      },
    });

    expect(exitCode).toBe(1);
    expect(errors.join("\n")).toContain("Missing model path");
    expect(errors.join("\n")).toContain("mlxts-serve");
  });
});

function fakeRunningServer(modelIds: readonly string[] = ["qwen-local"]): RunningModelServer & {
  stopped: boolean;
} {
  const server = Bun.serve({
    port: 0,
    fetch() {
      return new Response("ok");
    },
  });
  return {
    endpoint: "http://127.0.0.1:8000",
    modelId: modelIds[0] ?? "qwen-local",
    modelIds,
    server,
    stopped: false,
    stop() {
      server.stop(true);
      this.stopped = true;
    },
    [Symbol.dispose]() {
      server.stop(true);
      this.stopped = true;
    },
  };
}
