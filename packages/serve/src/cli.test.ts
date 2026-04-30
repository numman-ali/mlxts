import { describe, expect, test } from "bun:test";

import {
  formatPretrainedLoadProgress,
  formatServeEvent,
  formatServeReady,
  formatServeUsage,
  parseServeArgs,
  publicBindWarning,
  runServeCli,
  type ServeCliOptions,
  shouldLogServeEvent,
} from "./cli";
import type { RunningModelServer } from "./model-loading/server";

const ROUTE_STRATEGY = {
  schedulerMode: "auto",
  cacheBackend: "managed",
  attentionBackend: "auto",
  decodingBackend: "model",
} as const;

const SCHEDULER_TOKENS = {
  waitingTotalTokens: 0,
  prefillingTotalTokens: 0,
  activeTotalTokens: 128,
  scheduledPromptTokens: 96,
  maxScheduledPromptTokens: 384,
  scheduledCompletionTokens: 32,
  maxScheduledCompletionTokens: 128,
  scheduledTotalTokens: 128,
  maxScheduledTotalTokens: 512,
  scheduledMemoryBytes: 1024,
  maxScheduledMemoryBytes: 4096,
};

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
        modelRoots: [],
        modelLoadPolicy: "eager",
        pinnedModels: [],
        hostname: "127.0.0.1",
        port: 8000,
        maxGeneratedTokens: 2048,
        maxPromptTokens: 4096,
        maxTotalTokens: 4096,
        maxBatchSize: 32,
        batchWindowMs: 1,
        prefillStepSize: 512,
        activePrefillStepSize: 128,
        activeDecodeStepsPerPrefillChunk: 16,
        streamDecodeInterval: 1,
        maxConcurrentRequests: 1,
        promptPrefixCacheMaxEntries: 1,
        remoteImageHosts: [],
        gpuMemoryUtilization: 0.9,
        localFilesOnly: false,
        verbose: false,
      },
    });
  });

  test("parses serving, cache, auth, and safety options", () => {
    const parsed = parseServeArgs([
      "local-model",
      "--served-model-name",
      "mlx-community/Qwen3.6-27B-4bit",
      "--host",
      "0.0.0.0",
      "--port",
      "8080",
      "--max-generated-tokens",
      "512",
      "--max-prompt-tokens",
      "1024",
      "--max-total-tokens",
      "1536",
      "--max-batch-size",
      "16",
      "--batch-window-ms",
      "4",
      "--prefill-step-size",
      "1024",
      "--active-prefill-step-size",
      "256",
      "--active-decode-steps-per-prefill-chunk",
      "24",
      "--stream-decode-interval",
      "2",
      "--max-concurrent-requests",
      "2",
      "--prompt-prefix-cache-max-entries",
      "3",
      "--prompt-prefix-cache-max-bytes",
      "1048576",
      "--gpu-memory-utilization",
      "0.75",
      "--revision",
      "main",
      "--access-token",
      "hf_secret",
      "--cache-dir",
      ".cache/hf",
      "--api-key",
      "secret",
      "--remote-image-host",
      "EXAMPLE.com",
      "--remote-image-host",
      "cdn.example.com",
      "--local-files-only",
      "--verbose",
    ]);

    expect(parsed).toEqual({
      kind: "serve",
      options: {
        source: "local-model",
        modelId: "mlx-community/Qwen3.6-27B-4bit",
        models: [{ source: "local-model", modelId: "mlx-community/Qwen3.6-27B-4bit" }],
        modelRoots: [],
        modelLoadPolicy: "eager",
        pinnedModels: [],
        hostname: "0.0.0.0",
        port: 8080,
        maxGeneratedTokens: 512,
        maxPromptTokens: 1024,
        maxTotalTokens: 1536,
        maxBatchSize: 16,
        batchWindowMs: 4,
        prefillStepSize: 1024,
        activePrefillStepSize: 256,
        activeDecodeStepsPerPrefillChunk: 24,
        streamDecodeInterval: 2,
        maxConcurrentRequests: 2,
        promptPrefixCacheMaxEntries: 3,
        promptPrefixCacheMaxBytes: 1048576,
        gpuMemoryUtilization: 0.75,
        revision: "main",
        accessToken: "hf_secret",
        cacheDir: ".cache/hf",
        apiKey: "secret",
        remoteImageHosts: ["example.com", "cdn.example.com"],
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
      "--model-load-policy",
      "lazy",
      "--model-idle-ttl-ms",
      "60000",
      "--pin-model",
      "gemma",
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
        modelRoots: [],
        modelLoadPolicy: "lazy",
        modelIdleTtlMs: 60000,
        pinnedModels: ["gemma"],
        localFilesOnly: true,
      },
    });
  });

  test("parses local model-root discovery for lazy multi-model serving", () => {
    expect(parseServeArgs(["--model-root", "/models"])).toMatchObject({
      kind: "serve",
      options: {
        source: "/models",
        modelId: "/models",
        models: [],
        modelRoots: ["/models"],
        modelLoadPolicy: "lazy",
        pinnedModels: [],
      },
    });

    expect(
      parseServeArgs([
        "--model",
        "manual=repo/manual",
        "--model-root",
        "/models",
        "--model-load-policy",
        "lazy",
        "--pin-model",
        "org/qwen",
      ]),
    ).toMatchObject({
      kind: "serve",
      options: {
        source: "repo/manual",
        modelId: "manual",
        models: [{ source: "repo/manual", modelId: "manual" }],
        modelRoots: ["/models"],
        modelLoadPolicy: "lazy",
        pinnedModels: ["org/qwen"],
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
    expect(parseServeArgs(["model", "--max-prompt-tokens", "0"])).toMatchObject({
      kind: "help",
      exitCode: 1,
      message: 'Expected --max-prompt-tokens to be a positive integer, got "0".',
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
    expect(parseServeArgs(["model", "--prefill-step-size", "0"])).toMatchObject({
      kind: "help",
      exitCode: 1,
      message: 'Expected --prefill-step-size to be a positive integer, got "0".',
    });
    expect(parseServeArgs(["model", "--active-prefill-step-size", "0"])).toMatchObject({
      kind: "help",
      exitCode: 1,
      message: 'Expected --active-prefill-step-size to be a positive integer, got "0".',
    });
    expect(parseServeArgs(["model", "--active-decode-steps-per-prefill-chunk", "0"])).toMatchObject(
      {
        kind: "help",
        exitCode: 1,
        message:
          'Expected --active-decode-steps-per-prefill-chunk to be a positive integer, got "0".',
      },
    );
    expect(parseServeArgs(["model", "--stream-decode-interval", "0"])).toMatchObject({
      kind: "help",
      exitCode: 1,
      message: 'Expected --stream-decode-interval to be a positive integer, got "0".',
    });
    expect(parseServeArgs(["model", "--max-concurrent-requests", "0"])).toMatchObject({
      kind: "help",
      exitCode: 1,
      message: 'Expected --max-concurrent-requests to be a positive integer, got "0".',
    });
    expect(parseServeArgs(["model", "--prompt-prefix-cache-max-entries", "0"])).toMatchObject({
      kind: "help",
      exitCode: 1,
      message: 'Expected --prompt-prefix-cache-max-entries to be a positive integer, got "0".',
    });
    expect(parseServeArgs(["model", "--prompt-prefix-cache-max-bytes", "0"])).toMatchObject({
      kind: "help",
      exitCode: 1,
      message: 'Expected --prompt-prefix-cache-max-bytes to be a positive integer, got "0".',
    });
    expect(parseServeArgs(["model", "--gpu-memory-utilization", "1.5"])).toMatchObject({
      kind: "help",
      exitCode: 1,
      message:
        'Expected --gpu-memory-utilization to be a number greater than 0 and less than or equal to 1, got "1.5".',
    });
    expect(parseServeArgs(["model", "--model-load-policy", "unknown"])).toMatchObject({
      kind: "help",
      message: "Unknown model load policy: unknown.",
    });
    expect(parseServeArgs(["model", "--model-idle-ttl-ms", "0"])).toMatchObject({
      kind: "help",
      message: 'Expected --model-idle-ttl-ms to be a positive integer, got "0".',
    });
    expect(parseServeArgs(["model", "--model-idle-ttl-ms", "1000"])).toMatchObject({
      kind: "help",
      message: "--model-idle-ttl-ms requires --model-load-policy lazy.",
    });
    expect(
      parseServeArgs(["--model-root", "/models", "--model-load-policy", "eager"]),
    ).toMatchObject({
      kind: "help",
      message: "--model-root requires --model-load-policy lazy.",
    });
    expect(parseServeArgs(["model", "--model-root", "/models"])).toMatchObject({
      kind: "help",
      message: "Cannot mix a positional model source with --model-root.",
    });
    expect(parseServeArgs(["--model-root", "/models", "--model-id", "alias"])).toMatchObject({
      kind: "help",
      message:
        "Cannot use --model-id with --model-root; use --model <model-id=source> for explicit additions.",
    });
    expect(parseServeArgs(["model", "--pin-model", "model"])).toMatchObject({
      kind: "help",
      message: "--pin-model requires --model-load-policy lazy.",
    });
    expect(
      parseServeArgs([
        "--model",
        "alpha=repo/alpha",
        "--model-load-policy",
        "lazy",
        "--pin-model",
        "missing",
      ]),
    ).toMatchObject({
      kind: "help",
      message: 'Pinned model "missing" is not part of this serve command.',
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

    const options: ServeCliOptions = {
      source: "repo/model",
      modelId: "mlx-community/Qwen3.6-27B-4bit",
      models: [{ source: "repo/model", modelId: "mlx-community/Qwen3.6-27B-4bit" }],
      modelRoots: [],
      hostname: "0.0.0.0",
      port: 8000,
      maxGeneratedTokens: 64,
      maxPromptTokens: 128,
      maxTotalTokens: 256,
      maxBatchSize: 8,
      batchWindowMs: 2,
      prefillStepSize: 512,
      activePrefillStepSize: 128,
      activeDecodeStepsPerPrefillChunk: 16,
      streamDecodeInterval: 1,
      maxConcurrentRequests: 1,
      promptPrefixCacheMaxEntries: 1,
      modelLoadPolicy: "eager",
      pinnedModels: [],
      remoteImageHosts: [],
      gpuMemoryUtilization: 0.75,
      localFilesOnly: false,
      verbose: false,
    };
    expect(formatServeReady("http://127.0.0.1:8000", options)).toContain("/v1/completions");
    expect(formatServeReady("http://127.0.0.1:8000", options)).toContain(
      "Models: mlx-community/Qwen3.6-27B-4bit",
    );
    expect(formatServeReady("http://127.0.0.1:8000", options)).not.toContain("temperature");
    expect(formatServeReady("http://127.0.0.1:8000", options)).toContain("Prompt-token limit: 128");
    expect(formatServeReady("http://127.0.0.1:8000", options)).toContain(
      "Batch scheduler: max_batch=8 window_ms=2 prefill=512 active_prefill=128 active_decode_quantum=16",
    );
    expect(formatServeReady("http://127.0.0.1:8000", options)).toContain(
      "Streaming decode interval: 1 token(s)",
    );
    expect(formatServeReady("http://127.0.0.1:8000", options)).toContain(
      "Model execution lanes: max_in_flight=1",
    );
    expect(formatServeReady("http://127.0.0.1:8000", options)).toContain(
      "Prompt-prefix cache entries: 1",
    );
    expect(formatServeReady("http://127.0.0.1:8000", options)).toContain(
      "Prompt-prefix cache bytes: unbounded",
    );
    expect(formatServeReady("http://127.0.0.1:8000", options)).toContain(
      "Remote image hosts: disabled",
    );
    expect(
      formatServeReady("http://127.0.0.1:8000", {
        ...options,
        remoteImageHosts: ["example.com", "cdn.example.com"],
      }),
    ).toContain("Remote image hosts: example.com, cdn.example.com");
    expect(
      formatServeReady("http://127.0.0.1:8000", {
        ...options,
        promptPrefixCacheMaxBytes: 1_048_576,
      }),
    ).toContain("Prompt-prefix cache bytes: 1.0 MB");
    expect(formatServeReady("http://127.0.0.1:8000", options)).toContain("GPU memory budget: 75%");
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
        model: "mlx-community/Qwen3.6-27B-4bit",
        inputKind: "messages",
        maxTokens: 32,
        stream: false,
      }),
    ).toBe(
      "[generation] cmpl-test openai.chat_completions model=mlx-community/Qwen3.6-27B-4bit input=messages max_tokens=32 started",
    );
    expect(
      formatServeEvent({
        type: "generation_progress",
        id: "cmpl-test",
        protocol: "openai.chat_completions",
        model: "mlx-community/Qwen3.6-27B-4bit",
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
        type: "generation_route_decision",
        id: "cmpl-test",
        protocol: "openai.chat_completions",
        model: "mlx-community/Qwen3.6-27B-4bit",
        route: "single",
        eligible: false,
        reason: "unsupported_model_type",
        modelType: "qwen3_5_text",
        maxBatchSize: 32,
        ...ROUTE_STRATEGY,
        stream: false,
      }),
    ).toBe(
      "[route] cmpl-test model=mlx-community/Qwen3.6-27B-4bit route=single eligible=no reason=unsupported_model_type model_type=qwen3_5_text scheduler=auto cache=managed attention=auto decoding=model max_batch_size=32",
    );
    expect(
      formatServeEvent({
        type: "generation_prompt_prepare",
        phase: "start",
        id: "cmpl-test",
        protocol: "openai.chat_completions",
        model: "mlx-community/Qwen3.6-27B-4bit",
        inputKind: "messages",
      }),
    ).toBe("[prompt] cmpl-test model=mlx-community/Qwen3.6-27B-4bit input=messages preparing");
    expect(
      formatServeEvent({
        type: "generation_prompt_prepare",
        phase: "complete",
        id: "cmpl-test",
        protocol: "openai.chat_completions",
        model: "mlx-community/Qwen3.6-27B-4bit",
        inputKind: "messages",
        promptTokens: 4096,
        durationMs: 1234,
      }),
    ).toBe(
      "[prompt] cmpl-test model=mlx-community/Qwen3.6-27B-4bit input=messages prompt_tokens=4096 ready in 1.2s",
    );
    expect(
      formatServeEvent({
        type: "generation_model_lane_wait",
        id: "cmpl-test",
        protocol: "openai.chat_completions",
        model: "mlx-community/Qwen3.6-27B-4bit",
        lane: "model",
        waitMs: 1_234,
        inFlightAtQueue: 1,
        queuedAhead: 2,
        inFlightAtDispatch: 1,
        queuedAtDispatch: 0,
        maxConcurrentJobs: 1,
      }),
    ).toBe(
      "[queue] cmpl-test model=mlx-community/Qwen3.6-27B-4bit lane=model wait=1.2s queued_ahead=2 in_flight_at_queue=1 in_flight_at_dispatch=1 max_in_flight=1",
    );
    expect(
      formatServeEvent({
        type: "generation_prefill_progress",
        id: "cmpl-test",
        protocol: "openai.chat_completions",
        model: "mlx-community/Qwen3.6-27B-4bit",
        promptTokens: 4096,
        processedPrefillTokens: 2048,
        totalPrefillTokens: 4095,
        chunkTokens: 2048,
        maxTokens: 128,
        memory: {
          activeBytes: 1_000_000,
          cacheBytes: 2_000_000,
          peakBytes: 3_000_000,
          limitBytes: 4_000_000,
        },
      }),
    ).toBe(
      "[generation] cmpl-test prefill prompt_tokens=4096 prefill_tokens=2048/4095 chunk_tokens=2048 active=1.0 MB cache=2.0 MB peak=3.0 MB",
    );
    expect(
      formatServeEvent({
        type: "generation_batch_start",
        mode: "static",
        model: "mlx-community/Qwen3.6-27B-4bit",
        ids: ["cmpl-a", "cmpl-b"],
        batchSize: 2,
        maxTokens: 64,
        maxTokensByRequest: [32, 64],
      }),
    ).toBe(
      "[batch] static model=mlx-community/Qwen3.6-27B-4bit size=2 max_tokens=64 per_request=32,64 ids=cmpl-a,cmpl-b started",
    );
    expect(
      formatServeEvent({
        type: "generation_scheduler_phase",
        mode: "continuous",
        phase: "prefill_start",
        model: "mlx-community/Qwen3.6-27B-4bit",
        id: "cmpl-a",
        ids: ["cmpl-a"],
        promptTokens: 4096,
        maxTokens: 64,
        queuedMs: 2,
        schedulerMs: 2,
        waiting: 0,
        prefilling: 1,
        active: 1,
        maxBatchSize: 8,
        ...SCHEDULER_TOKENS,
      }),
    ).toBe(
      "[scheduler] continuous cmpl-a prefill_start queued=2.0ms prompt_tokens=4096 max_tokens=64 waiting=0 prefilling=1 active=1/8 scheduled_tokens=128/512 scheduled_prompt_tokens=96/384 scheduled_completion_tokens=32/128 scheduled_memory=1.0 KB/4.1 KB",
    );
    expect(
      formatServeEvent({
        type: "generation_scheduler_phase",
        mode: "continuous",
        phase: "deferred",
        model: "mlx-community/Qwen3.6-27B-4bit",
        id: "cmpl-a",
        ids: ["cmpl-a"],
        reason: "scheduled_token_budget",
        promptTokens: 4096,
        maxTokens: 64,
        queuedMs: 3,
        schedulerMs: 3,
        waiting: 1,
        prefilling: 0,
        active: 1,
        maxBatchSize: 8,
        ...SCHEDULER_TOKENS,
      }),
    ).toBe(
      "[scheduler] continuous cmpl-a deferred reason=scheduled_token_budget queued=3.0ms prompt_tokens=4096 max_tokens=64 waiting=1 prefilling=0 active=1/8 scheduled_tokens=128/512 scheduled_prompt_tokens=96/384 scheduled_completion_tokens=32/128 scheduled_memory=1.0 KB/4.1 KB",
    );
    expect(
      formatServeEvent({
        type: "generation_scheduler_phase",
        mode: "continuous",
        phase: "admitted",
        model: "mlx-community/Qwen3.6-27B-4bit",
        ids: ["cmpl-a", "cmpl-b"],
        batchSize: 2,
        maxTokens: 64,
        maxTokensByRequest: [32, 64],
        queuedMsByRequest: [1.5, 2.5],
        schedulerMs: 2.5,
        waiting: 0,
        prefilling: 0,
        active: 2,
        maxBatchSize: 8,
        ...SCHEDULER_TOKENS,
      }),
    ).toBe(
      "[scheduler] continuous admitted size=2 wait_ms=1.5,2.5 max_tokens=64 per_request=32,64 ids=cmpl-a,cmpl-b waiting=0 prefilling=0 active=2/8 scheduled_tokens=128/512 scheduled_prompt_tokens=96/384 scheduled_completion_tokens=32/128 scheduled_memory=1.0 KB/4.1 KB",
    );
    expect(
      formatServeEvent({
        type: "generation_scheduler_phase",
        mode: "continuous",
        phase: "first_token",
        model: "mlx-community/Qwen3.6-27B-4bit",
        id: "cmpl-a",
        ids: ["cmpl-a"],
        completionTokens: 1,
        queuedMs: 4,
        schedulerMs: 4,
        waiting: 0,
        prefilling: 0,
        active: 2,
        maxBatchSize: 8,
        ...SCHEDULER_TOKENS,
      }),
    ).toBe(
      "[scheduler] continuous cmpl-a first_token at=4.0ms queued=4.0ms completion_tokens=1 waiting=0 prefilling=0 active=2/8 scheduled_tokens=128/512 scheduled_prompt_tokens=96/384 scheduled_completion_tokens=32/128 scheduled_memory=1.0 KB/4.1 KB",
    );
    expect(
      formatServeEvent({
        type: "generation_scheduler_phase",
        mode: "continuous",
        phase: "finished",
        model: "mlx-community/Qwen3.6-27B-4bit",
        id: "cmpl-a",
        ids: ["cmpl-a"],
        completionTokens: 64,
        finishReason: "length",
        queuedMs: 12,
        schedulerMs: 12,
        waiting: 0,
        prefilling: 0,
        active: 0,
        maxBatchSize: 8,
        ...SCHEDULER_TOKENS,
      }),
    ).toBe(
      "[scheduler] continuous cmpl-a finished reason=length elapsed=12.0ms completion_tokens=64 waiting=0 prefilling=0 active=0/8 scheduled_tokens=128/512 scheduled_prompt_tokens=96/384 scheduled_completion_tokens=32/128 scheduled_memory=1.0 KB/4.1 KB",
    );
    expect(
      formatServeEvent({
        type: "generation_scheduler_phase",
        mode: "continuous",
        phase: "cancelled",
        model: "mlx-community/Qwen3.6-27B-4bit",
        id: "cmpl-a",
        ids: ["cmpl-a"],
        completionTokens: 0,
        queuedMs: 3,
        schedulerMs: 3,
        waiting: 0,
        prefilling: 0,
        active: 0,
        maxBatchSize: 8,
        ...SCHEDULER_TOKENS,
      }),
    ).toBe(
      "[scheduler] continuous cmpl-a cancelled elapsed=3.0ms completion_tokens=0 waiting=0 prefilling=0 active=0/8 scheduled_tokens=128/512 scheduled_prompt_tokens=96/384 scheduled_completion_tokens=32/128 scheduled_memory=1.0 KB/4.1 KB",
    );
    expect(
      formatServeEvent({
        type: "generation_admission_batch",
        mode: "micro",
        engineMode: "batch",
        model: "mlx-community/Qwen3.6-27B-4bit",
        ids: ["cmpl-a", "cmpl-b"],
        batchSize: 2,
        maxTokens: 64,
        maxTokensByRequest: [32, 64],
      }),
    ).toBe(
      "[batch] micro model=mlx-community/Qwen3.6-27B-4bit size=2 engine=batch max_tokens=64 per_request=32,64 ids=cmpl-a,cmpl-b admitted",
    );
    expect(
      formatServeEvent({
        type: "generation_complete",
        id: "cmpl-test",
        protocol: "openai.chat_completions",
        model: "mlx-community/Qwen3.6-27B-4bit",
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
        model: "mlx-community/Qwen3.6-27B-4bit",
        code: "model_not_found",
        message: 'Model "missing" is not served by this endpoint.',
        durationMs: 321,
      }),
    ).toBe(
      '[generation:error] cmpl-test openai.chat_completions model=mlx-community/Qwen3.6-27B-4bit model_not_found: Model "missing" is not served by this endpoint. in 321.0ms',
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
          type: "generation_scheduler_phase",
          mode: "continuous",
          phase: "queued",
          model: "mlx-community/Qwen3.6-27B-4bit",
          id: "cmpl-test",
          ids: ["cmpl-test"],
          queuedAhead: 0,
          promptTokens: 12,
          maxTokens: 32,
          schedulerMs: 0,
          waiting: 1,
          prefilling: 0,
          active: 0,
          maxBatchSize: 8,
          ...SCHEDULER_TOKENS,
        },
        false,
      ),
    ).toBe(true);
    expect(
      shouldLogServeEvent(
        {
          type: "generation_admission_batch",
          mode: "micro",
          engineMode: "batch",
          model: "mlx-community/Qwen3.6-27B-4bit",
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
          type: "generation_route_decision",
          id: "cmpl-test",
          protocol: "openai.chat_completions",
          model: "mlx-community/Qwen3.6-27B-4bit",
          route: "single",
          eligible: false,
          reason: "unsupported_model_type",
          modelType: "qwen3_5_text",
          maxBatchSize: 32,
          ...ROUTE_STRATEGY,
          stream: false,
        },
        false,
      ),
    ).toBe(true);
    expect(
      shouldLogServeEvent(
        {
          type: "generation_prompt_prepare",
          phase: "start",
          id: "cmpl-test",
          protocol: "openai.chat_completions",
          model: "mlx-community/Qwen3.6-27B-4bit",
          inputKind: "messages",
        },
        false,
      ),
    ).toBe(true);
    expect(
      shouldLogServeEvent(
        {
          type: "generation_model_lane_wait",
          id: "cmpl-test",
          protocol: "openai.chat_completions",
          model: "mlx-community/Qwen3.6-27B-4bit",
          lane: "model",
          waitMs: 1,
          inFlightAtQueue: 1,
          queuedAhead: 0,
          inFlightAtDispatch: 1,
          queuedAtDispatch: 0,
          maxConcurrentJobs: 1,
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
          model: "mlx-community/Qwen3.6-27B-4bit",
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
          stream: false,
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
    await runServeCli(
      ["repo/model", "--host", "0.0.0.0", "--remote-image-host", "example.com", "--verbose"],
      {
        async serveModel(options) {
          options.onProgress?.({ stage: "tokenizer", status: "complete", directory: "/tmp" });
          options.onEvent?.({
            type: "request_start",
            method: "POST",
            path: "/v1/chat/completions",
          });
          expect(options.source).toBe("repo/model");
          expect(options.hostname).toBe("0.0.0.0");
          expect(options.maxPromptTokens).toBe(4096);
          expect(options.maxBatchSize).toBe(32);
          expect(options.batchWindowMs).toBe(1);
          expect(options.prefillStepSize).toBe(512);
          expect(options.activePrefillStepSize).toBe(128);
          expect(options.activeDecodeStepsPerPrefillChunk).toBe(16);
          expect(options.streamDecodeInterval).toBe(1);
          expect(options.maxConcurrentRequests).toBe(1);
          expect(options.promptPrefixCacheMaxEntries).toBe(1);
          expect(options.promptPrefixCacheMaxBytes).toBeUndefined();
          expect(options.remoteImageHosts).toEqual(["example.com"]);
          expect(options.gpuMemoryUtilization).toBe(0.9);
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
      },
    );

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
        expect(options.promptPrefixCacheMaxEntries).toBe(1);
        expect(options.promptPrefixCacheMaxBytes).toBeUndefined();
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

  test("routes lazy single-model CLI serving through the source model pool", async () => {
    const running = fakeRunningServer(["repo/model"]);
    await runServeCli(
      ["repo/model", "--model-load-policy", "lazy", "--model-idle-ttl-ms", "1000"],
      {
        async serveModel() {
          throw new Error("eager single-model server should not be used for lazy loading");
        },
        async serveModels(options) {
          expect(options.models).toEqual([{ source: "repo/model", modelId: "repo/model" }]);
          expect(options.modelLoadPolicy).toBe("lazy");
          expect(options.modelIdleTtlMs).toBe(1000);
          expect(options.pinnedModels).toEqual([]);
          return running;
        },
        log() {},
        error() {},
        async waitForShutdown(server) {
          server.stop();
        },
      },
    );

    expect(running.stopped).toBe(true);
  });

  test("runs local model-root discovery through the lazy source pool", async () => {
    const logs: string[] = [];
    const errors: string[] = [];
    const running = fakeRunningServer(["manual", "org/qwen", "flat"]);

    await runServeCli(
      ["--model", "manual=repo/manual", "--model-root", "/models", "--pin-model", "org/qwen"],
      {
        async serveModel() {
          throw new Error("eager single-model server should not be used for model-root discovery");
        },
        async serveModels(options) {
          expect(options.models).toEqual([
            { source: "repo/manual", modelId: "manual" },
            { source: "/models/org/qwen", modelId: "org/qwen" },
            { source: "/models/flat", modelId: "flat" },
          ]);
          expect(options.modelLoadPolicy).toBe("lazy");
          expect(options.pinnedModels).toEqual(["org/qwen"]);
          return running;
        },
        discoverLocalModelSources(root) {
          expect(root).toBe("/models");
          return [
            {
              source: "/models/org/qwen",
              modelId: "org/qwen",
              modelType: "qwen3_5",
              architectures: ["Qwen3_5ForConditionalGeneration"],
              hasVisionConfig: true,
            },
            {
              source: "/models/flat",
              modelId: "flat",
              modelType: "gemma",
              architectures: [],
              hasVisionConfig: false,
            },
          ];
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
      },
    );

    expect(errors).toEqual([]);
    expect(logs.join("\n")).toContain("Models: manual, org/qwen, flat");
    expect(logs.join("\n")).toContain("Model load policy: lazy");
    expect(running.stopped).toBe(true);
  });

  test("reports empty model-root discovery before server start", async () => {
    const errors: string[] = [];
    let exitCode = -1;

    await runServeCli(["--model-root", "/empty"], {
      async serveModels() {
        throw new Error("server should not start when discovery is empty");
      },
      discoverLocalModelSources(root) {
        expect(root).toBe("/empty");
        return [];
      },
      error(message) {
        errors.push(message);
      },
      exit(code) {
        exitCode = code;
      },
    });

    expect(exitCode).toBe(1);
    expect(errors).toEqual([
      "No supported local autoregressive model checkpoints discovered under /empty.",
    ]);
  });

  test("reports invalid post-discovery model-root entries before server start", async () => {
    const duplicateErrors: string[] = [];
    let duplicateExitCode = -1;

    await runServeCli(["--model", "flat=repo/manual", "--model-root", "/models"], {
      async serveModels() {
        throw new Error("server should not start when discovered ids duplicate explicit ids");
      },
      discoverLocalModelSources(root) {
        expect(root).toBe("/models");
        return [
          {
            source: "/models/flat",
            modelId: "flat",
            modelType: "gemma",
            architectures: [],
            hasVisionConfig: false,
          },
        ];
      },
      error(message) {
        duplicateErrors.push(message);
      },
      exit(code) {
        duplicateExitCode = code;
      },
    });

    expect(duplicateExitCode).toBe(1);
    expect(duplicateErrors).toEqual(['model id "flat" is duplicated.']);

    const pinErrors: string[] = [];
    let pinExitCode = -1;

    await runServeCli(["--model-root", "/models", "--pin-model", "missing"], {
      async serveModels() {
        throw new Error("server should not start when discovered ids omit a pinned model");
      },
      discoverLocalModelSources(root) {
        expect(root).toBe("/models");
        return [
          {
            source: "/models/flat",
            modelId: "flat",
            modelType: "gemma",
            architectures: [],
            hasVisionConfig: false,
          },
        ];
      },
      error(message) {
        pinErrors.push(message);
      },
      exit(code) {
        pinExitCode = code;
      },
    });

    expect(pinExitCode).toBe(1);
    expect(pinErrors).toEqual(['Pinned model "missing" is not part of this serve command.']);
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

function fakeRunningServer(
  modelIds: readonly string[] = ["mlx-community/Qwen3.6-27B-4bit"],
): RunningModelServer & {
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
    modelId: modelIds[0] ?? "mlx-community/Qwen3.6-27B-4bit",
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
