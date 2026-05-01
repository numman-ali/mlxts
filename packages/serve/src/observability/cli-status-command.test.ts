import { describe, expect, test } from "bun:test";

import type { ServeInfoResponse } from "../http/route-info";
import {
  fetchServeStatusInfo,
  formatServeStatus,
  formatServeStatusError,
  formatServeStatusUsage,
  parseServeStatusArgs,
  runServeStatusCli,
  type ServeStatusCliOptions,
  type ServeStatusFetch,
} from "./cli-status-command";

const SAMPLE_INFO: ServeInfoResponse = {
  status: "ok",
  router: "@mlxts/serve",
  version: null,
  model_id: "mlx-community/Qwen3.6-27B-4bit",
  model_ids: ["mlx-community/Qwen3.6-27B-4bit", "google/gemma-3-4b-it"],
  model_count: 2,
  models: [
    {
      id: "mlx-community/Qwen3.6-27B-4bit",
      context_window: 262_144,
      max_prompt_tokens: 262_144,
      max_total_tokens: 262_144,
      effective_total_tokens: 262_144,
    },
    {
      id: "google/gemma-3-4b-it",
      context_window: 131_072,
      max_prompt_tokens: 131_072,
      max_total_tokens: 131_072,
      effective_total_tokens: 131_072,
    },
  ],
  endpoints: [
    "/health",
    "/info",
    "/metrics",
    "/v1/models",
    "/v1/completions",
    "/v1/chat/completions",
    "/v1/responses",
    "/v1/messages",
  ],
  limits: {
    max_generated_tokens: 32_768,
    max_prompt_tokens: 262_144,
    max_total_tokens: 262_144,
    max_client_batch_size: 32,
    batch_window_ms: 1,
    prefill_step_size: 512,
    active_prefill_step_size: 128,
    active_decode_steps_per_prefill_chunk: 16,
    stream_decode_interval: 1,
    max_concurrent_requests: 1,
    prompt_prefix_cache_max_entries: 8,
    prompt_prefix_cache_max_bytes: null,
    gpu_memory_utilization: 0.85,
  },
  capabilities: {
    completions: true,
    chat_completions: "text_and_image_when_supported",
    responses: "text_and_image_when_supported",
    anthropic_messages: "text_and_image_when_supported",
    sse_streaming: true,
    batch_generation: true,
    reasoning_content: true,
    tool_calls: true,
  },
  runtime_strategy: {
    scheduler: {
      mode: "auto",
      max_batch_size: 32,
      batch_window_ms: 1,
      prefill_step_size: 512,
      active_prefill_step_size: 128,
      active_decode_steps_per_prefill_chunk: 16,
      max_concurrent_requests: 1,
    },
    cache: {
      backend: "managed",
      precision: "model",
      prompt_prefix_max_entries: 8,
      prompt_prefix_max_bytes: null,
    },
    attention: {
      backend: "auto",
    },
    decoding: {
      backend: "model",
    },
    streaming: {
      stream_decode_interval: 1,
    },
    memory: {
      policy: "admit_only",
      gpu_memory_utilization: 0.85,
    },
  },
  model_pool: {
    load_policy: "lazy",
    pressure_policy: "shed_non_pinned",
    pressure_release_timeout_ms: 120_000,
    idle_ttl_ms: null,
    models: [
      {
        id: "mlx-community/Qwen3.6-27B-4bit",
        pinned: false,
        state: "loaded",
        active_requests: 1,
        pressure_aborted_requests: 0,
      },
      {
        id: "google/gemma-3-4b-it",
        pinned: true,
        state: "not_loaded",
        active_requests: 0,
        pressure_aborted_requests: 0,
      },
    ],
  },
  prompt_prefix_cache: {
    total_retained_snapshots: 2,
    total_retained_snapshot_bytes: 4096,
    total_indexed_block_hashes: 6,
    total_token_blocks: 4,
    total_token_block_references: 5,
    total_unique_token_count: 256,
    total_referenced_token_count: 320,
    models: [
      {
        id: "mlx-community/Qwen3.6-27B-4bit",
        retained_snapshots: 1,
        retained_snapshot_bytes: 2048,
        indexed_block_hashes: 3,
        token_blocks: {
          block_size: 64,
          block_count: 2,
          block_references: 2,
          unique_token_count: 128,
          referenced_token_count: 128,
        },
      },
      {
        id: "google/gemma-3-4b-it",
        retained_snapshots: 1,
        retained_snapshot_bytes: 2048,
        indexed_block_hashes: 3,
        token_blocks: {
          block_size: 64,
          block_count: 2,
          block_references: 3,
          unique_token_count: 128,
          referenced_token_count: 192,
        },
      },
    ],
  },
};

type FetchCall = {
  url: string;
  authorization: string | null;
};

function statusOptions(overrides: Partial<ServeStatusCliOptions> = {}): ServeStatusCliOptions {
  return {
    baseUrl: "http://127.0.0.1:8000",
    timeoutMs: 5_000,
    full: false,
    ...overrides,
  };
}

function createStatusFetch(info: ServeInfoResponse = SAMPLE_INFO): {
  calls: FetchCall[];
  fetch: ServeStatusFetch;
} {
  const calls: FetchCall[] = [];
  return {
    calls,
    fetch: async (input, init) => {
      const url = String(input);
      const headers = new Headers(init?.headers);
      calls.push({ url, authorization: headers.get("authorization") });
      if (url.endsWith("/health")) {
        return Response.json({ status: "ok" });
      }
      return Response.json(info);
    },
  };
}

describe("serve status CLI command", () => {
  test("parses defaults and environment fallbacks", () => {
    expect(parseServeStatusArgs([], {})).toEqual({
      kind: "status",
      options: {
        baseUrl: "http://127.0.0.1:8000",
        timeoutMs: 5_000,
        full: false,
      },
    });

    expect(
      parseServeStatusArgs([], {
        OPENAI_BASE_URL: "http://localhost:9000/v1",
        OPENAI_API_KEY: "env-token",
      }),
    ).toEqual({
      kind: "status",
      options: {
        baseUrl: "http://localhost:9000",
        apiKey: "env-token",
        timeoutMs: 5_000,
        full: false,
      },
    });
  });

  test("parses explicit status flags", () => {
    expect(
      parseServeStatusArgs(
        [
          "--base-url",
          "https://serve.example.test/proxy/v1/",
          "--api-key",
          "flag-token",
          "--timeout-ms",
          "2500",
          "--full",
        ],
        { OPENAI_BASE_URL: "http://ignored.test/v1", OPENAI_API_KEY: "ignored" },
      ),
    ).toEqual({
      kind: "status",
      options: {
        baseUrl: "https://serve.example.test/proxy",
        apiKey: "flag-token",
        timeoutMs: 2_500,
        full: true,
      },
    });
  });

  test("reports usage errors in structured stdout form", () => {
    expect(parseServeStatusArgs(["--base-url", "ftp://localhost"], {})).toMatchObject({
      kind: "help",
      exitCode: 2,
      message: 'Expected --base-url to use http or https, got "ftp:".',
    });
    expect(parseServeStatusArgs(["--base-url", "--full"], {})).toMatchObject({
      kind: "help",
      exitCode: 2,
      message: "Missing value for --base-url.",
    });
    expect(parseServeStatusArgs(["--timeout-ms", "1e3"], {})).toMatchObject({
      kind: "help",
      exitCode: 2,
      message: 'Expected --timeout-ms to be a positive integer, got "1e3".',
    });
    expect(formatServeStatusError("Missing value for --base-url.")).toContain("error:");
    expect(formatServeStatusUsage()).toContain("mlxts-serve status --base-url <url>");
  });

  test("formats compact default status with enough routing context", () => {
    const output = formatServeStatus(SAMPLE_INFO, "http://127.0.0.1:8000", false);

    expect(output).toContain("serve_status:");
    expect(output).toContain('base_url: "http://127.0.0.1:8000"');
    expect(output).toContain("models[2]{id,context_window,max_prompt_tokens,max_total_tokens}:");
    expect(output).toContain('"mlx-community/Qwen3.6-27B-4bit",262144,262144,262144');
    expect(output).toContain('model_pool: "lazy/shed_non_pinned"');
    expect(output).toContain('prompt_prefix_cache: "snapshots=2/token_blocks=4"');
    expect(output).toContain("limits:");
    expect(output).toContain("max_total_tokens: 262144");
    expect(output).toContain("Run `mlxts-serve status --base-url");
  });

  test("formats full status with endpoints and runtime strategy", () => {
    const output = formatServeStatus(SAMPLE_INFO, "http://127.0.0.1:8000", true);

    expect(output).toContain("runtime_strategy:");
    expect(output).toContain('cache: "managed/model"');
    expect(output).toContain("model_pool:");
    expect(output).toContain("pressure_release_timeout_ms: 120000");
    expect(output).toContain(
      "model_pool_models[2]{id,pinned,state,active_requests,pressure_aborted_requests}:",
    );
    expect(output).toContain('"google/gemma-3-4b-it",true,"not_loaded",0,0');
    expect(output).toContain("prompt_prefix_cache:");
    expect(output).toContain("retained_snapshot_bytes: 4096");
    expect(output).toContain(
      "prompt_prefix_cache_models[2]{id,retained_snapshots,retained_snapshot_bytes,indexed_block_hashes,block_size,token_blocks,token_block_references,unique_token_count,referenced_token_count}:",
    );
    expect(output).toContain('"google/gemma-3-4b-it",1,2048,3,64,2,3,128,192');
    expect(output).toContain("endpoints[8]:");
    expect(output).toContain('"/v1/chat/completions"');
  });

  test("fetches health and info with bearer auth", async () => {
    const statusFetch = createStatusFetch();
    const info = await fetchServeStatusInfo(
      statusOptions({
        baseUrl: "http://localhost:9000",
        apiKey: "secret",
      }),
      statusFetch.fetch,
    );

    expect(info).toEqual(SAMPLE_INFO);
    expect(statusFetch.calls).toEqual([
      { url: "http://localhost:9000/health", authorization: "Bearer secret" },
      { url: "http://localhost:9000/info", authorization: "Bearer secret" },
    ]);
  });

  test("translates remote and invalid payload failures", async () => {
    const authFailure: ServeStatusFetch = async (input) => {
      const url = String(input);
      if (url.endsWith("/health")) {
        return Response.json({ status: "ok" });
      }
      return Response.json({ error: { message: "Auth required." } }, { status: 401 });
    };

    await expect(fetchServeStatusInfo(statusOptions(), authFailure)).rejects.toThrow(
      "Auth required.",
    );

    const nonJsonInfo: ServeStatusFetch = async (input) =>
      String(input).endsWith("/health")
        ? Response.json({ status: "ok" })
        : new Response("not json");

    await expect(fetchServeStatusInfo(statusOptions(), nonJsonInfo)).rejects.toThrow(
      "Endpoint /info did not return JSON.",
    );
  });

  test("reports request timeouts", async () => {
    const hangingFetch: ServeStatusFetch = async (_input, init) =>
      await new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener("abort", () => reject(new Error("aborted")), {
          once: true,
        });
      });

    await expect(
      fetchServeStatusInfo(statusOptions({ timeoutMs: 1 }), hangingFetch),
    ).rejects.toThrow("Timed out after 1ms while requesting /health.");
  });

  test("runs success and error paths with AXI stdout exit codes", async () => {
    const logs: string[] = [];
    let exitCode = -1;
    const statusFetch = createStatusFetch();

    await runServeStatusCli(["--base-url", "http://localhost:9000/v1"], {
      fetch: statusFetch.fetch,
      env: {},
      log(message) {
        logs.push(message);
      },
      exit(code) {
        exitCode = code;
      },
    });

    expect(exitCode).toBe(0);
    expect(logs.join("\n")).toContain('base_url: "http://localhost:9000"');

    logs.length = 0;
    await runServeStatusCli(["--unknown"], {
      env: {},
      log(message) {
        logs.push(message);
      },
      exit(code) {
        exitCode = code;
      },
    });

    expect(exitCode).toBe(2);
    expect(logs.join("\n")).toContain("Unknown argument: --unknown");
  });
});
