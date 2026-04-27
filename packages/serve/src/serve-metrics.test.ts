import { describe, expect, test } from "bun:test";

import { createServeMetrics, normalizeServeMetricPath } from "./serve-metrics";

describe("serve metrics", () => {
  test("normalizes dynamic HTTP paths", () => {
    expect(normalizeServeMetricPath("/health")).toBe("/health");
    expect(normalizeServeMetricPath("/v1/models/qwen-local")).toBe("/v1/models/:model");
    expect(normalizeServeMetricPath("/not-real")).toBe("__unmatched__");
  });

  test("renders bounded Prometheus metrics from serve events", () => {
    const metrics = createServeMetrics({ modelIds: ["known"] });
    metrics.record({ type: "request_start", method: "GET", path: "/v1/models/known" });
    metrics.record({
      type: "request_complete",
      method: "GET",
      path: "/v1/models/known",
      status: 200,
      durationMs: 12,
    });
    metrics.record({
      type: "generation_start",
      id: "cmpl-1",
      protocol: "openai.completions",
      model: "known",
      inputKind: "text",
      maxTokens: 8,
      stream: false,
    });
    metrics.record({
      type: "generation_route_decision",
      id: "cmpl-1",
      protocol: "openai.completions",
      model: "unknown-model",
      route: "single",
      eligible: false,
      reason: "unsupported_model_type",
      modelType: 'odd"model',
      maxBatchSize: 1,
      schedulerMode: "auto",
      cacheBackend: "managed",
      attentionBackend: "auto",
      decodingBackend: "model",
      stream: false,
    });
    metrics.record({
      type: "generation_prefill_progress",
      id: "cmpl-1",
      protocol: "openai.completions",
      model: "known",
      promptTokens: 5,
      processedPrefillTokens: 5,
      totalPrefillTokens: 5,
      chunkTokens: 5,
      maxTokens: 8,
      memory: { activeBytes: 1, cacheBytes: 2, peakBytes: 3, limitBytes: 4 },
    });
    metrics.record({
      type: "generation_complete",
      id: "cmpl-1",
      protocol: "openai.completions",
      model: "known",
      finishReason: "stop",
      promptTokens: 5,
      completionTokens: 3,
      totalTokens: 8,
      durationMs: 40,
    });
    metrics.record({
      type: "generation_model_lane_wait",
      id: "cmpl-1",
      protocol: "openai.completions",
      model: "known",
      lane: "model",
      waitMs: 3,
      inFlightAtQueue: 0,
      queuedAhead: 0,
      inFlightAtDispatch: 1,
      queuedAtDispatch: 0,
      maxConcurrentJobs: 1,
    });

    const text = metrics.format();

    expect(text).toContain(
      'mlxts_serve_http_requests_total{method="GET",path="/v1/models/:model",status="200"} 1',
    );
    expect(text).toContain(
      'mlxts_serve_generation_requests_total{model="known",protocol="openai.completions",input_kind="text",stream="false"} 1',
    );
    expect(text).toContain(
      'mlxts_serve_generation_active{model="known",protocol="openai.completions"} 0',
    );
    expect(text).toContain(
      'mlxts_serve_generation_tokens_total{model="known",protocol="openai.completions",kind="completion"} 3',
    );
    expect(text).toContain(
      'mlxts_serve_generation_prefill_tokens_total{model="known",protocol="openai.completions"} 5',
    );
    expect(text).toContain('mlxts_serve_memory_bytes{model="known",kind="active"} 1');
    expect(text).toContain(
      'mlxts_serve_model_lane_requests{model="known",lane="model",state="max_concurrent_jobs"} 1',
    );
    expect(text).toContain(
      'mlxts_serve_generation_route_decisions_total{model="__unknown__",protocol="openai.completions",route="single",eligible="false",reason="unsupported_model_type",model_type="odd\\"model",scheduler="auto",cache="managed",attention="auto",decoding="model",stream="false"} 1',
    );
  });

  test("excludes metrics scrapes from HTTP counters", () => {
    const metrics = createServeMetrics();
    metrics.record({ type: "request_start", method: "GET", path: "/metrics" });
    metrics.record({
      type: "request_complete",
      method: "GET",
      path: "/metrics",
      status: 200,
      durationMs: 1,
    });

    expect(metrics.format()).not.toContain('path="/metrics"');
  });
});
