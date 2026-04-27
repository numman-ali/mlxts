# @mlxts/serve

OpenAI-compatible serving for mlxts.

`@mlxts/serve` is the first-class way to put a local or Hugging Face model behind
an endpoint. Examples can demonstrate patterns, but model serving itself belongs
in this package.

## CLI

```bash
mlxts-serve mlx-community/Qwen3.6-27B-4bit --port 8000
```

or directly from the package:

```bash
bunx @mlxts/serve mlx-community/Qwen3.6-27B-4bit --port 8000
```

Use repeatable `--model` entries when one local endpoint should expose multiple
models. Plain `--model <source>` uses the source as the served id; `id=source`
sets an explicit OpenAI model id:

```bash
mlxts-serve \
  --model gemma=google/gemma-4-E2B-it \
  --model qwen=mlx-community/Qwen3.6-27B-4bit \
  --port 8000 \
  --local-files-only
```

The server exposes `/health`, `/info`, `/metrics`, `/v1/models`,
`/v1/completions`, `/v1/chat/completions`, and `/v1/responses`:

```bash
curl -s http://127.0.0.1:8000/v1/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "mlx-community/Qwen3.6-27B-4bit",
    "prompt": "Write one crisp sentence about Apple Silicon ML.",
    "max_tokens": 64
  }'
```

Use `--api-key <key>` when binding outside localhost; it protects `/info`,
`/metrics`, and all `/v1/*` routes while leaving `/health` open for process
checks.
`--max-generated-tokens <n>` rejects unsafe generation lengths before they reach
the model. `--max-prompt-tokens <n>` separately caps tokenized prompt/prefill
size, and `--max-total-tokens <n>` caps `prompt_tokens + max_tokens` against
the lower of the server setting and checkpoint-declared context window.
`--gpu-memory-utilization <f>` adds a best-effort MLX memory preflight: the
server estimates request-local KV cache, recurrent cache state, and prefill
temporary memory from the loaded model config, then rejects requests whose
projected active memory would exceed that fraction of the MLX allocator limit.
`--max-batch-size <n>` plus `--batch-window-ms <n>` control the built-in
admission micro-batching queue for concurrent requests against one model
instance. `--max-concurrent-requests <n>` bounds the number of in-flight model
jobs; the default is `1` so one loaded Gemma/Qwen runtime is owned by one
generation at a time. `--stream-decode-interval <n>` controls how often the
transformer engine decodes generated tokens into SSE text; the default is `1`
for interactive chat responsiveness, while larger values can reduce tokenizer
work on long-output throughput runs.

Eligible continuous-scheduler requests also share a model-level scheduled-token
reservation budget derived from `max_total_tokens * max_batch_size`. This keeps
multiple scheduler keys, for example different sampling defaults, from
silently over-admitting more total prompt-plus-generation work than the model
instance was configured to carry. The budget is internal and derived from
existing limits; it is not a separate operator flag yet.

`GET /info` is a lightweight operator endpoint for confirming the served model
ids, enabled wire routes, configured request limits, per-model admission
metadata, selected runtime strategy, and whether the current engine exposes
streaming or batch generation. The reported strategy is derived from currently
implemented behavior: scheduler `auto`, managed model-precision cache, attention
`auto`, model-native decoding, streaming decode cadence, and admit-only memory
preflight when configured. The context metadata is an admission view, not a
memory guarantee: long Qwen contexts still need operator-set prompt/total limits
that fit the machine. The memory preflight is deliberately conservative and
machine-specific; it is a guardrail before prefill starts, not a promise that
every later scheduler or multi-request workload will fit. It does not expose
local paths, cache locations, access tokens, or queue internals.

`GET /metrics` exposes Prometheus text-format metrics from the same structured
serve events used by logs and benchmark reports. Labels are deliberately
bounded: dynamic model-route paths collapse to `/v1/models/:model`, unknown
served model ids collapse to `__unknown__` when the model list is known, and
request ids/prompts/errors are not labels. The surface currently covers HTTP
request counts/latency/in-flight gauges, generation starts/completions/errors,
token totals and histograms, route decisions, model-lane waits, scheduler
phases, batch sizes, prefill chunks, stream terminal results, server-side TTFT,
SSE frame/byte counts, output frame counts, and latest MLX allocator memory
gauges. Stream TTFT is measured inside the server from generation start to the
first output-bearing SSE frame; benchmark TTFT remains client-observed and is
kept separate. Scraping `/metrics` is excluded from HTTP counters so Prometheus
polling does not dominate local operator signals.

Generation start, admission micro-batch, static batch start, completion, and
errors are logged by default so native generation failures leave a useful last
known stage in the terminal. Route-decision logs include the selected scheduler,
cache, attention, and decoding strategy so operator output matches `/info` and
benchmark reports. Use `--verbose` while debugging to add request
start/completion logs. Multi-model CLI loads are logged with the model index and
model id so long startup sequences are easier to follow.

The first-class model server wraps one loaded model in a small scheduler-aware
admission queue. Eligible full-cache LLaMA-like, Qwen 3.6 text, and Gemma 3/4
layer-pattern requests can join the transformer-owned continuous scheduler,
including streaming completions and chat streams. The scheduler supports greedy
decode and model-native sampled defaults by keeping sampler state per request
row. Engines without native batching support still benefit from serialized
request admission instead of overlapping local generations on the same model
instance.

When `temperature`, `top_p`, or `top_k` are omitted, serving leaves them unset so
`@mlxts/transformers` can apply the checkpoint's `generation_config.json`.
Qwen-style thinking templates can be controlled per request with
`"chat_template_kwargs": { "enable_thinking": false }`; generated `<think>`
content is returned as `message.reasoning_content`, not mixed into
`message.content`. When tools are enabled, non-streaming chat completions format
valid generated `<tool_call>` blocks as OpenAI `message.tool_calls`, while
streaming chat completions buffer complete tool-call envelopes and emit
OpenAI-compatible `delta.tool_calls` chunks with `finish_reason: "tool_calls"`.
`/v1/completions` and `/v1/chat/completions` both support SSE streaming when the
served engine supports it, and chat streaming keeps reasoning in
`reasoning_content` deltas instead of leaking raw `<think>` tags.

`/v1/responses` starts with a deliberately narrow text-only subset of OpenAI's
Responses API. It accepts string `input` or text-only message item arrays,
optional `instructions`, `max_output_tokens`, model-native sampling fields,
`seed`, `metadata`, and non-persistent `store: false`; it returns a `response`
object with `output`, `output_text`, usage, and reasoning items when the model
result includes reasoning content. `stream: true` emits semantic Responses SSE
events such as `response.created`, `response.output_text.delta`,
`response.reasoning_text.delta`, and `response.completed` for text output.
Stateful continuation, background jobs, tools, files/images, prompt templates,
truncation, and non-text output formats are rejected explicitly until those
semantics are implemented for real.

## Programmatic Serving

```ts
import { serveModel } from "@mlxts/serve";

const server = await serveModel({
  source: "mlx-community/Qwen3.6-27B-4bit",
  modelId: "qwen-local",
  port: 8000,
  maxGeneratedTokens: 2048,
  maxPromptTokens: 4096,
  maxTotalTokens: 4096,
  gpuMemoryUtilization: 0.9,
  maxBatchSize: 16,
  batchWindowMs: 2,
  streamDecodeInterval: 1,
  maxConcurrentRequests: 1,
});

console.log(server.endpoint);
```

Use `serveModels()` to load and expose multiple local directories or Hub repos
from one process. Loading is sequential to avoid RAM spikes, and the loaded
models are owned by the returned server:

```ts
import { serveModels } from "@mlxts/serve";

const server = await serveModels({
  models: [
    { source: "google/gemma-4-E2B-it", modelId: "gemma-local" },
    { source: "mlx-community/Qwen3.6-27B-4bit", modelId: "qwen-local" },
  ],
  port: 8000,
  maxConcurrentRequests: 1,
});

console.log(server.modelIds);
```

If you already own the loaded model and tokenizer, use `serveLoadedModel()`:

```ts
import { serveLoadedModel } from "@mlxts/serve";

const server = serveLoadedModel({
  model,
  tokenizer,
  modelId: "local-model",
  port: 8000,
});
```

Use `serveLoadedModels()` when one local process should expose multiple already
loaded models behind the same endpoint. Each model id gets its own
transformer engine, request limits, concurrency gate, and micro-batch queue
before the shared router dispatches by the OpenAI `model` field:

```ts
import { serveLoadedModels } from "@mlxts/serve";

const server = serveLoadedModels({
  models: [
    { model: gemma, tokenizer: gemmaTokenizer, modelId: "gemma-local" },
    { model: qwen, tokenizer: qwenTokenizer, modelId: "qwen-local" },
  ],
  port: 8000,
  maxConcurrentRequests: 1,
});

console.log(server.modelIds);
```

Call `server.stop()` or dispose the returned server when the process should
release the endpoint and model resources.

## Benchmarking

Use the package-owned endpoint benchmark when the thing being tested is serving
quality rather than raw in-process model decode:

```bash
bun run bench:serve --model mlx-community/Qwen3.6-27B-4bit \
  --model-id qwen-local \
  --rungs 128x128@1,1024x512@1,10000x128@2 \
  --greedy \
  --ignore-eos \
  --report-json .tmp/qwen-serve-ladder.json \
  --max-concurrent-requests 1 \
  --max-batch-size 8 \
  --batch-window-ms 2
```

The benchmark is cached/local-only by default so overnight ladders do not hide
download time in endpoint numbers; pass `--allow-download` only when that is
intentional. Omitted sampling fields preserve model-native
`generation_config.json`; `--greedy` is explicit for deterministic throughput
runs. It sends exact token-array prompts through `/v1/completions` and reports
wall time, request throughput, end-to-end completion-token throughput,
total-token throughput, mean/p95/max latency, memory, finish reasons, admission
micro-batch rows, real static batch rows, and continuous scheduler admission
rows.
Completions benchmarks use deterministic token-array prompts for exact token
counts; chat and Responses synthesize text prompts for protocol health runs.
Use `--ignore-eos` for exact-length throughput ladders when comparing against
in-process benchmarks that intentionally decode the full requested token count;
normal serving behavior still honors EOS unless this extension is explicit.
Pass `--stream` to drive the same rungs through SSE completions with
`stream_options.include_usage=true`; streaming runs add mean time-to-first-token,
prompt-to-first-token throughput, post-TTFT completion throughput, stream chunk
gap timing, SSE chunk count, and streamed byte count. JSON reports also preserve
the server `streamDecodeInterval`, per-request duration, TTFT, token counts,
launch offset, streaming cadence, and finish reason. They also include
benchmark-observed server event timelines per
generation id, including route-decision timing, model-lane wait timing, prefill
progress timing, first completion-progress timing, completion/error timing,
continuous-scheduler queue timing, scheduled-token pressure, server-side stream
TTFT/result/chunk/byte evidence, and the largest silent gap between server
events. Staggered or concurrent runs can therefore be inspected without relying
only on trial averages.
Use `--stream-decode-interval` on `mlxts-serve` or `streamDecodeInterval` in
programmatic serving when you need an explicit tradeoff between per-token chat
responsiveness and lower tokenizer overhead.
Pass `--request-stagger-ms <n>` to launch concurrent requests at deliberate
offsets rather than all at once. That is the benchmark shape for testing
waiting-row scheduler fairness instead of only admission-window coalescing.
For huge prompt rungs, prefer streaming: the server sends SSE keepalive comments
and uses cooperative streaming prefill so long prefill phases do not leave the
client connection silent until the first generated token.
For very long buffered runs, `--request-timeout-ms` controls the benchmark
client timeout independently of server-side admission limits; it defaults to one
hour so long-context prefill does not get mislabeled as a model failure.

Use comma lists with the default cartesian matrix for broad serving sweeps,
`--matrix zip` for paired prompt/output rungs, or `--rungs` for a deliberate
capability ladder such as `128x128@1,1024x512@1,10000x128@2`. Add
`--report-json <path>` for overnight evidence that can be compared later. The
batch row counters are important: eligible Qwen and Gemma 3/4 requests now
report continuous scheduler admissions for both buffered and streaming
completions, including sampled/model-native-default requests. Endpoint benchmark
output should be used to separate real batch execution from admission coalescing.
This harness measures completions serving over token-array prompts; use `--protocol chat` or
`--protocol responses` when the thing under test is the wire adapter and
chat-template path. Completions remains the exact-token throughput mode; chat
and Responses use deterministic text prompts and should be reported as protocol
health, not exact token-array parity. `--ignore-eos` is rejected with
`--protocol responses` because the current Responses benchmark does not expose
that nonstandard serving extension. Tool quality still needs its own benchmark.

## Regression Matrix

Use the serving regression matrix before substantial serving/model changes:

```bash
bun run --filter '@mlxts/serve' regression:serve
```

That cheap path runs focused serving, protocol, streaming, batching, and
benchmark tests without loading real checkpoints. When cached Qwen/Gemma
checkpoints are available, the real endpoint smoke composes the existing
`bench:serve` harness and hard-fails on structural regressions:

```bash
bun run packages/serve/scripts/regression-serve-matrix.ts --real-models
```

For heavier local proof work, add `--capability-smoke`; it includes longer Qwen
output/context endpoint rungs and writes JSON reports under
`.tmp/serve-regression/`. These commands are lock-guarded and intentionally
sequential. The real smoke asserts Qwen/Gemma route reasons, server-request
evidence, client-observed streaming responsiveness, server-side stream writer
evidence, and continuous scheduler counters, including concurrent `@2` streamed
completions, so batching claims stay tied to observed endpoint behavior.

## Engine Primitives

The package starts with the OpenAI completions, chat completions, and narrow
Responses APIs, but the internal shape is protocol-neutral: wire requests
normalize into one `NormalizedGenerationRequest` before they reach a generation
engine. Anthropic Messages and broader OpenAI Responses capabilities should be
added as protocol adapters over the same core path, not as copied serving stacks.

```ts
import {
  createFetchHandler,
  createMicroBatchingGenerationEngine,
  createModelRouterGenerationEngine,
  createRequestLimitGenerationEngine,
} from "@mlxts/serve";

const fetch = createFetchHandler({
  engine: createRequestLimitGenerationEngine({
    maxGeneratedTokens: 2048,
    engine: createMicroBatchingGenerationEngine({
      engine: createModelRouterGenerationEngine({
        engines: {
          tiny: {
            generate(request) {
              return {
                text: `model saw: ${request.input.kind === "text" ? request.input.text : ""}`,
                finishReason: "stop",
              };
            },
          },
        },
      }),
    }),
  }),
});
```

`createRequestLimitGenerationEngine()` rejects oversized generations before they
reach the model. Use it for server-side `max_tokens` admission limits; do not
rely on model/runtime crashes as a safety boundary.

```ts
import {
  createModelRouterGenerationEngine,
  createRequestLimitGenerationEngine,
  createTransformersGenerationEngine,
  startServeServer,
} from "@mlxts/serve";
import { loadCausalLM, loadPretrainedTokenizer } from "@mlxts/transformers";

const model = await loadCausalLM("path-or-repo-id");
const tokenizer = await loadPretrainedTokenizer("path-or-repo-id");

startServeServer({
  port: 8000,
  apiKey: "optional-local-secret",
  models: [{ id: "local-model" }],
  engine: createModelRouterGenerationEngine({
    engines: {
      "local-model": createRequestLimitGenerationEngine({
        maxGeneratedTokens: 2048,
        engine: createTransformersGenerationEngine({
          model,
          tokenizer,
        }),
      }),
    },
  }),
});
```

`GET /v1/models` returns the configured `models` list. When `apiKey` is set,
`/info`, `/metrics`, and `/v1/*` routes require `Authorization: Bearer <key>`;
`/health` stays open for local process checks.

`createTransformersGenerationEngine()` now owns the first real continuous
batching path for loaded-model serving: eligible greedy or sampled requests can
join an active decode loop between token steps, including streaming
completions/chat streams for full-cache LLaMA-like models, Qwen 3.6 text models
with model-owned hybrid batch caches, and Gemma 3/4 layer-pattern models. It emits
`generation_scheduler_phase` events with `mode: "continuous"` so benchmark
output can separate scheduler queue, prefill, admission, first-token, and finish
phases from admission coalescing and static batch calls.

The continuous path is still intentionally narrow. Prefix cache, paged cache,
multimodal batching, and broader cache policies still fall back to the
single-model lane until their semantics are represented properly below the
serving layer.

`createMicroBatchingGenerationEngine()` and
`createConcurrencyLimitGenerationEngine()` remain available as lower-level
composition tools, but the first-class loaded-model server uses the
transformer-owned scheduler so eligible requests are not accidentally serialized
before they can join the active batch.

See `examples/serve-completions/` for a deterministic four-agent concurrency
harness. Real model serving is a package API and CLI, not an example entrypoint.
Use `@mlxts/agent` when you want tool execution and a Codex-style loop on top of
the served chat endpoint.
