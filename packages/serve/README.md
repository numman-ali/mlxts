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

The server exposes `/health`, `/info`, `/v1/models`, `/v1/completions`,
`/v1/chat/completions`, and `/v1/responses`:

```bash
curl -s http://127.0.0.1:8000/v1/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "mlx-community/Qwen3.6-27B-4bit",
    "prompt": "Write one crisp sentence about Apple Silicon ML.",
    "max_tokens": 64
  }'
```

Use `--api-key <key>` when binding outside localhost; it protects `/info` and
all `/v1/*` routes while leaving `/health` open for process checks.
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
generation at a time.

`GET /info` is a lightweight operator endpoint for confirming the served model
ids, enabled wire routes, configured request limits, per-model admission
metadata, and whether the current engine exposes streaming or batch generation.
The context metadata is an admission view, not a memory guarantee: long Qwen
contexts still need operator-set prompt/total limits that fit the machine. The
memory preflight is deliberately conservative and machine-specific; it is a
guardrail before prefill starts, not a promise that every later scheduler or
multi-request workload will fit. It does not expose local paths, cache
locations, access tokens, queue internals, or claims of continuous batching.

Generation start, admission micro-batch, static batch start, completion, and
errors are logged by default so native generation failures leave a useful last
known stage in the terminal. Use `--verbose` while debugging to add request
start/completion logs. Multi-model CLI loads are logged with the model index and
model id so long startup sequences are easier to follow.

The first-class model server wraps one loaded model in a small single-flight
admission queue. Nearby non-streaming requests can coalesce into one
micro-batch; the transformer-backed engine now turns eligible greedy full-cache
LLaMA-like groups into real static `generateBatch()` calls. Qwen hybrid caches,
Gemma 3/4 layer-pattern caches, sampled/model-native-default requests, and
streaming still fall back to the single-request path until a deeper scheduler
owns those decode patterns. Mixed `max_tokens` are supported inside that static
greedy batch path. Engines without native `generateBatch()` support still
benefit from serialized request admission instead of overlapping local
generations on the same model instance. Streaming requests pass through the same
concurrency gate, so one model-backed engine does not accept overlapping decode
loops just because the HTTP surface is async.

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
Responses API. It accepts string `input`, optional `instructions`,
`max_output_tokens`, model-native sampling fields, `seed`, `metadata`, and
non-persistent `store: false`; it returns a `response` object with `output`,
`output_text`, usage, and reasoning items when the model result includes
reasoning content. Stateful continuation, background jobs, tools, files/images,
streaming, prompt templates, truncation, and non-text output formats are rejected
explicitly until those semantics are implemented for real.

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
`/v1/*` routes require `Authorization: Bearer <key>`; `/health` stays open for
local process checks.

`createMicroBatchingGenerationEngine()` coalesces nearby non-streaming requests
into `generateBatch()` calls when the underlying engine supports them. This is
admission micro-batching, not full continuous token-level batching; production
continuous batching should live inside a batch-aware generation engine behind
the same contract.

`createTransformersGenerationEngine()` supports a narrow native static batch
path today: non-streaming greedy requests against full-cache LLaMA-like models
with compatible sampling options, including per-request `max_tokens`. Other
model families or sampled/default sampled requests remain correct by falling
back to single generation instead of pretending the serving layer can batch cache
shapes it does not own yet.

`createConcurrencyLimitGenerationEngine()` is the companion admission guard for
single-model serving. It bounds the number of active model jobs across
`generate()`, `generateBatch()`, and `stream()` so a local runtime is not asked
to run overlapping decode loops accidentally.

See `examples/serve-completions/` for a deterministic four-agent concurrency
harness. Real model serving is a package API and CLI, not an example entrypoint.
Use `@mlxts/agent` when you want tool execution and a Codex-style loop on top of
the served chat endpoint.
