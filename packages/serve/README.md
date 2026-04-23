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

The server exposes `/health`, `/v1/models`, `/v1/completions`, and
`/v1/chat/completions`:

```bash
curl -s http://127.0.0.1:8000/v1/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "mlx-community/Qwen3.6-27B-4bit",
    "prompt": "Write one crisp sentence about Apple Silicon ML.",
    "max_tokens": 64
  }'
```

Use `--api-key <key>` when binding outside localhost, `--max-generated-tokens <n>`
to reject unsafe generation lengths before they reach the model, and
`--max-batch-size <n>` plus `--batch-window-ms <n>` to control the built-in
admission micro-batching queue for concurrent requests against one model
instance. `--max-concurrent-requests <n>` bounds the number of in-flight model
jobs; the default is `1` so one loaded Gemma/Qwen runtime is owned by one
generation at a time.

Generation start, completion, and errors are logged by default so native
generation failures leave a useful last known stage in the terminal. Use
`--verbose` while debugging to add request start/completion logs.

The first-class model server wraps one loaded model in a small single-flight
admission queue. Nearby non-streaming requests can coalesce into one
micro-batch; the transformer-backed engine now turns eligible greedy full-cache
LLaMA-like groups into real static `generateBatch()` calls. Qwen hybrid caches,
Gemma 3/4 layer-pattern caches, sampled/model-native-default requests, mixed
`max_tokens`, and streaming still fall back to the single-request path until a
deeper scheduler owns those decode patterns. Engines without native
`generateBatch()` support still benefit from serialized request admission instead
of overlapping local generations on the same model instance. Streaming requests
pass through the same concurrency gate, so one model-backed engine does not
accept overlapping decode loops just because the HTTP surface is async.

When `temperature`, `top_p`, or `top_k` are omitted, serving leaves them unset so
`@mlxts/transformers` can apply the checkpoint's `generation_config.json`.
Qwen-style thinking templates can be controlled per request with
`"chat_template_kwargs": { "enable_thinking": false }`; generated `<think>`
content is returned as `message.reasoning_content`, not mixed into
`message.content`. `/v1/completions` and `/v1/chat/completions` both support
SSE streaming when the served engine supports it, and chat streaming keeps
reasoning in `reasoning_content` deltas instead of leaking raw `<think>` tags.

## Programmatic Serving

```ts
import { serveModel } from "@mlxts/serve";

const server = await serveModel({
  source: "mlx-community/Qwen3.6-27B-4bit",
  modelId: "qwen-local",
  port: 8000,
  maxGeneratedTokens: 2048,
  maxBatchSize: 16,
  batchWindowMs: 2,
  maxConcurrentRequests: 1,
});

console.log(server.endpoint);
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

Call `server.stop()` or dispose the returned server when the process should
release the endpoint and model resources.

## Engine Primitives

The package starts with the OpenAI completions and chat completions APIs, but the internal shape is
protocol-neutral: wire requests normalize into one `NormalizedGenerationRequest`
before they reach a generation engine. Anthropic Messages and OpenAI Responses
should be added as protocol adapters over the same core path, not as copied
serving stacks.

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
with compatible generation options. Other model families or sampled/default
sampled requests remain correct by falling back to single generation instead of
pretending the serving layer can batch cache shapes it does not own yet.

`createConcurrencyLimitGenerationEngine()` is the companion admission guard for
single-model serving. It bounds the number of active model jobs across
`generate()`, `generateBatch()`, and `stream()` so a local runtime is not asked
to run overlapping decode loops accidentally.

See `examples/serve-completions/` for a deterministic four-agent concurrency
harness. Real model serving is a package API and CLI, not an example entrypoint.
Use `@mlxts/agent` when you want tool execution and a Codex-style loop on top of
the served chat endpoint.
