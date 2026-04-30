# @mlxts/serve

Serving is a first-class package surface, not an example. Protocol adapters stay
thin. OpenAI completions, chat completions, OpenResponses, and Anthropic
Messages normalize into the shared `GenerationEngine` contract without copying
generation logic between protocols.

`/v1/responses` is an OpenResponses-spec surface, not an OpenAI Responses clone.
OpenAI wording appears only where a compatibility client or wire shape requires
it. OpenResponses OpenAPI/source truth is audited before response items, tools,
files, state, or streaming events widen.

Admission controls are explicit and operator-facing. Generated-token,
prompt-token, total-token, concurrency, and batching limits stay separate in code,
errors, CLI output, and `/info` so long-context failures explain which budget was
hit. `/info` context metadata is configured admission truth, not a promise that
every advertised model window fits local memory.

Agent-facing CLI work follows `.agents/skills/axi/SKILL.md`. The legacy server
startup and request-progress stream is the long-running operator surface until
the serve CLI AXI migration. New finite discovery, inspection, status, and error
surfaces use AXI-shaped stdout, compact schemas, definitive empty states, and
structured actionable errors.

Memory preflight stays best-effort and honest. It estimates cache and prefill
memory from family config geometry, compares it with current MLX active memory
and the configured utilization budget, and skips rather than fakes certainty
when the model config is not understood. Continuous routes also reserve
estimated request memory against a model-level scheduler budget when config
geometry and MLX memory telemetry are available.
Lazy model-pool pressure relief is operator-explicit. The default policy rejects
the blocked request; `shed_non_pinned` evicts idle non-pinned models before
aborting active non-pinned request scopes. Pinned models are not pressure-shed.

`src/` is organized into role-based folders: `http/`, `streaming/`, `engine/`,
`protocols/`, `admission/`, `runtime/`, `observability/`, `model-loading/`,
and `media/`. Do not add new top-level source files. New protocol adapters land
in `protocols/`. New stream writers land in `streaming/writer-*.ts`. New cache
backends, scheduler variants, attention dispatch, and decoding strategies land
in `engine/`, with subfolders only when one role grows past five files.

Engine, protocol, HTTP, and streaming roles are separate. The engine executes
generation against a `CausalLM` and does not parse wire bodies or format wire
responses. Protocol adapters parse and format wire shapes, but do not touch
model execution or admission budgets. HTTP is the `Request`/`Response` layer.
Streaming is the only layer that touches SSE controllers.

Do not call admission micro-batching continuous batching. True continuous
batching needs a scheduler-owned decode loop plus batch-aware cache semantics in
`@mlxts/transformers`. That route is now real for eligible LLaMA-like, Qwen
3.6 text, and Gemma 3/4 layer-pattern requests. Claims stay tied to the exact
route evidence and do not imply prefix/paged cache or multimodal batching.

Family-owned cache, serve-owned scheduling. Serve manipulates
`TransformerCacheSnapshot` and `TransformerCache` only through public snapshot,
fork, store, and dispose operations. KV layout, layer-pattern handling,
recurrent state, and quantized storage stay in `@mlxts/transformers`; serve owns
matching, identity gating, eviction, accounting, metrics, and protocol usage.
Serve dispatches cache-shape scheduling on `CacheLayerKind`, not on family
identifiers.

Serve owns media transport, bounded host-side I/O, decode, cancellation, and
model-lane scheduling. Local file-id image reads require configured image roots
and remain image-only transport, not a general files API. Model-family
preprocessing - smart resize, patch tokens, grid metadata, image token
expansion, and vision tower wiring - stays in `@mlxts/transformers`. The seam
is protocol-neutral content in and prepared prompt out.

SSE writers share lifecycle scaffolding. Heartbeats, stream
observability, stop filtering, and reasoning-tag streams are common machinery;
each protocol writer owns only its protocol-specific state machine and terminal
wire chunks.

For full-KV continuous batching, long waiting prompts must be chunk-prefilled
between active decode steps rather than forwarded as one admission wall.
Qwen/Gemma routes preserve the real regression guardrails that assert
per-request TTFT, scheduler queue time, SSE lifecycle, and mixed long-prefill /
short-arrival fairness separately from aggregate throughput.
