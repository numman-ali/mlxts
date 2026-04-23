# Runtime Review: Serve Foundation

## Summary

This change adds the first `@mlxts/serve` package slice. The package is
intentionally protocol-adapter-first: OpenAI completions are supported now, while
chat completions, OpenAI Responses, and Anthropic Messages can later normalize
into the same internal generation request instead of copying route-specific
generation logic.

The first engine contract is deliberately small. Protocol adapters translate
wire requests into `NormalizedGenerationRequest`; engines generate or stream
`GenerationStreamEvent`; the Bun server shell only routes requests, formats
responses, and keeps protocol-specific error shape at the boundary.

The slice now also includes the serving seams needed before real production
serving: model-id routing, request admission limits, optional Bearer auth,
OpenAI-compatible model listing, admission micro-batching, OpenAI chat
completions, model-native sampling defaults, Qwen-style thinking separation,
and a first-class model-serving package API plus CLI. The companion
`@mlxts/agent` package owns the tool loop over the chat endpoint, so tool
execution stays out of the serving protocol layer.
Micro-batching coalesces nearby non-streaming requests into `generateBatch()`
calls when the underlying engine supports them. It is not continuous token-level
batching.

The operator-facing CLIs now show useful state by default: serving logs
generation start/completion/error events without `--verbose`, and the agent CLI
prints reasoning, tool calls, tool results, and final answers as distinct
sections. The agent parser accepts both the package's JSON `tool_call` envelope
and Qwen-style `<function=...>` blocks observed from the live local model. If a
turn exhausts the loop budget before a final answer, the CLI now prints an
`[agent]` stop notice instead of silently returning to the prompt.

## Files Reviewed

- `packages/serve/src/errors.ts`
- `packages/serve/src/index.ts`
- `packages/serve/src/batching-engine.ts`
- `packages/serve/src/cli.ts`
- `packages/serve/src/model-server.ts`
- `packages/serve/src/model-router.ts`
- `packages/serve/src/memory-telemetry.ts`
- `packages/serve/src/protocols/openai-chat-completions.ts`
- `packages/serve/src/protocols/openai-completions.ts`
- `packages/serve/src/protocols/openai-models.ts`
- `packages/serve/src/request-limits.ts`
- `packages/serve/src/server.ts`
- `packages/serve/src/transformers-engine.ts`
- `packages/serve/src/types.ts`
- `packages/agent/src/chat-model.ts`
- `packages/agent/src/cli.ts`
- `packages/agent/src/local-tools.ts`
- `packages/agent/src/loop.ts`
- `packages/agent/src/tool-calls.ts`
- `packages/agent/src/types.ts`
- `packages/transformers/src/chat-template.ts`
- `packages/transformers/src/interaction-profile.ts`
- `packages/transformers/src/infrastructure/sampling/runtime.ts`
- `examples/serve-completions/index.ts`

## Tensor Lifetime Audit

The protocol adapter, request-limit, model-listing, CLI, auth, and server files
are host-side JSON, SSE, and request normalization code. They create no native
tensors.

`packages/serve/src/transformers-engine.ts` delegates text prompts to the
existing `generateTextStream()` helper and token prompts to `generateTokens()`.
Message prompts are compiled through `InteractionProfile.compileMessages()`
before the same token-generation path. It does not take ownership of model,
tokenizer, or array handles. The test model uses the normal `using` disposal
pattern and the package adds no new native lifetime boundary.

`packages/serve/src/model-server.ts` owns model lifetime only for models loaded
through `serveModel()`. `serveLoadedModel()` treats caller-provided models as
borrowed unless `disposeModelOnStop` is set, and server shutdown is idempotent.

`packages/agent/src/*` is host-side chat-loop orchestration, OpenAI chat request
formatting, read-only local file tools, and tool-call parsing. It does not own
model tensors or native runtime handles.

`packages/transformers/src/infrastructure/sampling/runtime.ts` now evaluates the
sampled categorical token before local RNG key handles are released. That keeps
the stochastic token choice materialized at the same ownership boundary as the
temporary native key resources.

## Memory / Performance Evidence

Validated with:

- `bun test packages/serve/src`
- `bun test packages/agent/src`
- `bun test packages/transformers/src/chat-template.test.ts packages/transformers/src/interaction-profile.test.ts`
- `bun test packages/serve/src/cli.test.ts packages/agent/src/cli.test.ts packages/agent/src/loop.test.ts packages/agent/src/chat-model.test.ts packages/serve/src/protocols/openai-chat-completions.test.ts packages/serve/src/protocols/openai-completions.test.ts packages/serve/src/transformers-engine.test.ts packages/transformers/src/interaction-profile.test.ts packages/transformers/src/infrastructure/sampling/index.test.ts`
- `bun run --filter '@mlxts/serve' typecheck`
- `bun run --filter '@mlxts/agent' typecheck`
- `bun run --filter '@mlxts/transformers' typecheck`
- `bun run --filter '@mlxts/serve' build`
- `bun run --filter '@mlxts/agent' build`
- `bunx biome check packages/agent packages/serve/src examples/serve-completions`
- `bun run examples/serve-completions/index.ts`
- `bun run packages/serve/src/cli.ts --help`
- `bun run packages/agent/src/cli.ts --help`
- Interactive PTY smoke against cached `mlx-community/Qwen3.6-27B-4bit` on
  `@mlxts/serve`: `mlxts-agent` displayed a real `[tool call] list_files`,
  executed the read-only tool, displayed `[tool result]`, continued the loop,
  and printed a final `[assistant]` response.
- Multi-turn interactive PTY smoke against the same server: turn one listed
  `packages/agent/src/*.ts`, turn two read `packages/agent/src/types.ts` and
  answered using that tool observation while preserving the same REPL session.
- Interactive PTY smoke with thinking enabled against the same server:
  `mlxts-agent` displayed `[thinking]` separately from `[assistant]`.
- Real model smoke with cached `mlx-community/Qwen3.6-27B-4bit`:
  `@mlxts/serve` CLI served `/v1/models`, generated a 4-token
  completion, and rejected `max_tokens: 65` against a server cap of 64 with
  `max_tokens_exceeded`.
- `bun run typecheck`
- `bun run lint`
- `bun run check:coverage`
- `bun run check:runtime-review`
- `bun run check:tensor-lifetimes`
- `bun run check:assertions`
- `bun run check:file-lines`
- `bun run build`

The full coverage gate includes `@mlxts/serve` and reports:

- `@mlxts/serve coverage: 97.31% lines, 96.61% funcs`
- `@mlxts/agent coverage: 95.38% lines, 93.94% funcs`

This slice changes the stochastic sampling boundary only to materialize the
sampled token before local RNG key handles are freed. The serving shell keeps
streaming as an async event interface even though the first transformers engine
adapter still wraps the current synchronous generation helper. The four-agent
example verified that concurrent requests can route through one endpoint and
coalesce by model behind the micro-batching wrapper.

The package-owned model server verifies the load-and-serve operator path without
running a heavy model during validation: it exercises argument parsing, help
output, loaded-model serving, `/v1/models`, `/v1/chat/completions`, model
disposal, and option validation locally, then composes `loadCausalLM()`,
`loadPretrainedTokenizer()`, `loadInteractionProfile()`, the transformers
engine, request limits, model routing, and optional auth for real local use.

The package-owned agent loop verifies the "talk to it like an agent" path without
turning examples into product: it calls `/v1/chat/completions`, forwards tools in
OpenAI-compatible shape, executes local read-only tools, appends tool
observations, and continues until the model returns a final answer or the
iteration budget is reached. A live PTY run against Qwen3.6-27B-4bit confirmed
the real model can emit tool calls, execute `list_files` and `read_file`,
receive observations, preserve multi-turn REPL history, and continue to final
assistant answers. A separate thinking-enabled PTY run confirmed Qwen reasoning
is displayed separately from visible assistant content.

A real-model follow-up reproduced the first agent failure mode with
`mlx-community/Qwen3.6-27B-4bit`: sampled agent turns could run long enough for
Bun to terminate on a native C++ exception. The temporary stabilizer was
deterministic `temperature: 0`, but the final serving/agent default is
model-native: omitted sampling parameters stay omitted so checkpoint
`generation_config.json` can apply. Deterministic operation remains explicit via
`--greedy` / `--deterministic`.

## Independent Review

`Faraday` reviewed the intended serve package layering and recommended a single
canonical generation request with thin protocol adapters. That review also
flagged the main trap in the reference repos: useful serving seams, but too much
route-level duplication if copied directly.

`Huygens` reviewed the agent/tool split and flagged that `@mlxts/agent` should
not own a homemade completions prompt renderer. The final implementation routes
the loop through the package-owned chat-completions endpoint and leaves chat
template/tool formatting in `@mlxts/transformers` and `@mlxts/serve`.

`Kierkegaard` checked `.reference/text-generation-inference`,
`.reference/vllm-mlx`, `.reference/omlx`, and `.reference/mlx-lm` for first
serve-example ergonomics. That pass specifically called out `/v1/models`,
served-model naming, request limits, optional Bearer auth, localhost defaults,
and explicit non-streaming behavior as the right first slice.

The local implementation follows that recommendation by starting with
completions only while keeping the normalized request and engine contract
protocol-neutral.

## Remaining Risks / Follow-ups

- The first transformers engine accepts text, token, and chat-message prompts.
  Responses input items and Anthropic messages should continue to land as
  adapters over the same normalized request shape.
- `/v1/chat/completions` is non-streaming in this slice. Streaming chat deltas
  and structured OpenAI `tool_calls` output formatting remain follow-ups.
- The first agent package has read-only local file tools only. Shell tools,
  write tools, approvals, sandboxing policy, and durable conversation/session
  state should land deliberately rather than as example-only shortcuts.
- Streaming is shaped as async events, but the first real model adapter still
  depends on the synchronous `generateTextStream()` helper. A production serving
  engine should own cancellation, disconnect handling, and backpressure.
- Admission micro-batching is not continuous batching. A production engine still
  needs a decode scheduler, cache-aware request admission, cancellation, and
  per-token batching under the same `GenerationEngine` contract.
- The request-limit wrapper currently caps generated tokens only. Full context
  window validation should add prompt-token and total-token limits once the
  serving layer has a clean model-context source of truth.
- Prompt-cache lookup, richer memory budgeting, and model pool
  eviction/placement are still Phase 9 follow-ups.
