# OpenResponses Function Tools

## Summary

Added bounded non-streaming function-tool support to the OpenResponses `/v1/responses` adapter. The endpoint now accepts flat Responses function tools, maps active tools into the protocol-neutral `ChatTool` surface, normalizes `function_call` / `function_call_output` history items into assistant/tool turns, and formats generated tool-call envelopes back as Responses `function_call` output items.

Streaming requests with active tools are rejected before generation until the Responses function-call SSE events are implemented.

## Files Reviewed

- `packages/serve/src/protocols/openai-responses.ts`
- `packages/serve/src/protocols/openai-responses-input.ts`
- `packages/serve/src/protocols/openai-responses-turns.ts`
- `packages/serve/src/protocols/openai-responses-formatting.ts`
- `packages/serve/src/protocols/openai-responses-tools.ts`
- `packages/serve/src/protocols/openai-responses.test.ts`
- `packages/serve/src/http/server.test.ts`
- `packages/serve/src/streaming/writer-openai-responses.ts`
- `packages/transformers/src/chat-template.ts`

## Reference Check

- OpenResponses function tools are flat `tools` entries with `type: "function"`, `name`, optional `description`, optional JSON-schema `parameters`, and optional `strict`.
- OpenResponses model output can include `function_call` output items with `call_id`, `name`, and JSON-string `arguments`.
- OpenResponses input history can include `function_call_output` items keyed by the prior `call_id`.
- Reasoning items returned beside tool calls must be included in the next input turn with the tool outputs so reasoning-model tool loops preserve state.
- Responses streaming function calls have a distinct event sequence for function-call argument deltas and completion. This tranche does not implement that stream writer path.

References:

- https://platform.openai.com/docs/guides/function-calling?api-mode=responses
- `.reference/vllm-mlx/vllm_mlx/api/responses_models.py`
- `.reference/vllm-mlx/vllm_mlx/server.py`
- `.reference/omlx/omlx/api/responses_models.py`
- `.reference/omlx/omlx/api/responses_utils.py`

## Tensor Lifetime Audit

The changed code normalizes JSON protocol shapes and formats JSON response objects only. It allocates no `MxArray` handles, creates no transformer caches, and does not alter scheduler, cache, model execution, or media preprocessing paths.

Streaming tools are rejected during request normalization before any generation engine call, so `writer-openai-responses.ts` remains text/reasoning-only for this tranche.

## Memory / Performance Evidence

- `bun test packages/serve/src/protocols/openai-responses.test.ts packages/serve/src/http/server.test.ts` passed: `56 pass`, `324 expect()`.
- `bun run --filter '@mlxts/serve' typecheck` passed.
- `bun run lint` passed.
- `bun run check:file-lines` passed.
- `bun run check:runtime-review` passed.
- `bun run check:assertions && bun run check:per-package-agents && bun run check:cross-package-imports && bun run check:tensor-lifetimes` passed.
- `bun run check:coverage` passed.
- `bun run validate` passed.

No model hot path, tensor lifetime, cache route, or streaming writer loop changed. There are no performance claims beyond preserving the existing non-tool and text-streaming paths.

## Independent Review

Bohr reviewed the uncommitted tranche for runtime/API correctness and test gaps. The review called out three items: `parallel_tool_calls: false` with active tools, reasoning-plus-tool-call round-trip behavior, and explicit scope for string-only tool outputs. This tranche resolves the first by rejecting `parallel_tool_calls: false` when active function tools are selected, resolves the second by accepting reasoning items immediately before assistant output items, and covers the third with explicit rejection coverage for non-string function outputs.

## Guardrails

- Generated tool-call extraction runs only when active tools are present. Tool-call-looking text remains text when no tools are active or `tool_choice: "none"` suppresses tools.
- `tool_choice: "none"` keeps the wire-level tool list echo but does not expose active tools to the chat-template path.
- `parallel_tool_calls: false` is rejected when active tools are present. This endpoint does not silently promise single-call enforcement while using a model-native generation path.
- Streaming with active tools is rejected until `writer-openai-responses.ts` owns the OpenResponses function-call SSE event sequence.
- Reasoning input items are accepted only when they immediately precede assistant output items; reasoning plus function-call outputs round-trip into one assistant message with `reasoning_content` and `tool_calls`.
- Function-call history requires the immediately following `function_call_output` set to match the preceding call IDs exactly, with no missing or duplicated outputs.
- Built-in tools, custom tool types, forced tool choice objects, standalone or misordered reasoning input items, stateful `previous_response_id` / `conversation`, and rich non-string tool outputs remain rejected.

## Remaining Risks / Follow-ups

- Implement Responses streaming tool-call events before accepting `stream: true` with active tools.
- Add built-in OpenResponses tools only after the serve-side execution contract is designed.
- Forced tool choice and stateful response continuation need separate protocol and engine review.
- Rich function output content remains a future transport widening.

## Out-of-scope Drift Noticed

None.
