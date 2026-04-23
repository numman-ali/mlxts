# Runtime Review: Serve Loaded Models

## Summary

Added `serveLoadedModels()` as a first-class package API for exposing multiple
already-loaded local models behind one OpenAI-compatible endpoint. The existing
single-model `serveLoadedModel()` path now delegates through the same
multi-model implementation, so both surfaces share routing, request limits,
concurrency control, and micro-batching behavior.

Each served model id gets its own transformer generation engine, request-limit
wrapper, concurrency gate, and admission micro-batch queue before requests meet
the shared model router. This keeps model runtime/cache ownership separate while
allowing one process to advertise and route models such as Gemma and Qwen from
the same `/v1/models`, `/v1/completions`, and `/v1/chat/completions` endpoint.

## Files Reviewed

- `packages/serve/src/model-server.ts`
- `packages/serve/src/index.ts`

## Tensor Lifetime Audit

`packages/serve/src/model-server.ts` does not allocate native tensors directly.
The change composes existing serving engines and keeps tensor ownership inside
`createTransformersGenerationEngine()` and `@mlxts/transformers` generation
helpers.

Model disposal remains explicit. Borrowed single-model serving still defaults to
no disposal, `serveLoadedModel({ disposeModelOnStop: true })` disposes one model,
and `serveLoadedModels({ disposeModelsOnStop: true })` disposes each model once
when the server stops.

## Memory / Performance Evidence

Validated with:

- `bun test packages/serve/src/model-server.test.ts`
- `bun test packages/serve/src/model-server.test.ts packages/serve/src/cli.test.ts`
- `bun run typecheck`

The tests prove:

- `serveLoadedModel()` still serves one model and reports `modelIds`
- `serveLoadedModels()` advertises multiple model ids through `/v1/models`
- completions route to the requested model id and return `404` for unknown ids
- stopping an owned multi-model server disposes each loaded model exactly once
- existing CLI test doubles conform to the returned server handle shape

No generation benchmark is required for this tranche because the change is a
serving composition/API layer and does not modify transformer generation hot
paths.

## Independent Review

Ptolemy independently audited the current serving and batching posture against
the local `mlx-lm`, `vllm-mlx`, and `omlx` references. That review confirmed the
existing router/micro-batching split and recommended keeping multi-model routing
as a serving-layer concern while reserving deeper throughput work for a
scheduler-shaped batch primitive.

The implementation follows that boundary: multi-model serving composes existing
per-model engines instead of sharing one cache/runtime state across model ids or
pretending this is continuous batching.

## Remaining Risks / Follow-ups

This API serves multiple loaded models in one process, but it does not solve
memory pressure from loading several large checkpoints at once. Operators should
still choose model sizes/quantization that fit local Apple Silicon memory.

The next serving-quality tranche is static batch decode v2: per-row generation
lengths and token-event output for the existing greedy full-cache batch path,
followed later by streaming over that primitive, Gemma layer-pattern batch cache
support, and Qwen hybrid batching.
