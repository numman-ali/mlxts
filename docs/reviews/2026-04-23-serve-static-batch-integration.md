# Runtime Review: Serve Static Batch Integration

## Summary

Wired the transformer-backed serving engine into the static batch generation
primitive from the previous tranche. `createTransformersGenerationEngine()` now
exposes `generateBatch()` and the existing admission micro-batcher can feed real
fixed batches to full-cache LLaMA-like models instead of only coalescing into
sequential single generations.

This remains intentionally narrower than continuous batching. The serving engine
only uses static batching for non-streaming, greedy, full-cache model requests
with compatible generation options. Sampled/model-native-default requests,
mixed generation lengths, streaming, and unsupported cache families fall back to
the existing single-request path.

The same tranche also makes static batching observable through a
`generation_batch_start` serve event and default CLI log line, so local operators
can see when requests actually coalesced.

## Files Reviewed

- `packages/serve/src/transformers-engine.ts`
- `packages/serve/src/transformers-engine-generation.ts`
- `packages/serve/src/transformers-engine-shared.ts`
- `packages/serve/src/types.ts`
- `packages/serve/src/cli.ts`

## Tensor Lifetime Audit

`packages/serve/src/transformers-engine.ts` does not own native tensors directly.
The new batch path prepares host-side request metadata and delegates tensor/cache
ownership to `generateBatchTokens()`, which owns its internal `BatchKVCache` and
sampled token tensors.

The integration avoids swallowing `generateBatchTokens()` runtime errors after
eligibility checks; unsupported request shapes are routed to single generation
before entering the batch primitive. Text/token/message prompt compilation stays
host-side, and model/tokenizer ownership remains with the caller or
`serveModel()` as before.

## Memory / Performance Evidence

Validated with:

- `bun test packages/serve/src/transformers-engine.test.ts packages/serve/src/model-server.test.ts`
- `bun test packages/serve/src/transformers-engine.test.ts packages/serve/src/cli.test.ts`
- `bun test packages/serve/src`
- `bun run typecheck`

The tests prove:

- eligible greedy full-cache requests use the transformer static batch path
- concurrent non-streaming HTTP completions coalesce through `serveLoadedModel()`
- per-row stop handling and Qwen-style prompt-open reasoning post-processing are
  preserved after batch decoding
- mixed `max_tokens`, explicit sampled requests, and model-default sampled
  requests fall back instead of entering static batching
- text prompt usage counts the tokenization mode actually used for generation
- static batch start events are emitted by the transformer engine and formatted
  by the CLI logs

No new generation hot-path benchmark was required by the runtime-review gate for
this tranche because only the serving adapter changed; the lower transformer
batch primitive was benchmarked in
`docs/reviews/2026-04-23-static-batch-generation.md`.

## Independent Review

Parfit independently audited the serving integration plan against
`@mlxts/serve` and the local reference repos. That review confirmed the right
boundary: add a real `generateBatch()` implementation inside
`packages/serve/src/transformers-engine.ts`, let the existing micro-batcher feed
it, keep streaming separate, and avoid calling this continuous batching.

The implementation follows those constraints and keeps the deeper scheduler,
active-row decode loop, cancellation, and streaming backpressure as future
engine-owned work rather than another HTTP wrapper.

## Remaining Risks / Follow-ups

Static batching currently covers only full-cache LLaMA-like model paths and
greedy decoding. Qwen hybrid caches, Gemma 3/4 layer-pattern caches,
sliding-window batch caches, sampled batched decode, streaming batch output, and
true continuous token-level scheduling remain separate tranches.

The next serving quality step is a scheduler-owned engine that can manage active
requests, cancellation, queue state, and per-token output collection behind the
same `GenerationEngine` contract.
