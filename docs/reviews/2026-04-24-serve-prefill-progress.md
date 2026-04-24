# Runtime Review: Serve Prefill Progress

## Summary

This change adds prompt-prefill progress telemetry to the generation path and
surfaces it through `@mlxts/serve` logs. Long-context requests can now show
chunked `prefill_tokens=x/y` progress before first-token decode, which addresses
the silent multi-minute prefill behavior observed in the Qwen 64k/128k ladder.

The callback is observational only. It does not change model math, cache shape,
sampling, batching eligibility, or tensor ownership.

## Files Reviewed

- `packages/transformers/src/types.ts`
- `packages/transformers/src/index.ts`
- `packages/transformers/src/infrastructure/generation/helpers.ts`
- `packages/transformers/src/infrastructure/generation/runtime.ts`
- `packages/transformers/src/infrastructure/generation/runtime-streaming.ts`
- `packages/serve/src/types.ts`
- `packages/serve/src/cli.ts`
- `packages/serve/src/transformers-engine.ts`
- `packages/serve/src/transformers-engine-generation.ts`
- `packages/serve/src/transformers-engine-shared.ts`

## Tensor Lifetime Audit

`prefillPromptCache()` still owns each chunk input tensor through `using
chunkIds`, frees optional sliced prompt embeddings and position ids in the
existing `finally`, and materializes cache state through the existing
`materializeCacheState()` boundary. The new progress callback runs only after
the chunk forward/cache materialization finishes and before `clearMemoryCache()`.
It receives numbers only, so it cannot retain `MxArray` handles.

`generateWithCache()` and `streamWithCache()` pass the resolved
`onPrefillProgress` callback into `prefillPromptCache()`. They do not add new
arrays or alter current token/next token disposal.

The serve-side reporter reads allocator telemetry through
`readGenerationMemoryUsage()` and emits host-side event objects only. It does
not own model, cache, tokenizer, or tensor handles.

## Memory / Performance Evidence

Validated with:

- `bun test packages/transformers/src/generation.test.ts packages/serve/src/transformers-engine.test.ts packages/serve/src/cli.test.ts`
- `bun run typecheck`
- No new `bench:generation` or `bench:generation:parity` run was taken for
  this telemetry-only slice. The preceding Qwen parity commit already records
  the relevant generation performance evidence; this change adds host-side
  progress events after existing prefill chunks and does not alter decode math,
  sampling, cache representation, or benchmark target behavior.

The callback fires once per cached prefill chunk. It should add negligible
overhead relative to multi-thousand-token prefill chunks and gives operators
visibility into exactly the long-running phase that previously looked like a
dead request.

## Independent Review

A read-only implementation review from sub-agent Leibniz found three issues.
The streaming prefill was initially outside the cleanup `try/finally`; this was
fixed so a prefill or callback error still disposes the sampler, generation
stream, wired-limit scope, and any retained prompt arrays. `BatchGenerationOptions`
initially exposed `onPrefillProgress` even though static batch generation has no
batch-aware prefill callback; the option is now omitted from batch options and
static batches continue to report batch start/completion only. The reviewer also
noted unnecessary memory telemetry reads when no serve event sink exists; the
progress emitters now return before reading allocator telemetry if `onEvent` is
absent.

## Remaining Risks / Follow-ups

This is observability, not continuous batching. Qwen hybrid-cache batching,
prefix-cache reuse, admission memory preflight, cancellation for non-streaming
requests, and multi-model memory management remain separate serving-quality
tranches.
