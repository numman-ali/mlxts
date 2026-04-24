# Runtime Review: Serve Memory Preflight

## Summary

This change adds a machine-aware serving admission guard. `@mlxts/serve` now
accepts `gpuMemoryUtilization` / `--gpu-memory-utilization`, reports it through
`/info`, and checks estimated request-local memory before cached prefill or
decode begins.

The preflight combines current MLX allocator active memory with a model-config
estimate for KV cache, fixed recurrent/conv cache state, and large prefill SDPA
temporary memory. It is intentionally conservative and best-effort: if a family
config cannot be interpreted, the guard skips rather than pretending to know.

## Files Reviewed

- `packages/serve/src/cli-options.ts`
- `packages/serve/src/cli.ts`
- `packages/serve/src/index.ts`
- `packages/serve/src/model-context.ts`
- `packages/serve/src/model-server.ts`
- `packages/serve/src/model-sources.ts`
- `packages/serve/src/server-info.ts`
- `packages/serve/src/transformers-engine-generation.ts`
- `packages/serve/src/transformers-engine-shared.ts`
- `packages/serve/src/transformers-engine.ts`

## Tensor Lifetime Audit

The new estimator and admission guard are host-side only. They read
`model.config.rawConfig`, numeric request metadata, and MLX allocator telemetry;
they do not allocate, retain, or dispose `MxArray` values.

`enforceGenerationMemoryBudget()` runs after prompt compilation/token counting
and after prompt/total token admission, but before `emitGenerationProgress()`,
`generateTextStream()`, `generateTokens()`, `generatePreparedTokenEvents()`, or
static batch generation. A rejected request exits before model forward, cache
mutation, or prefill chunking starts.

The static batch path gets an aggregate guard for the grouped batch before
`generateBatchTokens()`. Non-static Qwen and Gemma paths are still admitted as
single requests because those models fall back to the single-request engine
until a scheduler/cache tranche owns them.

## Memory / Performance Evidence

Validated with:

- `bun test packages/serve/src/model-context.test.ts packages/serve/src/transformers-engine.test.ts packages/serve/src/model-server.test.ts packages/serve/src/server.test.ts packages/serve/src/cli.test.ts packages/serve/src/model-sources.test.ts`
- `bun run typecheck`
- `bun run lint`

The tests cover Qwen hybrid memory geometry, Gemma sliding/full/global-head KV
geometry, direct memory-budget rejection, CLI parsing/ready output, and `/info`
limit reporting.

No generation benchmark is required for this slice because model math, sampling,
KV update logic, and decode loops are unchanged. The change is an admission
guard that stops unsafe work before those paths start.

## Independent Review

Sub-agent Beauvoir completed a read-only audit of the proposed slice. The review
confirmed the right insertion points in the transformer-backed non-streaming and
streaming engines, recommended a vLLM/oMLX-style utilization budget over a
transformer contract change, and caught estimator gaps for Qwen fixed
linear-attention state plus Gemma sliding/full/global-head cache geometry.

The implementation follows that recommendation and records the package-local
guidance in `packages/serve/AGENTS.md`.

## Remaining Risks / Follow-ups

This is still not a continuous batching scheduler. Admission happens before a
request starts; it does not yet reserve memory across several in-flight jobs,
prefix-cache reuse, or future Qwen-aware token-level batching.

The prefill temporary estimate is intentionally coarse and conservative for
large non-fused SDPA paths. Future scheduler work should move from per-request
guardrails to explicit memory reservation, chunked prefill scheduling, and
backpressure/cancellation.
