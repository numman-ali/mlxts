# Serve Prefix Cache Retention Evidence

## Files Reviewed

- `packages/serve/scripts/benchmark-serve.ts`
- `packages/serve/scripts/benchmark-serve.test.ts`
- `packages/serve/scripts/regression-agent-cache.ts`
- `packages/serve/scripts/regression-serve-matrix.test.ts`
- `packages/serve/src/admission/request-limits.ts`
- `packages/serve/src/admission/request-limits.test.ts`
- `packages/serve/src/cli.ts`
- `packages/serve/src/engine/engine.test.ts`
- `packages/serve/src/engine/generation.ts`
- `packages/serve/src/engine/index.ts`
- `packages/serve/src/engine/prefix-cache-info.ts`
- `packages/serve/src/engine/prefix-cache.ts`
- `packages/serve/src/engine/prefix-cache.test.ts`
- `packages/serve/src/http/route-info.ts`
- `packages/serve/src/http/server.test.ts`
- `packages/serve/src/index.ts`
- `packages/serve/src/model-loading/pool.ts`
- `packages/serve/src/model-loading/pool.test.ts`
- `packages/serve/src/model-loading/router.ts`
- `packages/serve/src/model-loading/router.test.ts`
- `packages/serve/src/model-loading/server.ts`
- `packages/serve/src/model-loading/server.test.ts`
- `packages/serve/src/observability/cli-status-command.ts`
- `packages/serve/src/observability/cli-status-command.test.ts`
- `packages/serve/src/observability/cli-status-prompt-cache.ts`
- `packages/serve/src/observability/cli-status-runtime.ts`
- `packages/serve/src/prompt-cache-observability.ts`
- `packages/serve/src/types.ts`

## Summary

This tranche exposes existing prompt-prefix cache retention evidence without changing cache matching, snapshot creation, cache forking, or eviction semantics. Generation prompt-cache events now carry retained snapshot counters, shared token-block counters, and hit source metadata when a retained snapshot is reused. `/info`, `mlxts-serve status`, `bench:serve`, and the agent-cache regression report can now show whether a server has retained cache state that future requests can hit.

## Tensor Lifetime Audit

The cache lookup/store path remains structurally unchanged. `PromptPrefixCache` still selects hits, forks snapshots, stores snapshots, deduplicates token blocks, and evicts overflow with the same rules. The only new data flow is host-side observation: event details read `PromptPrefixCache.stats()` and already-built hit metadata, then protocol-neutral observability surfaces render those counters.

No tensor-producing expressions, MLX array ownership, cache tensor layout, scheduler admission, or generation sampling logic changed. `packages/serve/src/engine/prefix-cache-info.ts` aggregates numeric counters only.

## Memory / Performance Evidence

The change adds synchronous host-side reads of existing cache counters at miss, hit, and write event boundaries. It does not add model forwards, cache tensor clones, snapshot forks beyond the existing hit path, or retained tensor state. The focused tests verify the new counters and status surfaces using synthetic caches and in-process serve engines.

## Evidence

- `bun test packages/serve/src/engine/prefix-cache.test.ts packages/serve/src/engine/engine.test.ts packages/serve/src/model-loading/router.test.ts packages/serve/src/model-loading/pool.test.ts packages/serve/src/admission/request-limits.test.ts packages/serve/src/http/server.test.ts packages/serve/src/observability/cli-status-command.test.ts packages/serve/scripts/benchmark-serve.test.ts packages/serve/scripts/regression-serve-matrix.test.ts`
- `bun run --filter '@mlxts/serve' typecheck`
- `bun run check:file-lines`
- `bun run check:tensor-lifetimes`
- `bun run check:runtime-review`
- `bun run validate`

## Independent Review

The prior read-only sub-agent pass identified this as the highest-leverage safe follow-up after the cache QA rerun: expose retained snapshot and token-block state in operator-facing surfaces, while treating `packages/transformers/src/infrastructure/cache/tensor-block-snapshot.ts` as review-only unless counters revealed a cache bug. No transformer cache bug was found or touched.

## Remaining Risks / Follow-ups

These counters prove retained cache state and hit source shape; they do not claim arbitrary shared-prefix reuse for exact-boundary Qwen hybrid or Gemma layer-pattern caches. Cold concurrent requests can still both miss when no completed retained snapshot exists. Paged KV, SSD KV, TurboQuant KV, speculative decode, and broader cache-backend work remain separate Phase 9 work.

## Out-of-scope Drift Noticed

None.
