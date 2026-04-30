# Model Pool Observability

## Summary

Added operator-visible lazy model-pool status to `/info` and the finite
`mlxts-serve status` command. The status surface reports the lazy load policy,
pressure policy, release timeout, idle TTL, per-model loaded/loading state,
pinned state, active requests, and pressure-aborted requests.

Extended the real lazy-pool pressure smoke with staged MLX allocator snapshots.
The smoke resets peak memory at start and records memory at start, after the
active stream first chunk, after the first pressure event, after blocked-request
completion, and after server stop.

This is observability only. It does not change lazy loading, eviction, pressure
victim selection, retry behavior, scheduling, or memory-budget policy.

## Files Reviewed

- `packages/serve/src/types.ts`
- `packages/serve/src/model-loading/pool-info.ts`
- `packages/serve/src/model-loading/pool-routing.ts`
- `packages/serve/src/model-loading/pool.ts`
- `packages/serve/src/model-loading/pool.test.ts`
- `packages/serve/src/model-loading/sources.test.ts`
- `packages/serve/src/http/route-info.ts`
- `packages/serve/src/http/server.test.ts`
- `packages/serve/src/observability/cli-status-model-pool.ts`
- `packages/serve/src/observability/cli-status-command.ts`
- `packages/serve/src/observability/cli-status-command.test.ts`
- `packages/serve/scripts/regression-lazy-pool-pressure.ts`
- `packages/serve/scripts/regression-lazy-pool-pressure.test.ts`

## Tensor Lifetime Audit

The production changes expose existing lazy-pool state and do not create,
retain, or dispose `MxArray` values. The regression harness reads allocator
telemetry through `readGenerationMemoryUsage()` and does not touch tensor
handles directly.

## Memory / Performance Evidence

- `bun test packages/serve/src/model-loading/pool.test.ts packages/serve/src/model-loading/sources.test.ts packages/serve/src/http/server.test.ts packages/serve/src/observability/cli-status-command.test.ts packages/serve/scripts/regression-lazy-pool-pressure.test.ts`:
  passed, `91` tests / `437` assertions.
- `bun run --filter '@mlxts/serve' typecheck`: passed.
- `bun test packages/serve/scripts/regression-lazy-pool-pressure.test.ts`:
  passed, `5` tests / `15` assertions.
- `bun run check:file-lines`: passed, `353` production source files checked.
- `bun run check:runtime-review`: passed, artifact accepted for the
  runtime-sensitive diff.
- `bun run regression:lazy-pool-pressure -- --report-dir .tmp/lazy-pool-pressure-real`:
  passed. The real smoke loaded `google/gemma-4-E2B-it`, pressure-loaded
  `mlx-community/Qwen3.6-27B-4bit`, recorded `2` pressure events, aborted `1`
  active request, completed the blocked request with `35` output chars, and
  emitted `6` pressure metric lines.
- The real smoke wrote staged memory snapshots at `start`,
  `after_active_first_chunk`, `after_pressure_event`,
  `after_blocked_completion`, and `after_server_stop`. Active bytes moved from
  `0` at start to `9301115032` during the active Gemma stream, then to
  `16224974180` after blocked Qwen completion, and back to `60` after server
  stop. Peak bytes reached `16429904628` under a `65283502899` byte memory
  limit.
- `bun run validate`: passed.

## Independent Review

Rawls reviewed MLX, MLX-C, oMLX, vLLM-MLX, and Rapid-MLX references. The review
found no missing allocator-control blocker and specifically recommended staged
allocator telemetry plus pool-status introspection before any further policy
changes. This tranche implements that recommendation without using MLX memory
limits as dynamic pressure relief.

## Remaining Risks / Follow-ups

The `/info` pool state is a point-in-time snapshot. Active request counts and
loading state can change immediately after the response is formatted.

The real pressure smoke still records current streaming cancellation behavior,
including the Bun-visible stderr stack on pressure-cancelled streams. Stream
terminal-shape polish remains a separate product tranche.

## Out-of-scope Drift Noticed

- Full MLX device-info exposure is still only partial through the existing
  recommended working-set helper.
- Richer victim scoring, process-style background enforcement, paged KV, and
  quantized KV are not part of this tranche.
