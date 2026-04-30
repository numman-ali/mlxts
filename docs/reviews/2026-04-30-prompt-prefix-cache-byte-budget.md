# Runtime Review: Prompt-prefix Cache Byte Budget

## Summary

This tranche adds byte-accounted retention to prompt-prefix cache snapshots.
`TransformerCacheSnapshot` now exposes `estimatedByteSize`, generic managed
caches and Qwen hybrid caches compute it from retained snapshot tensors, and
`@mlxts/serve` can bound retained prompt-prefix snapshots with
`promptPrefixCacheMaxBytes` / `--prompt-prefix-cache-max-bytes`.

The cache still keeps the existing entry-count policy and evicts shorter
snapshots before LRU ties. Oversized single snapshots are disposed immediately
instead of being retained.

## Files Reviewed

- `packages/serve/src/cli-options.ts`
- `packages/serve/src/cli.ts`
- `packages/serve/src/engine/index.ts`
- `packages/serve/src/engine/prefix-cache-entry.ts`
- `packages/serve/src/engine/prefix-cache.ts`
- `packages/serve/src/http/route-info.ts`
- `packages/serve/src/model-loading/server-options.ts`
- `packages/serve/src/model-loading/server.ts`
- `packages/serve/src/model-loading/sources.ts`
- `packages/serve/src/runtime/strategy.ts`
- `packages/transformers/src/families/qwen3_5/cache/index.ts`
- `packages/transformers/src/infrastructure/cache/runtime.ts`
- `packages/transformers/src/infrastructure/cache/single.ts`
- `packages/transformers/src/types.ts`

## Tensor Lifetime Audit

The byte accounting reads `MxArray.nbytes` from already-owned snapshot tensors.
It does not allocate new MLX tensors, retain additional arrays, or move tensor
creation into nested calls.

Generic managed cache snapshots sum key/value state bytes per layer. Qwen
hybrid snapshots sum full-attention keys/values plus linear-attention
conv/recurrent state arrays. Snapshot fork and disposal ownership is unchanged.

`PromptPrefixCache.store()` disposes snapshots that cannot be retained because
the entry limit is disabled, the snapshot offset is unusable, or the snapshot
exceeds the configured byte budget. Eviction releases the block-index entry,
subtracts retained bytes, disposes the snapshot, and releases retained token
blocks.

## Memory / Performance Evidence

Focused tests passed:

```bash
bun test packages/serve/src/engine/prefix-cache.test.ts packages/serve/src/runtime/strategy.test.ts packages/serve/src/cli.test.ts packages/serve/src/model-loading/server.test.ts packages/serve/src/http/server.test.ts packages/transformers/src/infrastructure/cache/index.test.ts packages/transformers/src/families/qwen3_5/cache/cache.test.ts
```

Result: `113 pass`, `0 fail`.

Typecheck passed:

```bash
bun run typecheck
```

Result: all workspaces exited with code `0`.

Full validation passed:

```bash
bun run validate
```

Result: typecheck, lint, assertions, file-lines, per-package agents,
cross-package imports, tensor lifetimes, runtime review, and coverage all
passed.

Quick Qwen/Gemma regression passed:

```bash
bun run regression:qwen-gemma -- --profile quick
```

Result: transformer focused regressions `84 pass`, `0 fail`; serve focused
regressions `205 pass`, `0 fail`.

`bench:generation` and `bench:generation:parity` were not rerun for this
tranche because model forward, decode scheduling, attention kernels, sampling,
cache append/fetch tensor layout, and cache fork semantics are unchanged. The
new work is byte metadata computed at snapshot construction time and a serve
retention policy that runs only when a prompt-boundary snapshot is stored or
evicted.

## Independent Review

Hegel, a GPT-5.5 xhigh sub-agent, recommended this tranche shape before
implementation: add snapshot byte accounting to the transformer cache contract,
track retained bytes in `PromptPrefixCache`, expose `promptPrefixCacheMaxBytes`
through serve runtime options and CLI, evict after store using the existing
shorter-prefix then LRU policy, and dispose over-budget single snapshots.

Hegel also reviewed the current diff. The only finding was a low-risk ordering
issue where byte-budget rejection could hide an invalid snapshot offset; the
store path now validates offset bounds before applying the byte-budget policy,
with direct test coverage.

## Out-of-scope Drift Noticed

None.

## Remaining Risks / Follow-ups

`estimatedByteSize` is logical retained tensor bytes, not allocator overhead.
Paged KV, tensor block deduplication, SSD spill, and quantized KV retention
remain later Phase 9 work.
