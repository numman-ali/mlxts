# Block-Indexed Prefix Cache Lookup

## Summary

This tranche uses the serve-owned prompt-token block chain to narrow
prompt-prefix cache lookup candidates before running the existing LCP and
snapshot-fork checks. The index is a host-side candidate filter only:
`TransformerCacheSnapshot.canFork()` and `fork()` remain the authority for
restoring Qwen hybrid, Gemma layer-pattern, and full-KV cache state.

This is not paged attention, tensor-block deduplication, or a new cache tensor
backend. The retained block store still owns token ids, parent-linked hashes,
reference counts, and stats.

## Files Reviewed

- `packages/serve/src/engine/prefix-cache.ts`
- `packages/serve/src/engine/prefix-cache-blocks.ts`
- `packages/serve/src/engine/prefix-cache-index.ts`

## Runtime Sensitivity Notes

`prefix-cache.ts` is runtime-sensitive because prompt-prefix hits can skip
prompt prefill by restoring a family-owned transformer cache fork. The new
block index changes only which retained entries are inspected first. Hit
selection still computes token LCP, checks media identity, and calls
`snapshot.canFork({ offset })` before any cache is returned.

When the index cannot prove complete coverage, lookup falls back to scanning
all retained entries. This preserves shorter-entry fallback when the deepest
block-sharing entry cannot fork and preserves media identity fallback when a
media-aware entry shares tokens with a text-only request.

## Tensor Lifetime Audit

The changed code retains no tensors. `PromptPrefixCacheBlockIndex` stores only
entry references grouped by token-block hash. Entry disposal removes the entry
from index buckets before releasing the snapshot and token-block handle.

Snapshot ownership remains unchanged: retained entries own snapshots,
`lookup()` returns a forked cache owned by the caller, and
`PromptPrefixCacheSession` disposes unconsumed forks.

## Memory / Performance Evidence

- `bun test packages/serve/src/engine/prefix-cache-blocks.test.ts packages/serve/src/engine/prefix-cache-index.test.ts packages/serve/src/engine/prefix-cache.test.ts`: passed, `21` tests.
- `bun run typecheck`: passed.
- `bun run lint`: passed.
- `bun run check:file-lines`: passed.
- `bench:generation`: not rerun; no model math, tensor kernels, decode loop, or attention/cache tensor layout changed.
- `bench:generation:parity`: not rerun for the same reason. Real Qwen/Gemma cache restore behavior remains guarded by the previous continuous-prefix seeding regression evidence.

## Independent Review

Pascal reviewed the block-indexed lookup tranche before commit. The review
flagged stale runtime-review coverage, a small-cache CPU regression, and a
missing direct test for the coverage-proof fallback. The artifact now validates
through `check:runtime-review`, single-entry caches skip block hashing, and the
focused tests include a forced-hash-collision regression proving full-scan
fallback can replace an under-covering indexed hit.

## Out-of-scope Drift Noticed

None.

## Remaining Risks / Follow-ups

The index narrows host-side candidate scanning but does not deduplicate cache
tensors or make scheduler restore batch-native. The next cache backend tranche
should decide whether memory-aware prefix eviction or paged/block cache tensor
reuse is the right next serving step, then pair it with endpoint evidence.
