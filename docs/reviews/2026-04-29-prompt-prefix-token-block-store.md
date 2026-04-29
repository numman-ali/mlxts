# Prompt Prefix Token Block Store

## Summary

This tranche adds serve-owned token-block retention under the existing
prompt-prefix cache. Cache entries now retain a parent-linked chain of prompt
token blocks, with ref counts, deduplicated block ownership, and retention
stats. Lookup, generation suffixing, usage accounting, and cache tensor restore
still use the existing snapshot path: `TransformerCacheSnapshot.canFork()` and
`fork()`.

This is not paged attention and not block-indexed lookup. It is the block
ownership foundation for the next prefix-cache backend tranche.

## Files Reviewed

- `packages/serve/src/engine/prefix-cache.ts`
- `packages/serve/src/engine/prefix-cache-blocks.ts`

## Runtime Sensitivity Notes

The changed code is runtime-sensitive because prompt-prefix cache hits can skip
prompt prefill work by restoring a transformer cache fork. This tranche does
not change hit selection authority or model execution. Serve still scans
retained prompt entries, computes token LCP, checks media identity, and asks the
family-owned snapshot whether a requested offset can fork.

The token-block store owns only token ids, hashes, reference counts, and stats.
It does not expose or retain KV tensors, Qwen recurrent state, Gemma sliding
state, RoPE deltas, prepared embeddings, or position ids.

## Tensor Lifetime Audit

Retained token blocks are host-side metadata only. Snapshot ownership remains
unchanged: retained entries own snapshots, `lookup()` returns a forked cache
owned by the caller, and `PromptPrefixCacheSession` disposes unconsumed forks.

Entry disposal releases snapshot ownership and token-block ownership together.
Eviction, disabled caches, invalid snapshot offsets, cache disposal, and failed
block retain paths leave token-block reference counts balanced. Hash collision
detection rolls back blocks retained earlier in the failed chain before
throwing.

## Memory / Performance Evidence

- `bun test packages/serve/src/engine/prefix-cache-blocks.test.ts packages/serve/src/engine/prefix-cache.test.ts`: passed, `16` tests.
- `bun run typecheck`: passed.
- `bun run lint`: passed.
- `bun run check:file-lines`: passed.
- `bench:generation`: not rerun; no model math, tensor kernels, decode loop, or attention/cache tensor layout changed.
- `bench:generation:parity`: not rerun for the same reason. Real Qwen/Gemma cache restore behavior remains guarded by the previous continuous-prefix seeding regression evidence.

## Independent Review

Hilbert reviewed the WIP design and recommended keeping this tranche scoped to
block ownership and stats, not block-indexed lookup. The review also called out
transactional rollback on hash collision and tests for boundary behavior,
fallback after `canFork()` rejection, media identity shadowing, lifecycle
accounting, and usage invariants. The current diff adds those rollback semantics
and focused tests.

## Out-of-scope Drift Noticed

None.

## Remaining Risks / Follow-ups

Lookup still scans retained entries and compares token arrays. The next cache
backend tranche should add a block candidate index for longest-prefix lookup,
memory-aware eviction, and endpoint evidence that divergent repeated chats
reuse common prefixes without changing usage accounting or scheduler fairness.
