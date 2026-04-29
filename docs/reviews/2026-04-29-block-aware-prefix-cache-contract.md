# Block-Aware Prefix Cache Contract

## Summary

This tranche makes prompt-prefix cache hits describe their reuse shape as
`exact`, `prefix`, `supersequence`, or `lcp`, and attaches source snapshot
metadata for layer kinds and trimmability. The serving cache still restores
only through `TransformerCacheSnapshot.canFork()` and `fork()`; no paged
attention backend, block table, or model hot path changes land here.

The purpose is to give the next block-deduplicated prefix store a stable,
testable contract without changing endpoint behavior or token accounting.

## Files Reviewed

- `packages/serve/src/engine/prefix-cache.ts`

## Runtime Sensitivity Notes

`prefix-cache.ts` is runtime-sensitive because it decides whether request
prefill can be skipped by restoring an owned transformer cache fork. This
change does not widen the restore authority: `snapshot.canFork({ offset })`
continues to be the only gate for exact continuation, shorter-source reuse,
longer-source trimming, and LCP reuse.

Serve records the source snapshot's `layerKinds` and `trimmable` value for the
hit. Family-owned cache implementations still decide whether full-KV, sliding,
or linear-recurrent state can fork at a requested logical offset.

## Tensor Lifetime Audit

The cache fork lifetime is unchanged. `lookup()` still returns a forked
`TransformerCache` owned by the caller, and sessions dispose unconsumed forks
through the existing `PromptPrefixCacheSession` disposal path.

The added metadata copies arrays and prompt identity strings out of the retained
entry. It does not expose cache tensors, snapshot internals, or mutable
references to serve-owned identity state.

## Memory / Performance Evidence

- `bun test packages/serve/src/engine/prefix-cache.test.ts`: passed, `9` tests.
- `bench:generation`: not rerun; this tranche does not change model math,
  decode scheduling, tensor kernels, cache tensor layout, or endpoint routing.
- `bench:generation:parity`: not rerun for the same reason. The previous
  continuous prefix-cache tranche passed `bun run regression:qwen-gemma --
  --profile real`; this tranche keeps the same restore path and only annotates
  hits.

## Independent Review

Feynman reviewed the next cache step before implementation. The recommendation
was to land a block-aware prefix-storage contract before paged attention:
preserve `CausalLM`, `TransformerCache`, `TransformerBatchCache`, snapshot
`canFork/fork`, usage semantics, and continuous scheduler behavior; make
exact, prefix, supersequence, and LCP decisions explicit; and keep Qwen/Gemma
non-trimmable semantics controlled by family-owned snapshots.

## Out-of-scope Drift Noticed

None.

## Remaining Risks / Follow-ups

This is not block deduplication. The next tranche still needs block hashing,
reference-counted block ownership, memory-aware eviction, and endpoint evidence
that divergent repeated chats reuse a longest common prefix without changing
usage accounting or continuous-scheduler fairness.
