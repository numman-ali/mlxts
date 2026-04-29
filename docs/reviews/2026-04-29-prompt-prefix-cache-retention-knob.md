# Prompt-Prefix Cache Retention Knob

## Summary

This tranche exposes prompt-prefix cache retention as an operator-bounded serve
runtime setting. The engine still defaults to one retained prompt-boundary
snapshot per served model, so existing Qwen, Gemma, and full-KV behavior stays
unchanged unless the operator explicitly raises
`promptPrefixCacheMaxEntries` / `--prompt-prefix-cache-max-entries`.

This is not paged KV, tensor-block cache deduplication, byte-budgeted eviction,
or a new transformer cache backend. Serve still restores cache state only
through family-owned `TransformerCacheSnapshot.canFork()` and `fork()`.

## Files Reviewed

- `packages/serve/src/cli-options.ts`
- `packages/serve/src/cli.ts`
- `packages/serve/src/engine/index.ts`
- `packages/serve/src/http/route-info.ts`
- `packages/serve/src/index.ts`
- `packages/serve/src/model-loading/server-options.ts`
- `packages/serve/src/model-loading/server.ts`
- `packages/serve/src/model-loading/sources.ts`
- `packages/serve/src/runtime/strategy.ts`

## Runtime Sensitivity Notes

`engine/index.ts` is runtime-sensitive because the configured retention limit is
passed to the `PromptPrefixCache` instance that may skip prompt prefill after a
cache hit. The change affects only how many retained prompt-boundary snapshots
can be considered by the existing lookup path. Hit selection still checks token
LCP, prompt identity, and family-owned snapshot forkability before returning a
cache fork.

The public serve option intentionally remains positive. The lower-level
`PromptPrefixCache` still supports `0` for direct tests and internal seams, but
the serving surface does not expose a cache-disable setting for a cache behavior
that is now part of the regression contract.

## Tensor Lifetime Audit

No model math, attention tensors, cache tensor layout, or snapshot/fork
implementation changed. Raising the retention value can keep more
family-owned snapshots alive, so memory usage grows by retained snapshot count;
operators must choose values that fit local memory until byte-budgeted eviction
lands.

Snapshot ownership remains unchanged: retained entries own snapshots, lookup
returns a forked cache owned by the request, and engine disposal releases the
prompt-prefix cache.

## Memory / Performance Evidence

- `bun test packages/serve/src/engine/engine.test.ts packages/serve/src/cli.test.ts packages/serve/src/model-loading/server.test.ts packages/serve/src/http/server.test.ts packages/serve/src/runtime/strategy.test.ts`: passed, `120` tests.
- `bun test packages/serve/src/engine/prefix-cache.test.ts packages/serve/src/engine/prefix-cache-blocks.test.ts packages/serve/src/engine/prefix-cache-index.test.ts`: passed, `21` tests.
- `bun test packages/serve`: passed, `312` tests.
- `bun run --filter '@mlxts/serve' typecheck`: passed.
- `bun run check:runtime-review`: passed and validated this artifact.
- `bun run validate`: passed, including coverage.
- `bun run regression:qwen-gemma -- --profile quick`: passed.

## Independent Review

Chandrasekhar reviewed the in-progress tranche. The review found the core
runtime wiring complete, flagged missing README/docs/review-artifact coverage,
and recommended stronger `/info` plus runtime-default assertions. Those gaps
are covered by this commit.

## Out-of-scope Drift Noticed

None.

## Remaining Risks / Follow-ups

Multi-entry retention can retain more cache snapshot memory, but it is not yet
tied to a byte budget or scheduler memory-reservation accounting. The next cache
backend tranche should decide whether byte-budgeted eviction or paged/block
cache tensor reuse is the right follow-up, then pair it with endpoint evidence.
