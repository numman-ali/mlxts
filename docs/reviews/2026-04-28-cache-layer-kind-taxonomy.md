# Runtime Review: Cache Layer Kind Taxonomy

## Summary

Tranche 13 adds the shared `CacheLayerKind` taxonomy under transformer cache
infrastructure and exposes `layerKinds` on single caches, batch caches, and
snapshots. Full KV caches report `full`, sliding-window caches report `sliding`,
Gemma layer-pattern caches report their mixed full/sliding pattern, and Qwen
hybrid caches report `linear-recurrent` for linear-attention state.

## Files Reviewed

- `packages/transformers/src/types.ts`
- `packages/transformers/src/index.ts`
- `packages/transformers/src/infrastructure/cache/index.ts`
- `packages/transformers/src/infrastructure/cache/layer-kind.ts`
- `packages/transformers/src/infrastructure/cache/single.ts`
- `packages/transformers/src/infrastructure/cache/batch.ts`
- `packages/transformers/src/infrastructure/cache/layer-pattern-batch.ts`
- `packages/transformers/src/infrastructure/generation/batch-cache-factory.ts`
- `packages/transformers/src/families/qwen3_5/cache/index.ts`
- `packages/transformers/src/families/qwen3_5/cache/batch-cache.ts`

## Tensor Lifetime Audit

The changed cache production files do not alter tensor update, append,
slice, retain, dispose, or eval ownership logic. `layerKinds` are immutable
string metadata created at cache construction or snapshot construction time.
No tensor-producing expressions were moved into nested calls, and no disposable
`MxArray` lifetime was hidden by the taxonomy.

## Family Mapping Verification

Full KV LLaMA-like caches map every layer to `full`.

Gemma layer-pattern caches map `undefined` window sizes to `full` and positive
window sizes to `sliding`; the shared attention-label mapper also maps
`full_attention` and `sliding_attention` before batch-cache selection.

Qwen hybrid caches map `full_attention` to `full` and `linear_attention` to
`linear-recurrent` for both single and batch cache implementations.

## Memory / Performance Evidence

No generation hot path changed: cache append/fetch, snapshot restore, batch
filter/extend, and model forward semantics are untouched. The new metadata is
allocated once per cache or snapshot and returned by reference from getters.
`bench:generation` and `bench:generation:parity` were not run because this
tranche does not change model forward, cache tensor mutation, sampling, or
scheduler loop behavior.

Focused validation passed:

- `bun test packages/transformers/src/infrastructure/cache/index.test.ts`
- `bun test packages/transformers/src/families/qwen3_5/cache/cache.test.ts`
- `bun test packages/transformers/src/infrastructure/generation/continuous-batch.test.ts`
- `bun test packages/serve/src/engine/prefix-cache.test.ts`
- `bun test packages/serve/src/engine/engine.test.ts`
- `bun run typecheck`
- `bun run validate`

## Independent Review

Harvey, a GPT-5.5 xhigh explorer sub-agent, independently inspected the cache
and routing surfaces. The review recommended keeping the taxonomy in
`@mlxts/transformers`, exposing it through existing cache objects, and not
adding new family mapping logic to `@mlxts/serve`.

## Out-of-scope Drift Noticed

None.

## Remaining Risks / Follow-ups

`@mlxts/serve` still has route eligibility checks that know about family model
types. This tranche does not move those checks; future paged or quantized cache
backends should dispatch from `cache.layerKinds`, not from serve-side family
strings.
