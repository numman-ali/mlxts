# Runtime Review: Full-KV Prefix Tensor Blocks

## Summary

This tranche adds a transformer-owned snapshot backend for full-KV,
trimmable, single-sequence cache snapshots. When a prompt-prefix cache hit
forks a cache and later stores an extended prompt-boundary snapshot, the new
snapshot reuses complete 64-token key/value blocks from the source snapshot and
clones only the new tail blocks.

This is a retained-snapshot backend, not paged KV. Active decode storage,
attention execution, cache append/fetch behavior, Qwen hybrid caches, and Gemma
sliding/layer-pattern caches remain unchanged.

Serve continues to store opaque `TransformerCacheSnapshot` objects. Its only
runtime change is that retained snapshot bytes are computed from live snapshot
estimates, because shared tensor-block ownership can transfer when an older
snapshot is disposed.

## Files Reviewed

- `packages/transformers/src/infrastructure/cache/runtime.ts`
- `packages/transformers/src/infrastructure/cache/batch.ts`
- `packages/transformers/src/infrastructure/cache/layer-pattern-batch.ts`
- `packages/transformers/src/infrastructure/cache/single.ts`
- `packages/transformers/src/infrastructure/cache/snapshot.ts`
- `packages/transformers/src/infrastructure/cache/tensor-block-snapshot.ts`
- `packages/transformers/src/infrastructure/cache/index.test.ts`
- `packages/serve/src/engine/prefix-cache-entry.ts`
- `packages/serve/src/engine/prefix-cache.ts`
- `packages/serve/src/engine/prefix-cache.test.ts`

## Tensor Lifetime Audit

`CacheBase.snapshot()` still clones visible layer state before building a
retained snapshot. The full-KV block backend then slices block views from those
owned clones, clones each retained block, and disposes the temporary full-layer
snapshot state in a `finally` block.

`FullKVTensorBlock` owns exactly one cloned `MxArray`. Snapshot owners and
borrowed cache forks retain references separately. Disposing a snapshot releases
snapshot ownership; disposing a cache fork releases borrowed source references.
The tensor is freed only when no snapshot owner and no borrower remain.

`FullKVBlockSnapshot.fork()` materializes retained blocks into temporary
key/value tensors, restores them through the existing cache restore path, and
disposes the temporary layer snapshot in all success and error paths. The forked
cache receives a source lineage only after all layers restore and the cache
offset advances successfully.

`BatchKVCache` and `LayerPatternBatchKVCache` retain full-KV source lineage per
row when a seeded single-cache fork is restored, filtered, extended, and later
extracted. This keeps continuous serving from losing the source lineage before
the next prompt-boundary snapshot is written.

Qwen hybrid and Gemma mixed sliding/full layer-pattern snapshots do not enter
the new backend. `createCacheSnapshot()` selects the block backend only when
the trim policy is `"prefix"` and every `CacheLayerKind` is `"full"`.

## Memory / Performance Evidence

Focused tests passed after the continuous-shaped lineage fix:

```bash
bun test packages/transformers/src/infrastructure/cache/index.test.ts packages/serve/src/engine/prefix-cache.test.ts
```

Result: `42 pass`, `0 fail`.

Typecheck passed:

```bash
bun run typecheck
```

Result: all workspaces exited with code `0`.

The mechanical runtime gates passed:

```bash
bun run lint
bun run check:file-lines
bun run check:assertions
bun run check:tensor-lifetimes
bun run check:runtime-review
```

Result: each command exited with code `0`.

Focused serve and cache tests passed:

```bash
bun test packages/serve packages/transformers/src/infrastructure/cache
```

Result: `350 pass`, `0 fail`.

Coverage passed:

```bash
bun run check:coverage
```

Result: `Coverage thresholds satisfied.`

Qwen/Gemma quick regression passed after the batch-lineage fix:

```bash
bun run regression:qwen-gemma -- --profile quick
```

Result: `84` transformer focused tests and `205` serve focused tests passed
with `0` failures.

The final repo validation fence passed:

```bash
bun run validate
```

Result: all validation steps passed, including `typecheck`, `lint`,
`check:assertions`, `check:file-lines`, `check:per-package-agents`,
`check:cross-package-imports`, `check:tensor-lifetimes`,
`check:runtime-review`, and `check:coverage`.

The new transformer test proves a `65`-token parent snapshot and `80`-token
child snapshot share the first complete `64`-token full-KV block. The child
charges only its private `16`-token tail while the parent is live, then charges
the shared block after parent disposal. A trimmed fork at offset `70` restores
the expected key/value prefix.

The batch-cache regression test covers the continuous serving shape Bohr
flagged: restore a seeded single-cache fork into `BatchKVCache`, dispose the
fork, extend that batch into another batch cache, dispose the temporary batch,
extract the row, and then snapshot it. The extracted snapshot still charges
only the private tail while the parent snapshot is live.

The new serve test proves prompt-prefix retained byte stats and hit metadata
read live `snapshot.estimatedByteSize` values instead of stale entry-time
numbers.

`bench:generation` and `bench:generation:parity` were not rerun for this
tranche because model forward, active decode, attention kernels, sampling,
cache append/fetch layout, and scheduler behavior are unchanged. The runtime
surface touched here is prompt-boundary snapshot retention and fork
materialization.

## Independent Review

Kant recommended the design-first full-KV tensor-block proof before this
implementation: keep it under the existing `TransformerCacheSnapshot`
contract, limit it to full-KV/trimmable snapshots, keep Qwen hybrid and Gemma
sliding semantics unchanged, and avoid exposing a paged-cache flag.

Bohr reviewed the first diff and blocked it because direct single-cache reuse
worked but continuous serving could lose source lineage when converting the
seeded fork into a batch cache. The tranche now retains source lineage through
managed batch restore/filter/extend/extract, and the focused transformer test
covers that path.

Planck reviewed the final diff after the batch-lineage fix and found no
correctness, ownership/lifetime, Qwen/Gemma route, or serve cache semantic
blockers. The one non-blocking note is that cache-hit fork restoration now
materializes block pieces and then restores through the existing clone/slice
path, so this tranche must not claim lower cache-hit latency without paired
TTFT evidence.

## Out-of-scope Drift Noticed

None.

## Remaining Risks / Follow-ups

The implementation deduplicates retained prompt-boundary snapshots only. It
does not reduce active decode KV allocation, add paged attention, add
copy-on-write active cache blocks, quantize KV retention, or spill cache blocks
to SSD.

Snapshot creation still takes a full cloned visible layer snapshot before
packing retained blocks, so this tranche reduces retained snapshot memory after
construction but does not claim lower snapshot-time peak memory.

Cache-hit restore still materializes retained block pieces and feeds them
through the existing cache restore path. Measure repeated-turn cache-hit TTFT
before claiming serving latency or restore peak-memory improvement from this
tranche.
