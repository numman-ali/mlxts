# Full-KV Prefix Tensor Blocks

## Goal

Prompt-prefix cache retention currently deduplicates prompt-token metadata but
stores each family-owned cache snapshot as a complete cloned tensor set. The
next cache-backend proof deduplicates retained full-KV snapshot tensors at
block-aligned prompt prefixes while preserving the existing
`TransformerCacheSnapshot` contract.

This is not paged KV and does not change active decode storage. It is a
retention backend for prompt-boundary snapshots.

## Scope

- Applies only to trimmable full-KV single-sequence cache snapshots.
- Reuses complete tensor blocks from a cache fork's source snapshot when a later
  snapshot extends that prefix.
- Keeps partial tail blocks private to the newer snapshot.
- Keeps Qwen hybrid snapshots exact-only and unchanged.
- Keeps Gemma sliding/layer-pattern snapshots on the current non-deduplicated
  path when any layer is sliding.
- Keeps serve routing, prompt matching, media identity, and usage accounting
  unchanged.

## Ownership

`@mlxts/transformers` owns tensor-block retention, ref-counting, fork materialization,
and block disposal. `@mlxts/serve` continues to store opaque
`TransformerCacheSnapshot` objects and uses only `estimatedByteSize`, `canFork`,
`fork`, and `dispose`.

Snapshot byte accounting is dynamic: each retained tensor block is charged to
exactly one live snapshot. When that snapshot is disposed, the charge transfers
to another live snapshot that references the block. Serve computes retained
bytes from live entries instead of maintaining a stale increment/decrement total.

## Validation

- Full-KV snapshot tests prove prefix forks remain byte-for-byte identical.
- A source-fork-extend snapshot test proves shared complete blocks reduce
  retained bytes and transfer accounting on parent disposal.
- Prompt-prefix cache tests prove byte-budget eviction uses dynamic retained
  snapshot bytes.
- Qwen/Gemma quick regression proves hybrid and layer-pattern serving routes do
  not change.

## Out Of Scope

- Active paged KV allocation.
- Copy-on-write active cache blocks.
- SSD-persistent cache.
- Quantized KV storage.
- Partial-block sharing.
- Any operator-facing `paged` or tensor-block flag.
