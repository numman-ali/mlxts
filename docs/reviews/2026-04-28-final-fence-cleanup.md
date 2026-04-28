# Runtime Review: Final Fence Cleanup

## Summary

Tranche 15 closes the fence-only cleanup from the architectural audit. The
tracked source changes keep runtime profiling fully env-gated in `MxArray`
construction/free and FFI result handling, and add the missing rationale for the
private transposed-weight caches in dense `Linear` and `Embedding`.

The nanoGPT README documents the committed example boundary and its ignored
`dist/` output. The empty quantize providers directory and nanoGPT `dist/`
artifact were untracked local residue and were removed from the working tree.

## Files Reviewed

- `packages/core/src/array.ts`
- `packages/nn/src/layers/linear.ts`
- `packages/nn/src/layers/embedding.ts`

## Tensor Lifetime Audit

`readResultPointer()` preserves the same ownership path: one `OutSlot` per FFI
call, caller-owned invoke, and one pointer read by label. The disabled-profile
branch skips timing calls only; it does not reuse output slots, change pointer
ownership, or hide disposable `MxArray` values.

The `MxArray` constructor still registers every native handle with the
FinalizationRegistry, and `free()` still unregisters before `mlx_array_free`.
The profile flag only controls timing capture around those operations.

The `Linear` and `Embedding` changes are comment-only. Their cached
`weight.T` handles still refresh when the public `weight` object changes and
are freed during disposal.

## Memory / Performance Evidence

Focused validation passed:

- `bun test packages/core/src/runtime-profile.test.ts`
- `bun test packages/nn/src/layers/linear.test.ts packages/nn/src/layers/embedding.test.ts`
- `bun run --filter '@mlxts/core' typecheck`
- `bun run --filter '@mlxts/nn' typecheck`

No generation benchmark was run for this tranche because model forward math,
sampling, cache update/fetch, scheduler behavior, and generation hot-path files
were not changed. The core change removes disabled-profile timestamp work while
preserving the enabled `MLXTS_RUNTIME_PROFILE=1` path.

## Independent Review

Kepler, a GPT-5.5 xhigh explorer sub-agent, performed a read-only second
opinion on the final cleanup state. The review confirmed that the nanoGPT
`dist/` and quantize providers items were untracked residue, the align
optimizer dependency premise was invalidated, and the touched core/nn files
require this runtime review artifact.

## Remaining Risks / Follow-ups

None for this tranche.
