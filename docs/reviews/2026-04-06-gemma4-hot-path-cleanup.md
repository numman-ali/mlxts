# Runtime Review: Gemma 4 Hot-Path Cleanup and Metadata Caching

## Summary

This cleanup removed the loose experimental direction from the Gemma 4 parity
work and reduced the remaining diff to three intentional pieces:

1. tracked arrays can now carry known shape and dtype metadata across common
   core wrappers, which avoids unnecessary FFI round-trips for metadata reads
   without changing ownership semantics
2. Gemma 4 now builds per-layer inputs lazily and avoids redundant retained KV
   pairs for non-reused layers
3. generation and benchmark paths no longer force an extra explicit eval before
   `.item()` when the scalar read already performs the required sync

The cleaned patch is easier to reason about and aligned with the repo's design
rules, but it does not close the Gemma 4 decode gap against `mlx-lm`. The
remaining bottleneck still appears to live in the steady-state cache/update
contract rather than in the removed experimental surfaces.

## Files Reviewed

- `packages/core/src/array-metadata.ts`
- `packages/core/src/array.ts`
- `packages/core/src/fast.ts`
- `packages/core/src/ops/arithmetic.ts`
- `packages/core/src/ops/linalg.ts`
- `packages/core/src/ops/shape.ts`
- `packages/transformers/src/families/gemma4/attention.ts`
- `packages/transformers/src/families/gemma4/model.ts`
- `packages/transformers/src/generation.ts`

## Tensor Lifetime Audit

The cleanup keeps the tracked `MxArray` ownership model intact. No untracked or
GC-bypassing surfaces remain in the production diff.

`readResultArrayWithMetadata()` still returns normal tracked arrays and routes
through the same per-call `OutSlot` result ownership as the existing wrappers.
The new `ArrayMetadata` helper only seeds cached shape and dtype fields on the
wrapper; it does not change disposal or aliasing behavior.

The Gemma 4 attention/model changes keep local tensor lifetimes visible. The
lazy per-layer input path creates one sliced/reshaped view per layer and frees
it in the same lexical scope. Shared-KV cleanup now returns borrowed shared
buffers directly and frees non-retained fresh buffers explicitly in the same
call frame.

The generation changes remove redundant explicit eval calls but keep one scalar
sync per generated token via `.item()`. The decode loop still has a visible and
auditable synchronization point per token.

## Memory / Performance Evidence

Fresh sequential measurements on cached `google/gemma-4-E2B-it` with
`1024` prompt tokens, `128` generated tokens, and `3` trials:

- `bun run bench:generation` equivalent:
  `bun packages/transformers/scripts/benchmark-generation.ts --model google/gemma-4-E2B-it --prompt-tokens 1024 --generation-tokens 128 --trials 3`
  - `prompt_tps=8171.868`
  - `generation_tps=75.536`
  - `peak_memory=9.985 GB`
  - `evals_per_token=1.00`
- `bun run bench:generation:parity` equivalent:
  `bun packages/transformers/scripts/benchmark-generation-parity.ts --model google/gemma-4-E2B-it --prompt-tokens 1024 --generation-tokens 128 --trials 3 --mlx-lm-python .tmp/venvs/mlx-lm-bench/bin/python`
  - `mlx-lm`: `prompt_tps=1241.769`, `generation_tps=90.618`,
    `peak_memory=9.889 GB`
  - `mlxts`: `prompt_tps=7634.302`, `generation_tps=75.614`,
    `peak_memory=9.985 GB`

These numbers show that the cleaned patch leaves the repo in a better state,
but not a faster one for Gemma 4 decode. The measured decode gap remains about
`16.6%`, so the next performance step should target cache update/fetch churn
rather than reintroducing speculative ownership APIs.

## Independent Review

The cleanup direction was cross-checked against independent external reviews
from Claude and Codex after the broader experiment was stripped back. The
reviewers agreed that the public `untracked` / `unchecked` branch had drifted
too far and should not ship as part of the parity fix.

Additional local verification:

- `bun run --filter @mlxts/transformers typecheck`
- `bun test packages/core/src/array.test.ts packages/core/src/ops/ops.test.ts packages/core/src/fast-rms-norm.test.ts packages/core/src/fast.test.ts packages/transformers/src/families/gemma4/model.test.ts packages/transformers/src/load.test.ts packages/transformers/src/interaction-profile.test.ts packages/transformers/scripts/benchmark-common.test.ts packages/transformers/scripts/benchmark-long-context.test.ts`

## Remaining Risks / Follow-ups

- Gemma 4 decode parity is still open. The current best next move is a narrow
  cache-local optimization pass in `packages/transformers/src/infrastructure/cache.ts`
  that reduces steady-state wrapper churn without widening public APIs.
- The metadata-caching helpers are intentionally conservative. If they prove
  valuable across more families, they can stay; if not, they should be kept
  small and explicit rather than expanded into a broader abstraction layer.
- The prompt-throughput gap relative to `mlx-lm` is not apples-to-apples and
  should not be treated as the main benchmark conclusion here. Decode
  throughput remains the acceptance metric for this parity work.
