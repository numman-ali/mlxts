# Gemma 4 Parity Experiments

## Goal

Close the remaining Gemma 4 decode gap against `mlx-lm` without violating the
repo's readability and semantic-surface rules.

This document is the working memory for the parity loop:

1. start from a clean committed baseline
2. try one bounded idea at a time
3. measure it honestly
4. keep only the ideas that help
5. remove failed experiments before moving on

## Baseline

- Branch: `feat/gemma4-kimi-pass`
- Baseline commit: `a4ba663` (`Fix Gemma 4 sampling and KEqV hot-path waste`)
- Local Bun: `1.3.4`

### Canonical parity benchmark

- Command:
  `MLX_LM_BENCH_PYTHON=.tmp/venvs/mlx-lm-bench/bin/python bun run bench:generation:parity --model google/gemma-4-E2B-it --prompt-tokens 1024 --generation-tokens 128 --trials 3`
- `mlxts`: `generation_tps=80.865`
- `mlx-lm`: `generation_tps=90.016`
- Gap: `-9.151 tok/s` (`-10.2%`)

### Profile canary

- Command:
  `MLXTS_RUNTIME_PROFILE=1 bun run bench:generation:parity --model google/gemma-4-E2B-it --prompt-tokens 1024 --generation-tokens 32 --trials 1`
- `mlxts`: `generation_tps=84.410`
- Top core labels:
  - `matmul: 0.0773 ms/token`
  - `fast_rms_norm: 0.0720 ms/token`
  - `fast_rope: 0.0646 ms/token`
  - `reshape: 0.0375 ms/token`
  - `add: 0.0356 ms/token`
  - `transpose: 0.0350 ms/token`
  - `fast_scaled_dot_product_attention: 0.0325 ms/token`

## Summary

The current kept baseline has two real winners. Experiment 35 stays because it
shrinks the full-cache growth cliff at the `1023 -> 1024` boundary. Experiment
37 stays because it fixes the `mlx-lm` reference helper so parity runs measure
the same fixed-length greedy decode window that `mlxts` measures.

With both kept changes in place, the healthy-window canonical Gemma 4 parity
benchmark is now at or slightly above paired parity across repeated
confirmations. The first `1024 / 128 / 3` confirmation landed at
`mlx-lm=81.152`, `mlxts=82.250`, gap `+1.098`, ratio `1.014`. The immediate
repeat landed at `mlx-lm=81.333`, `mlxts=82.039`, gap `+0.706`, ratio `1.009`.
The wider methodology cross-checks stayed aligned too: our `1024 / 128 / 5`
parity run averaged `mlx-lm=78.114`, `mlxts=80.375`, while `mlx-lm`'s own
official `benchmark.py` averaged `80.099` on the same local snapshot.

## Files Reviewed

- `packages/transformers/src/infrastructure/cache/ops.ts`
- `packages/transformers/src/infrastructure/cache/ops.test.ts`
- `packages/transformers/scripts/benchmark-common.ts`
- `packages/transformers/scripts/benchmark-generation-parity.ts`
- `packages/transformers/scripts/benchmark-mlx-lm.py`
- `packages/transformers/src/families/gemma4/attention.ts`
- `packages/transformers/src/families/gemma4/runtime/attention.ts`
- `packages/transformers/src/infrastructure/cache/runtime.ts`
- `.reference/mlx-lm/mlx_lm/models/gemma4_text.py`
- `.reference/mlx-lm/mlx_lm/models/cache.py`
- `.reference/mlx-lm/mlx_lm/benchmark.py`

## Tensor Lifetime Audit

The kept cache-growth change stays inside the existing cache ownership model.
`growCacheBuffer()` still returns a fresh owned array, and
`appendFullCacheState()` still frees replaced buffers before swapping in the new
ones. The new benchmark-helper change does not touch production tensor
lifetimes; it only changes how the Python reference harness decides when to
stop generation.

Focused cache and Gemma 4 attention tests passed after the kept code changes,
and no new nested tensor-lifetime hazards were introduced in the touched TypeScript
production files.

## Memory / Performance Evidence

- `bun run bench:generation:parity --model google/gemma-4-E2B-it --prompt-tokens 1024 --generation-tokens 128 --trials 3`
  is the canonical paired benchmark used throughout this loop.
- `bun run bench:generation:parity --model google/gemma-4-E2B-it --prompt-tokens 1023 --generation-tokens 8 --trials 5`
  and the matching `prompt_tokens=1024` probe were used for the boundary-cliff
  check that justified Experiment 35.
- `bun run bench:generation` remains the local non-reference companion surface
  for quick decode checks, but keeper decisions in this round were made from
  paired `bench:generation:parity` evidence.
- Experiment 35 boundary probe:
  `mlxts` cliff across `1023 -> 1024` shrank from `-15.360 tok/s` to
  `-4.661 tok/s`.
- Experiment 35 same-weather reverted control:
  patched canonical pair `mlx-lm=73.919`, `mlxts=65.952`, gap `-7.967`;
  reverted control `mlx-lm=64.073`, `mlxts=48.208`, gap `-15.865`.
- Experiment 37 reference-harness correction:
  before fix, long synthetic `32000 / 64` could report nonsense
  `mlx-lm=36697.248`;
  after fix, `32000 / 64` became `mlx-lm=71.967`, `mlxts=72.813`, gap `+0.846`.
- Canonical confirmation under the fixed harness:
  `1024 / 128 / 3` -> `mlx-lm=81.152`, `mlxts=82.250`, gap `+1.098`.
- Immediate repeat canonical confirmation under the fixed harness:
  `1024 / 128 / 3` -> `mlx-lm=81.333`, `mlxts=82.039`, gap `+0.706`.
- `5`-trial methodology cross-check:
  `1024 / 128 / 5` -> helper reference `mlx-lm=78.114`, `mlxts=80.375`,
  while official `mlx-lm benchmark.py` on the same snapshot averaged
  `generation_tps=80.099`.
- Longer decode confirmations under the fixed helper:
  `1024 / 1000 / 1` -> `mlx-lm=67.148`, `mlxts=80.989`, gap `+13.841`.
  `1024 / 10000 / 1` -> `mlx-lm=72.468`, `mlxts=78.450`, gap `+5.982`.

## Independent Review

Two high-effort explorers were used as pressure tests during this round. One
focused on Gemma 4 parity shape versus `.reference/mlx-lm`; the other focused
on native/core seam candidates. Their main useful conclusion was not a new kept
code path but a prioritization lesson: if parity had still lagged after the
benchmark fix, the next bounded seam would likely have stayed in the full-cache
source-layer path rather than in sliding caches, isolated output fusion, or
shared-KV wrapper churn.

## Remaining Risks / Follow-ups

The main remaining risk is over-trusting a single benchmark shape. The
reference helper is now fairer, but future winners should still be rechecked on
boundary-sensitive and longer-context runs so we do not accidentally optimize
for one healthy window only.

If a real gap reappears under the fixed harness, the next candidate should come
from the full-cache source-layer neighborhood, not from previously rejected
shared-KV wrapper cleanup, isolated output-only helpers, or sliding-cache
micro-optimizations.

## Experiment Log

### Experiment 1: Remove MLP weight transposes from the compiled hot path

- Status: `failed and reverted`
- Hypothesis:
  Gemma 4 MLP currently transposes `gate`, `up`, and `down` weights inside the
  compiled closure on every decode step. `Linear` already caches transposed
  weights for its own eager `forward`, so exposing or reusing that cached form
  should reduce hot-path `transpose` work without making the semantic MLP
  surface less readable.
- Success criteria:
  - no readability regression in `families/gemma4/mlp.ts`
  - no semantic API drift at model call sites
  - measurable improvement on the Gemma 4 parity benchmark
- Rollback rule:
  if the change does not materially help, remove it before trying the next idea

#### Outcome

- Result:
  rejected
- Measurement:
  - paired parity command:
    `MLX_LM_BENCH_PYTHON=.tmp/venvs/mlx-lm-bench/bin/python bun run bench:generation:parity --model google/gemma-4-E2B-it --prompt-tokens 1024 --generation-tokens 128 --trials 3`
  - `mlxts`: `generation_tps=80.746`
  - `mlx-lm`: `generation_tps=91.196`
  - gap: `-10.450 tok/s` (`-11.5%`)
- Conclusion:
  removing the transposes from the Gemma 4 MLP compiled closure did not reduce
  the paired decode gap, so the change was removed.

### Experiment 2: Normalize q/k/v before transpose to match `mlx-lm`

- Status: `failed and reverted`
- Hypothesis:
  `mlx-lm` normalizes q/k/v in `[B, L, H, D]` layout before transposing to
  `[B, H, L, D]`, while `mlxts` normalizes after transpose. Matching the
  reference layout order might reduce head-prep cost without changing semantic
  surfaces.
- Success criteria:
  - `attention.ts` stays semantic
  - no public API drift
  - paired Gemma 4 gap improves relative to the baseline commit
- Rollback rule:
  if the paired gap does not improve, remove the experiment before the next one

#### Outcome

- Result:
  rejected
- Measurement:
  - paired parity command:
    `MLX_LM_BENCH_PYTHON=.tmp/venvs/mlx-lm-bench/bin/python bun run bench:generation:parity --model google/gemma-4-E2B-it --prompt-tokens 1024 --generation-tokens 128 --trials 3`
  - `mlxts`: `generation_tps=81.058`
  - `mlx-lm`: `generation_tps=91.861`
  - gap: `-10.803 tok/s` (`-11.8%`)
- Conclusion:
  aligning q/k/v normalization order with `mlx-lm` did not improve the paired
  gap, so the change was removed.

### Experiment 3: Skip fresh KV materialization for non-retained source layers

- Status: `failed and reverted`
- Hypothesis:
  Gemma 4 only needs a very small number of source-layer KV pairs for later
  shared reuse, but `mlxts` may still be materializing returned key/value pairs
  for every non-shared source layer. Letting the model opt out of returning
  fresh KV pairs for layers that will never be reused might remove needless
  retain/materialize/free churn.
- Success criteria:
  - no semantic drift in the readable model surface
  - no correctness regressions in cache-backed Gemma 4 attention
  - paired Gemma 4 gap improves relative to the baseline commit
- Rollback rule:
  if the paired gap does not improve, remove the experiment before the next one

#### Outcome

- Result:
  rejected
- Measurement:
  - paired parity command:
    `MLX_LM_BENCH_PYTHON=.tmp/venvs/mlx-lm-bench/bin/python bun run bench:generation:parity --model google/gemma-4-E2B-it --prompt-tokens 1024 --generation-tokens 128 --trials 3`
  - `mlxts`: `generation_tps=80.945`
  - `mlx-lm`: `generation_tps=91.428`
  - gap: `-10.483 tok/s` (`-11.5%`)
- Conclusion:
  removing the fresh-KV return path for non-retained source layers did not
  improve the paired gap enough to keep the extra control-flow parameter, so
  the change was removed.

### Experiment 4: Replace Gemma 4's compiled MLP helper with eager module calls

- Status: `failed and reverted`
- Hypothesis:
  Gemma 4 is the only decoder family still routing its dense MLP through a
  family-local compiled runtime helper. Gemma 3 and the llama-like families use
  plain `Linear.forward` calls plus `gegluApprox`, and `Linear.forward` already
  caches its transposed weight internally. Reverting Gemma 4 to the same eager
  pattern might reduce compile-island overhead while also improving cross-family
  consistency.
- Success criteria:
  - `Gemma4TextMLP` reads like the other family MLPs
  - no semantic API drift outside the MLP implementation
  - paired Gemma 4 gap improves relative to the baseline commit
- Rollback rule:
  if the paired gap does not improve, restore the compiled Gemma 4 MLP path

#### Outcome

- Result:
  rejected
- Measurement:
  - paired parity command:
    `MLX_LM_BENCH_PYTHON=.tmp/venvs/mlx-lm-bench/bin/python bun run bench:generation:parity --model google/gemma-4-E2B-it --prompt-tokens 1024 --generation-tokens 128 --trials 3`
  - `mlxts`: `generation_tps=79.411`
  - `mlx-lm`: `generation_tps=90.781`
  - gap: `-11.370 tok/s` (`-12.5%`)
- Conclusion:
  moving Gemma 4 back to the eager MLP shape made the paired gap worse, which
  is a useful result in itself: the family-local compiled MLP helper is
  probably buying us something on this branch, so future experiments should aim
  at cache/shared-KV or per-layer-input behavior instead of removing the MLP
  runtime path.

## Research Notes

- Official `google-deepmind/gemma` still does not publish a Gemma 4 technical
  report as of `2026-04-20`, so the research loop is using the official model
  card, JAX implementation, and upstream reference repos instead.
- The strongest remaining architecture clues are:
  - E2B relies heavily on shared KV and per-layer inputs
  - shared-KV consumer layers should not build fresh K/V at all
  - sliding and global layers have intentionally different head geometry and
    likely different kernel sensitivities
- The most promising next hypothesis area is now cache/shared-KV and
  per-layer-input behavior, not MLP simplification.

### Experiment 5: Keep shared KV as a private cache view across source and shared layers

- Status: `failed and reverted`
- Hypothesis:
  Gemma 4 retained source layers currently collapse their cache-facing
  `TransformerCacheView` into an owned `{ keys, values }` pair, then later
  shared-consumer layers wrap those same tensors back into a borrowed cache
  view. Keeping the private cache view alive all the way through the source and
  shared-consumer layers might reduce retain/free and wrapper churn without
  changing the readable model surface.
- Success criteria:
  - no public cache API change
  - no readability regression in the Gemma 4 semantic files
  - paired Gemma 4 gap improves relative to the baseline commit
- Rollback rule:
  if the paired gap does not improve, restore the owned-pair handoff

#### Outcome

- Result:
  rejected
- Measurement:
  - paired parity command:
    `MLX_LM_BENCH_PYTHON=.tmp/venvs/mlx-lm-bench/bin/python bun run bench:generation:parity --model google/gemma-4-E2B-it --prompt-tokens 1024 --generation-tokens 128 --trials 3`
  - `mlxts`: `generation_tps=81.132`
  - `mlx-lm`: `generation_tps=90.698`
  - gap: `-9.566 tok/s` (`-10.5%`)
- Conclusion:
  the private cache-view handoff did not beat the baseline paired gap cleanly
  enough to keep. The cache/view seam was a reasonable thing to test, but the
  result says it is not the next meaningful lever on its own, so the experiment
  was removed.

### Experiment 6: Compile Gemma 4 per-layer-input construction

- Status: `failed and reverted`
- Hypothesis:
  E2B relies on per-layer inputs, and `createPerLayerInputs()` is one of the
  remaining repeated pure tensor subgraphs unique to the small Gemma 4 models.
  Compiling that helper behind the existing semantic surface might reduce the
  fixed per-token tax without making the model files less readable.
- Success criteria:
  - the semantic `createPerLayerInputs()` surface stays intact
  - no public API drift outside the runtime helper
  - paired Gemma 4 gap improves relative to the baseline commit
- Rollback rule:
  if the paired gap does not improve, restore the eager helper implementation

#### Outcome

- Result:
  rejected
- Measurement:
  - paired parity command:
    `MLX_LM_BENCH_PYTHON=.tmp/venvs/mlx-lm-bench/bin/python bun run bench:generation:parity --model google/gemma-4-E2B-it --prompt-tokens 1024 --generation-tokens 128 --trials 3`
  - `mlxts`: `generation_tps=80.597`
  - `mlx-lm`: `generation_tps=91.223`
  - gap: `-10.626 tok/s` (`-11.6%`)
- Conclusion:
  compiling the per-layer-input construction did not help the paired decode gap.
  That pushes the evidence away from the small-model PLE setup path and back
  toward the remaining attention/cache/kernel boundary as the more likely
  bottleneck.

### Experiment 7: Compile query head prep before RoPE

- Status: `failed and reverted`
- Hypothesis:
  Query head prep is still one of the hottest repeated pure motifs in the
  decode profile. Compiling the `qProjection -> reshape -> transpose -> qNorm`
  path before numeric RoPE might reduce fixed decode overhead on every layer
  without forcing the dynamic offset into the compiled region.
- Success criteria:
  - the readable attention surface stays semantic
  - no cache or mask behavior changes
  - paired Gemma 4 gap improves relative to the baseline commit
- Rollback rule:
  if the paired gap does not improve, remove the experiment before the next one

#### Outcome

- Result:
  rejected
- Measurement:
  - paired parity command:
    `MLX_LM_BENCH_PYTHON=.tmp/venvs/mlx-lm-bench/bin/python bun run bench:generation:parity --model google/gemma-4-E2B-it --prompt-tokens 1024 --generation-tokens 128 --trials 3`
  - `mlxts`: `generation_tps=80.652`
  - `mlx-lm`: `generation_tps=91.701`
  - gap: `-11.049 tok/s` (`-12.0%`)
- Conclusion:
  query-only compiled head prep was not the lever. The math is hot, but this
  isolated compile island did not narrow the paired decode gap enough to keep.

### Experiment 8: CompileMany standard key/value prep before RoPE

- Status: `failed and reverted`
- Hypothesis:
  The standard Gemma 4 source-layer path still does separate `kProjection` and
  `vProjection` preparation before RoPE and cache update. Fusing those two
  pure pre-RoPE branches with `compileMany()` might reduce decode overhead
  without dragging cache or mask state into the compiled region.
- Success criteria:
  - no semantic drift in Gemma 4 attention
  - no new dynamic offset handling inside the compiled helper
  - paired Gemma 4 gap improves relative to the baseline commit
- Rollback rule:
  if the paired gap does not improve, remove the experiment before the next one

#### Outcome

- Result:
  rejected
- Measurement:
  - paired parity command:
    `MLX_LM_BENCH_PYTHON=.tmp/venvs/mlx-lm-bench/bin/python bun run bench:generation:parity --model google/gemma-4-E2B-it --prompt-tokens 1024 --generation-tokens 128 --trials 3`
  - `mlxts`: `generation_tps=80.325`
  - `mlx-lm`: `generation_tps=90.849`
  - gap: `-10.524 tok/s` (`-11.6%`)
- Conclusion:
  key/value pre-RoPE fusion on its own also failed to narrow the paired gap,
  so the next experiments should avoid more small compile fragments on the same
  side of the attention boundary.

### Experiment 9: Compile `SDPA -> transpose -> reshape -> output projection`

- Status: `failed and reverted`
- Hypothesis:
  `mlx-lm` effectively leaves the post-SDPA attention tail in one lazy region.
  Compiling the bias-free `scaledDotProductAttention -> transpose -> reshape ->
  output projection` motif for `null` and `"causal"` masks might remove a
  repeated decode tail without touching cache mutation or dynamic RoPE.
- Success criteria:
  - attention.ts stays semantic
  - mask-array paths keep their eager fallback
  - paired Gemma 4 gap improves relative to the baseline commit
- Rollback rule:
  if the paired gap does not improve, remove the experiment before the next one

#### Outcome

- Result:
  rejected
- Measurement:
  - paired parity command:
    `MLX_LM_BENCH_PYTHON=.tmp/venvs/mlx-lm-bench/bin/python bun run bench:generation:parity --model google/gemma-4-E2B-it --prompt-tokens 1024 --generation-tokens 128 --trials 3`
  - `mlxts`: `generation_tps=80.810`
  - `mlx-lm`: `generation_tps=90.525`
  - gap: `-9.715 tok/s` (`-10.7%`)
- Conclusion:
  this was the wrong side of the boundary. It covered too narrow a slice of the
  real calls, and the full-attention decode shapes made it a poor compile seam.

### Experiment 10: Decode-only source-layer `q/k/v` prep with `compileMany()`

- Status: `failed and reverted`
- Hypothesis:
  The most promising pre-RoPE seam left was the standard source-layer decode
  path as one unit: `qProjection`, `kProjection`, `vProjection`, reshape,
  transpose, and head RMSNorm, with RoPE, cache update, and SDPA left eager.
  Restricting the fast path to `sequenceLength === 1` was meant to keep the
  compiled region stable and decode-specific.
- Success criteria:
  - semantic attention flow stays readable
  - the fast path only applies to standard source-layer decode
  - paired Gemma 4 gap improves cleanly enough to survive an A/B check
- Rollback rule:
  if the paired gap does not hold up under confirmation, remove the experiment

#### Outcome

- Result:
  rejected after A/B check
- Measurement:
  - first paired run:
    - `mlxts`: `generation_tps=81.569`
    - `mlx-lm`: `generation_tps=87.516`
    - gap: `-5.947 tok/s` (`-6.8%`)
  - confirmation paired run:
    - `mlxts`: `generation_tps=81.225`
    - `mlx-lm`: `generation_tps=91.151`
    - gap: `-9.926 tok/s` (`-10.9%`)
  - same-conditions baseline control after revert:
    - `mlxts`: `generation_tps=81.736`
    - `mlx-lm`: `generation_tps=91.051`
    - gap: `-9.315 tok/s` (`-10.2%`)
- Conclusion:
  the first run looked genuinely promising, but the controlled baseline check
  showed the fast path was not a reliable improvement. It does not beat the
  clean baseline under like-for-like conditions, so it was removed.

### Experiment 11: Replace prefix-view retargeting with fresh prefix slices

- Status: `failed and reverted`
- Hypothesis:
  Full-attention cache growth still pays `sliceViewInPlace()` retargeting on the
  borrowed prefix views. `mlx-lm` uses direct prefix slices instead. Replacing
  the reusable visible-handle retargeting with fresh prefix slices might reduce
  wrapper or FFI churn while keeping the cache semantics unchanged.
- Success criteria:
  - cache behavior and tests remain identical
  - no public cache API changes
  - paired Gemma 4 gap improves relative to the baseline commit
- Rollback rule:
  if our side does not get faster or the paired gap does not improve, remove it

#### Outcome

- Result:
  rejected
- Measurement:
  - paired parity command:
    `MLX_LM_BENCH_PYTHON=.tmp/venvs/mlx-lm-bench/bin/python bun run bench:generation:parity --model google/gemma-4-E2B-it --prompt-tokens 1024 --generation-tokens 128 --trials 3`
  - `mlxts`: `generation_tps=81.387`
  - `mlx-lm`: `generation_tps=89.747`
  - gap: `-8.360 tok/s` (`-9.3%`)
  - same-period baseline control:
    - `mlxts`: `generation_tps=81.736`
    - `mlx-lm`: `generation_tps=91.051`
    - gap: `-9.315 tok/s` (`-10.2%`)
- Conclusion:
  the paired reference drifted a bit, but the more important signal is that our
  own decode rate got slightly worse than the baseline control. The
  `sliceViewInPlace()` retargeting is not the hidden drag here, so the change
  was removed.

### Experiment 12: Reuse an all-null decode attention-mask plan

- Status: `failed and reverted`
- Hypothesis:
  On Gemma 4 single-token decode with an active cache, every attention mask
  resolves to `null` anyway. Reusing a cached all-null per-layer mask array
  should remove a little JS/runtime bookkeeping without changing any actual
  attention behavior.
- Success criteria:
  - no semantic change in mask behavior
  - our side gets faster often enough to survive an A/B/A check
  - paired Gemma 4 gap does not regress relative to the clean baseline
- Rollback rule:
  if the result does not survive A/B/A confirmation, remove it

#### Outcome

- Result:
  rejected after A/B/A check
- Measurement:
  - first patched run:
    - `mlxts`: `generation_tps=82.151`
    - `mlx-lm`: `generation_tps=91.684`
    - gap: `-9.533 tok/s` (`-10.4%`)
  - control baseline after revert:
    - `mlxts`: `generation_tps=81.622`
    - `mlx-lm`: `generation_tps=89.807`
    - gap: `-8.185 tok/s` (`-9.1%`)
  - patched rerun:
    - `mlxts`: `generation_tps=81.383`
    - `mlx-lm`: `generation_tps=91.111`
    - gap: `-9.728 tok/s` (`-10.7%`)
- Conclusion:
  the first patched run was the outlier, not the new baseline. The mask-plan
  reuse did not hold up under A/B/A confirmation, so it was removed.

### Experiment 13: Skip `TransformerCacheView` wrapping for shared KV layers

- Status: `failed and reverted`
- Hypothesis:
  Shared Gemma 4 layers already receive a stable `(keys, values)` pair. Passing
  that pair straight into SDPA and output instead of wrapping it in a borrowed
  `TransformerCacheView` might remove some wrapper churn across the majority of
  decode layers without changing semantics.
- Success criteria:
  - shared-layer attention stays semantically identical
  - no cache ownership or lifetime regressions
  - paired Gemma 4 gap improves relative to the clean baseline
- Rollback rule:
  if the paired gap does not improve, remove it

#### Outcome

- Result:
  rejected
- Measurement:
  - paired parity command:
    `MLX_LM_BENCH_PYTHON=.tmp/venvs/mlx-lm-bench/bin/python bun run bench:generation:parity --model google/gemma-4-E2B-it --prompt-tokens 1024 --generation-tokens 128 --trials 3`
  - `mlxts`: `generation_tps=81.354`
  - `mlx-lm`: `generation_tps=90.961`
  - gap: `-9.607 tok/s` (`-10.6%`)
- Conclusion:
  removing the shared-layer cache-view wrapper was not enough to matter. The
  remaining issue is deeper than that boundary object.

### Experiment 14: Native pair cache-write helper

- Status: `failed and reverted`
- Hypothesis:
  The cache hot path still performs separate in-place writes for keys and
  values on every update. A tiny native helper that mutates both buffers in one
  call should reduce FFI pressure on the exact hot mutable state the repo’s
  runtime rules reserve native helpers for.
- Success criteria:
  - core and cache tests stay green after a real native rebuild
  - Gemma 4 correctness stays intact
  - paired Gemma 4 gap improves relative to the clean baseline
- Rollback rule:
  if the paired gap does not improve, remove the helper and rebuild back to the
  clean source state

#### Outcome

- Result:
  rejected
- Measurement:
  - paired parity command:
    `MLX_LM_BENCH_PYTHON=.tmp/venvs/mlx-lm-bench/bin/python bun run bench:generation:parity --model google/gemma-4-E2B-it --prompt-tokens 1024 --generation-tokens 128 --trials 3`
  - `mlxts`: `generation_tps=81.234`
  - `mlx-lm`: `generation_tps=90.437`
  - gap: `-9.203 tok/s` (`-10.2%`)
- Conclusion:
  even a targeted native helper at the cache-write boundary did not move the
  real paired Gemma 4 decode number enough to keep. That is strong evidence
  that the remaining gap lives in a deeper cache-plus-attention seam, not just
  in the pair of write calls.

### Experiment 15: Remove parity-only logprobs materialization from the benchmark loop

- Status: `failed and reverted`
- Hypothesis:
  The parity benchmark currently materializes logprobs on our side inside the
  generation loop even though the headline comparison is decode throughput. If
  that extra evaluation is benchmark-only overhead, removing it should narrow
  the paired gap without touching production model code.
- Success criteria:
  - benchmark semantics stay fair enough to compare against the local `mlx-lm`
    reference
  - paired Gemma 4 gap improves repeatably, not just on one noisy run
  - the result survives an A/B/A check against the clean baseline
- Rollback rule:
  if the paired gap does not improve repeatably, revert the benchmark-only
  change

#### Outcome

- Result:
  rejected after A/B/A check
- Measurement:
  - first patched run:
    - `mlxts`: `generation_tps=81.064`
    - `mlx-lm`: `generation_tps=90.334`
    - gap: `-9.270 tok/s` (`-10.3%`)
  - control baseline after revert:
    - `mlxts`: `generation_tps=81.583`
    - `mlx-lm`: `generation_tps=91.546`
    - gap: `-9.963 tok/s` (`-10.9%`)
  - patched rerun:
    - `mlxts`: `generation_tps=81.535`
    - `mlx-lm`: `generation_tps=92.347`
    - gap: `-10.812 tok/s` (`-11.7%`)
- Conclusion:
  the reference side drifted enough that the first run looked tempting, but
  the A/B/A check did not hold up. This was measurement hygiene, not a stable
  path to parity, so it was removed.

### Experiment 16: Narrow native fused query-plus-attention helper

- Status: `failed and reverted`
- Hypothesis:
  A decode-only native helper that fuses query projection, query norm, RoPE,
  SDPA, and output projection for Gemma 4 single-token attention might cut a
  meaningful amount of JS-visible lazy graph and FFI overhead without changing
  the readable semantic model surface.
- Success criteria:
  - core, nn, and Gemma 4 attention tests pass after a real native rebuild
  - the paired Gemma 4 gap improves relative to the clean baseline
  - the result survives a same-period reverted control
- Rollback rule:
  if the paired gap does not beat the reverted control, remove the helper

#### Outcome

- Result:
  rejected after same-period control
- Measurement:
  - patched run:
    - `mlxts`: `generation_tps=81.246`
    - `mlx-lm`: `generation_tps=91.010`
    - gap: `-9.764 tok/s` (`-10.7%`)
  - reverted control:
    - `mlxts`: `generation_tps=81.276`
    - `mlx-lm`: `generation_tps=90.790`
    - gap: `-9.514 tok/s` (`-10.5%`)
- Conclusion:
  fusing the pure attention side alone was still too shallow. The reverted
  baseline was slightly better on the paired gap, which means the remaining
  issue is not just query-side graph shape or output projection overhead. The
  next honest seam is deeper: cache mutation plus cache consumption together,
  not more pure-attention fusion on its own.

### Experiment 17: Decode-only cache-plus-consume helper without JS cache views

- Status: `failed and reverted`
- Hypothesis:
  The remaining readable seam was to keep cache mutation in the cache runtime,
  but stop constructing JS-visible cache views on the single-token source-layer
  decode path. If we updated the cache and then handed the backing buffers plus
  `visibleLength` straight into one native SDPA-plus-output helper, we might
  finally remove the cache-plus-attention glue overhead that the narrower
  helpers were still leaving behind.
- Success criteria:
  - focused core, nn, cache, and Gemma 4 tests pass after a real native rebuild
  - paired Gemma 4 gap moves materially relative to the current clean baseline
  - the result survives a second patched confirmation run
- Rollback rule:
  if the paired gap does not improve clearly, remove the seam and restore the
  readable baseline

#### Outcome

- Result:
  rejected after confirmation run
- Measurement:
  - first patched run:
    - `mlxts`: `generation_tps=81.372`
    - `mlx-lm`: `generation_tps=91.939`
    - gap: `-10.567 tok/s` (`-11.5%`)
  - patched confirmation run:
    - `mlxts`: `generation_tps=81.371`
    - `mlx-lm`: `generation_tps=91.357`
    - gap: `-9.986 tok/s` (`-10.9%`)
- Conclusion:
  this was the right deeper seam to test next, but it still did not close the
  real paired gap. The decode helper stayed readable and passed focused tests,
  but the benchmark said no. That is strong evidence that the remaining Gemma 4
  gap is deeper than cache-view churn and deeper than source-layer
  cache-plus-consume glue at this granularity.

### Experiment 18: Compile the shared-layer decode block after query prep

- Status: `failed and reverted`
- Hypothesis:
  Gemma 4’s later KV-shared layers are pure on single-token decode. If we
  compile the shared-layer consumer block from SDPA onward — taking
  `rotatedQueries`, shared keys, shared values, and optional per-layer input as
  inputs — we might give MLX a larger stable region to optimize without
  reopening cache mutation or muddying the semantic model surface.
- Success criteria:
  - focused Gemma 4 shared-layer tests stay green
  - paired Gemma 4 gap improves clearly relative to the clean baseline
  - the result survives a confirmation rerun
- Rollback rule:
  if the paired gap is only flat or worse, remove the shared-layer compiled
  path and return to the clean baseline

#### Outcome

- Result:
  rejected after confirmation run
- Measurement:
  - first patched run:
    - `mlxts`: `generation_tps=81.447`
    - `mlx-lm`: `generation_tps=90.971`
    - gap: `-9.524 tok/s` (`-10.5%`)
  - patched confirmation run:
    - `mlxts`: `generation_tps=81.430`
    - `mlx-lm`: `generation_tps=91.134`
    - gap: `-9.704 tok/s` (`-10.6%`)
- Conclusion:
  compiling the pure shared block from `rotatedQueries` onward was not enough.
  It stayed readable and mechanically correct, but it only produced another
  near-tie with the baseline. That suggests the still-untested part of the same
  idea is larger: query prep and RoPE likely need to sit inside the shared
  compiled region too, or this direction is simply not the lever.

### Experiment 19: Compile the full shared-layer decode block including query prep

- Status: `failed and reverted`
- Hypothesis:
  Gemma 4's later KV-shared layers are pure on single-token decode, and the
  remaining hot labels still include query-side work like `fast_rms_norm`,
  `matmul`, and `fast_rope`. If we move query projection, query norm, dynamic
  RoPE, shared-KV SDPA, output projection, and the whole feedforward tail into
  one shapeless compiled shared-layer helper, MLX might finally get a large
  enough pure region to close the paired gap without reopening cache mutation.
- Success criteria:
  - focused Gemma 4 shared-layer tests and transformers typecheck stay green
  - paired Gemma 4 gap improves clearly relative to the clean baseline
  - the result survives a confirmation rerun
- Rollback rule:
  if the paired gap is flat or worse, remove the compiled shared-layer fast
  path and return to the clean readable baseline

#### Outcome

- Result:
  rejected after confirmation run
- Measurement:
  - first patched run:
    - `mlxts`: `generation_tps=81.228`
    - `mlx-lm`: `generation_tps=91.105`
    - gap: `-9.877 tok/s` (`-10.8%`)
  - patched confirmation run:
    - `mlxts`: `generation_tps=80.992`
    - `mlx-lm`: `generation_tps=91.564`
    - gap: `-10.572 tok/s` (`-11.5%`)
- Conclusion:
  the larger shared-layer compile was not the answer either. Pulling query prep
  and dynamic RoPE inside the compiled region made the paired result slightly
  worse, not better. That is strong evidence that the remaining Gemma 4 gap is
  not explained by shared-layer pure graph boundaries alone, even when the
  compiled region covers the whole shared consumer block.

### Experiment 20: Remove per-token tokenizer decode bookkeeping from the parity harness

- Status: `kept`
- Hypothesis:
  The `mlxts` parity harness was still doing extra per-token CPU bookkeeping in
  the steady-state decode loop by decoding each generated token just to count
  string length, while the paired metric is supposed to reflect model decode
  throughput against `mlx-lm`. Removing that bookkeeping should tighten the
  measured paired gap if it is materially distorting the benchmark.
- Success criteria:
  - transformers typecheck stays green
  - patched paired runs improve the gap relative to the current clean harness
  - a same-period reverted control clearly gives back the improvement
- Rollback rule:
  if a reverted control lands essentially the same, restore the old harness and
  treat the change as benchmark noise

#### Outcome

- Result:
  accepted after same-period reverted control
- Measurement:
  - patched run 1:
    - `mlxts`: `generation_tps=81.509`
    - `mlx-lm`: `generation_tps=90.655`
    - gap: `-9.146 tok/s` (`-10.1%`)
  - patched run 2:
    - `mlxts`: `generation_tps=81.413`
    - `mlx-lm`: `generation_tps=90.948`
    - gap: `-9.535 tok/s` (`-10.5%`)
  - reverted control:
    - `mlxts`: `generation_tps=81.281`
    - `mlx-lm`: `generation_tps=91.304`
    - gap: `-10.023 tok/s` (`-11.0%`)
- Conclusion:
  this was not model-speed magic, but it was a real fairness bug in the paired
  measurement. The old harness was taxing `mlxts` with extra per-token decode
  bookkeeping that did not belong in the throughput comparison. Removing it
  tightened the measured Gemma 4 gap by roughly `0.5` to `0.9 tok/s` against a
  same-period reverted control, so the cleaner harness is the right baseline to
  keep for future experiments.

### Experiment 21: Drop parity-only `logprobs` work from the greedy harness

- Status: `failed and reverted`
- Hypothesis:
  Once the per-token tokenizer decode bookkeeping was removed, the parity
  harness still carried one obvious piece of unused work: it was computing
  `logprobs` for every greedy step even though the benchmark only consumed the
  next token. Removing that should further tighten the paired gap if the extra
  tensor work was still distorting the harness.
- Success criteria:
  - transformers typecheck stays green
  - the paired gap improves relative to the cleaner Experiment 20 baseline
- Rollback rule:
  if the paired gap worsens or stays flat, restore the previous harness

#### Outcome

- Result:
  rejected after first paired run
- Measurement:
  - patched run:
    - `mlxts`: `generation_tps=81.450`
    - `mlx-lm`: `generation_tps=91.806`
    - gap: `-10.356 tok/s` (`-11.3%`)
- Conclusion:
  removing the `logprobs` path did not help once the harness was otherwise
  cleaned up. The paired result got worse, so this was not hidden benchmark
  waste; it was just a losing change. The fairer Experiment 20 harness remains
  the right baseline.

### Experiment 22: Mirror mlx-lm's Gemma 4 cache topology with per-layer source handles

- Status: `failed and reverted`
- Hypothesis:
  The remaining Gemma 4 gap might still come from the cache object shape on the
  decode hot path. If we mirror mlx-lm more closely by giving source layers
  real per-layer cache handles and shared layers lightweight offset-only
  handles, while leaving the attention math unchanged, the paired gap might
  tighten without muddying the readable model surface.
- Success criteria:
  - focused Gemma 4 shared-KV decode tests and transformers typecheck stay green
  - the paired Gemma 4 gap improves clearly relative to the fairer Experiment 20
    baseline
- Rollback rule:
  if the paired gap stays flat or gets worse, restore the simpler generic cache
  baseline

#### Outcome

- Result:
  rejected after confirmation run
- Measurement:
  - first patched run:
    - `mlxts`: `generation_tps=81.512`
    - `mlx-lm`: `generation_tps=91.077`
    - gap: `-9.565 tok/s` (`-10.5%`)
  - patched confirmation run:
    - `mlxts`: `generation_tps=81.397`
    - `mlx-lm`: `generation_tps=91.014`
    - gap: `-9.617 tok/s` (`-10.6%`)
- Conclusion:
  the reference-shaped cache handles were not the lever. They stayed correct
  and readable, but the paired result only tied or slightly trailed the fairer
  Experiment 20 baseline. That rules out cache-object topology, on its own, as
  a meaningful remaining Gemma 4 win.

### Experiment 23: Materialize the cached transposed tied-output weight

- Status: `failed and reverted`
- Hypothesis:
  Gemma 4's tied output projection is unusually large because of the vocabulary
  size. If `Embedding.asLinear()` pays any extra lazy-transpose cost there, it
  could hurt Gemma 4 much more than the smaller models that are already at
  parity. Forcing the cached transposed embedding weight to materialize once
  might reduce decode overhead on the LM head.
- Success criteria:
  - `@mlxts/nn` and `@mlxts/transformers` typecheck stay green
  - embedding and Gemma 4 tests stay green
  - the paired Gemma 4 gap improves relative to the Experiment 20 baseline
- Rollback rule:
  if the paired gap worsens or stays flat, restore the lazy cached transpose

#### Outcome

- Result:
  rejected after first paired run
- Measurement:
  - patched run:
    - `mlxts`: `generation_tps=81.458`
    - `mlx-lm`: `generation_tps=91.223`
    - gap: `-9.765 tok/s` (`-10.7%`)
    - peak memory: `9.894 GB`
- Conclusion:
  the tied-output transpose was not hiding the Gemma 4 gap. Forcing the cached
  transpose to materialize once neither helped speed nor changed peak memory in
  a useful way, so the lazy cached transpose remains the better baseline.

### Experiment 24: Compile the full per-layer-input tail inside each Gemma 4 block

- Status: `failed and reverted`
- Hypothesis:
  Every Gemma 4 layer pays the per-layer-input gate, projection, norm, and
  residual-add tail. That motif is pure, Gemma 4-specific, and repeated across
  the whole decoder, so compiling it behind the semantic `applyPerLayerInput`
  seam might cut a meaningful amount of repeated graph construction without
  muddying the readable block surface.
- Success criteria:
  - focused Gemma 4 tests and transformers typecheck stay green
  - the paired Gemma 4 benchmark completes normally
  - the paired gap improves relative to the Experiment 20 baseline
- Rollback rule:
  if the benchmark becomes pathological or the paired gap does not improve,
  remove the compiled tail helper and restore the eager block path

#### Outcome

- Result:
  rejected after pathological benchmark behavior
- Measurement:
  - focused Gemma 4 tests: passed
  - transformers typecheck: passed
  - paired benchmark: did not produce any output and became effectively idle for
    over two minutes before being killed
- Conclusion:
  this compile seam is not viable in practice. Even though the code stayed
  mechanically correct under tests, the real benchmark path became
  pathologically slow or hung before it could report a first trial. That is
  enough evidence to reject it and keep the eager per-layer-input tail.

## Historical Checks

- Older runtime-heavy commit `4144102` (`runtime: clean gemma4 hot-path keeper set`)
  was benchmarked in a temporary worktree under the current local `mlx-lm`
  reference and came out much worse than the present baseline:
  - `mlxts`: `generation_tps=75.502`
  - `mlx-lm`: `generation_tps=91.634`
  - gap: `-16.132 tok/s` (`-17.6%`)
- Conclusion:
  the readable baseline did not regress away from some hidden faster local
  state. The older runtime-heavy Gemma 4 path is not the answer.

## Late-Round Note

- During the later native-helper experiments on `2026-04-21`, the machine fell
  into a clearly throttled state while on automatic power mode.
- Absolute prompt/decode throughput collapsed for both `mlxts` and `mlx-lm`, so
  the only trustworthy signal from that window is the paired gap and ratio from
  back-to-back runs under the same machine state.
- Those tiny paired runs were:
  `prompt_tokens=128`, `generation_tokens=8`, `trials=1`

### Experiment 25: Native query/key/value head prep for the standard sliding source path

- Status: `kept for further widening`
- Hypothesis:
  The remaining Gemma 4 gap still sits in the uncollapsed pure head-prep seam:
  projection, reshape, transpose, RMSNorm, and RoPE. A private native helper
  that returns prepared heads directly, while leaving the readable attention
  surface alone, might tighten the paired gap without reopening the fragile
  mutable-cache boundary.
- Success criteria:
  - focused core and Gemma 4 attention tests stay green
  - transformers typecheck stays green
  - the paired tiny-run gap improves relative to the same-period baseline
- Outcome:
  - new core helpers landed for query prep and standard key/value prep
  - focused tests passed
  - same-period tiny paired runs improved the gap
- Measurement:
  - baseline tiny run:
    - `mlxts`: `generation_tps=2.260`
    - `mlx-lm`: `generation_tps=2.843`
    - gap: `-0.583 tok/s`
    - ratio: `0.795`
  - native head-prep tiny run:
    - `mlxts`: `generation_tps=2.413`
    - `mlx-lm`: `generation_tps=2.770`
    - gap: `-0.357 tok/s`
    - ratio: `0.871`
- Conclusion:
  even in a badly throttled machine state, collapsing the head-prep seam
  improved the paired gap materially. This is the first deeper runtime seam in
  a while that produced a repeatable positive signal.

### Experiment 26: Widen native head prep to proportional RoPE and KEqV

- Status: `kept for further comparison`
- Hypothesis:
  If native head prep is the right seam, the improvement should not stay trapped
  inside only the easiest sliding source layers. Supporting proportional RoPE
  and Gemma 4's KEqV path should broaden the win or at least preserve it.
- Success criteria:
  - new core helper coverage passes for proportional RoPE and KEqV
  - Gemma 4 attention parity tests stay green with the experiment flag on
  - paired tiny-run signal stays positive relative to baseline
- Outcome:
  - widened helpers landed for proportional-RoPE query prep and KEqV key/value
    prep
  - focused tests passed
  - paired tiny-run signal stayed positive
- Measurement:
  - baseline tiny run:
    - `mlxts`: `generation_tps=2.229`
    - `mlx-lm`: `generation_tps=2.756`
    - gap: `-0.527 tok/s`
    - ratio: `0.808`
  - widened native head-prep tiny run:
    - `mlxts`: `generation_tps=2.398`
    - `mlx-lm`: `generation_tps=2.760`
    - gap: `-0.362 tok/s`
    - ratio: `0.869`
- Conclusion:
  widening the head-prep helper preserved the positive signal. That makes the
  seam more credible: it is not just a lucky win trapped inside one narrow
  slice of the Gemma 4 decoder.

### Experiment 27: Native attention-output projection seam

- Status: `kept as the strongest current candidate`
- Hypothesis:
  The attention output path still rebuilds `transpose -> reshape -> output
  projection` in JS-visible graph form on every layer. A private native helper
  that projects `[B, H, L, D]` back to `[B, L, hidden]` might outperform the
  head-prep seam or stack with it.
- Success criteria:
  - new core helper test passes
  - Gemma 4 attention tests stay green
  - the paired tiny-run gap improves relative to the same-period baseline
- Outcome:
  - native output projection helper landed
  - focused tests passed
  - the output-only variant beat both baseline and head-prep-only in the tiny
    paired matrix
- Measurement:
  - four-way tiny matrix:
    - baseline: `mlxts=2.321`, `mlx-lm=2.940`, gap `-0.619`, ratio `0.789`
    - head-only: `mlxts=2.363`, `mlx-lm=2.775`, gap `-0.412`, ratio `0.851`
    - output-only: `mlxts=2.388`, `mlx-lm=2.678`, gap `-0.290`, ratio `0.892`
    - both: `mlxts=2.393`, `mlx-lm=2.782`, gap `-0.389`, ratio `0.860`
  - confirmation pair:
    - baseline: `mlxts=2.367`, `mlx-lm=2.867`, gap `-0.500`, ratio `0.826`
    - output-only: `mlxts=2.378`, `mlx-lm=2.687`, gap `-0.309`, ratio `0.885`
- Conclusion:
  output projection is currently the strongest seam in this throttled test
  window. Head prep helps, output helps more, and the two together were not
  additive in the first matrix. That points to the next round: keep both
  helpers available, but treat output projection as the better default
  candidate to re-anchor once the machine returns to a healthy benchmark state.

### Experiment 28: Canonical validation of the native head-prep seam

- Status: `failed and reverted`
- Hypothesis:
  The tiny-run native head-prep win should survive the real canonical parity
  benchmark if it is a genuine Gemma 4 decode improvement rather than a
  throttled-window artifact.
- Success criteria:
  - the paired canonical benchmark still beats the clean baseline
  - the result holds up under a longer `5`-trial confirmation run
- Rollback rule:
  if the result does not survive the canonical confirmation run, remove the
  helper from the keep set and keep using the readable baseline

#### Outcome

- Result:
  rejected after canonical confirmation
- Measurement:
  - first clean sequential `3`-trial matrix:
    - baseline: `mlxts=80.290`, `mlx-lm=91.299`, gap `-11.009`, ratio `0.879`
    - head-only: `mlxts=81.861`, `mlx-lm=91.392`, gap `-9.531`, ratio `0.896`
  - confirmation `5`-trial run:
    - head-only: `mlxts=80.489`, `mlx-lm=91.882`, gap `-11.393`, ratio `0.876`
  - same-round clean baseline control:
    - baseline: `mlxts=82.046`, `mlx-lm=92.201`, gap `-10.155`, ratio `0.890`
- Conclusion:
  native head prep looked promising in the first healthy `3`-trial matrix, but
  it did not survive the more stable `5`-trial check. That is exactly the kind
  of thing this research loop is meant to catch. The tiny-run signal was not a
  durable canonical win, so head prep does not stay in the keep set.

### Experiment 29: Canonical validation of the native attention-output seam

- Status: `failed and reverted`
- Hypothesis:
  The native attention-output helper was the strongest tiny-run candidate and
  the best-looking small native seam. If it is a real win, it should stay ahead
  of the readable baseline in bracketed canonical parity runs once the machine
  returns to a healthy state.
- Success criteria:
  - the paired canonical benchmark beats the clean baseline
  - a same-weather baseline control does not erase the gain
- Rollback rule:
  if the bracketed baseline control is still better, remove the helper from the
  keep set

#### Outcome

- Result:
  rejected after bracketed canonical control
- Measurement:
  - first clean `3`-trial pair:
    - baseline: `mlxts=82.015`, `mlx-lm=92.685`, gap `-10.670`, ratio `0.885`
    - output-only: `mlxts=82.799`, `mlx-lm=91.640`, gap `-8.841`, ratio `0.904`
  - later `5`-trial output-only run:
    - output-only: `mlxts=80.361`, `mlx-lm=88.417`, gap `-8.056`, ratio `0.909`
  - bracket baseline immediately after:
    - baseline: `mlxts=81.543`, `mlx-lm=89.247`, gap `-7.704`, ratio `0.914`
- Conclusion:
  output projection was the strongest small native seam in the tiny and early
  canonical windows, but the same-weather baseline control still came out
  slightly better. That means the helper is not yet a trustworthy keep. It may
  still be pointing at the right neighborhood, but this implementation does not
  earn permanence.

### Experiment 30: Native sliding decode-attention stage

- Status: `failed and reverted`
- Hypothesis:
  The remaining gap might only close if we hand MLX a whole single-token
  sliding source-layer attention stage at once: `q/k/v projection -> norm ->
  RoPE -> cache update -> SDPA -> output projection`. If Bun is currently
  paying for too many seams around that stage, one deeper native helper should
  finally narrow the paired gap.
- Success criteria:
  - the new helper survives a real native rebuild and focused correctness tests
  - the canonical paired gap beats the clean baseline
  - the result holds up under a longer reverse-order confirmation run
- Rollback rule:
  if the helper is stable but slower, remove it and keep only the ABI lesson in
  the research record

#### Outcome

- Result:
  rejected after clean canonical confirmation
- Measurement:
  - first implementation used a large mixed-argument Bun FFI signature and
    reproducibly segfaulted Bun on Apple Silicon
  - repackaging the helper to `one tensor vector + tiny config buffers + three
    out slots` fixed the crash, and focused tests passed after a forced native
    rebuild
  - steady-state profile canary:
    - native decode-attention: `generation_tps=83.175`
    - profile improved on JS-visible overhead:
      - `ffi_ms_per_token=0.1173`
      - `wrapper_ms_per_token=0.1495`
  - clean canonical `3`-trial pair:
    - baseline: `mlxts=81.585`, `mlx-lm=90.618`, gap `-9.033`, ratio `0.900`
    - native decode-attention: `mlxts=79.828`, `mlx-lm=89.305`, gap `-9.477`, ratio `0.894`
  - reverse-order `5`-trial confirmation:
    - native decode-attention: `mlxts=80.690`, `mlx-lm=91.382`, gap `-10.692`, ratio `0.883`
    - same-round clean baseline control:
      - baseline: `mlxts=82.046`, `mlx-lm=92.201`, gap `-10.155`, ratio `0.890`
- Conclusion:
  this was an ambitious and worthwhile experiment. It proved two important
  things. First, the original high-arity helper failure really did look like a
  Bun/ARM64 ABI packaging problem, not a math problem. Second, even after the
  helper became stable with pointer-only packaging, this particular
  cache-plus-attention seam was still slower in real paired benchmarks. So the
  Bun-facing packaging lesson is worth keeping, but the helper itself is not.

### Experiment 31: Pre-slice Gemma 4 per-layer inputs outside the decoder loop

- Status: `failed and reverted`
- Hypothesis:
  `mlx-lm` turns Gemma 4's per-layer inputs into a Python list before the
  decoder loop and then just indexes that list inside the loop. Our path keeps
  one `[B, L, layerCount, hidden]` tensor and performs `slice + reshape` on
  every layer iteration. Pre-slicing those inputs once outside the loop should
  reduce repeated loop-local graph work without changing the semantic model
  surface.
- Success criteria:
  - Gemma 4 model, block, and attention tests stay green
  - transformers typecheck stays green
  - the paired canonical Gemma 4 benchmark improves relative to the current
    fair baseline
- Rollback rule:
  if the paired result gets worse, remove the change immediately and return to
  the baseline tensor path

#### Outcome

- Result:
  rejected after first canonical run
- Measurement:
  - focused Gemma 4 tests: passed
  - transformers typecheck: passed
  - canonical `5`-trial pair:
    - `mlxts`: `generation_tps=80.744`
    - `mlx-lm`: `generation_tps=92.585`
    - gap: `-11.841 tok/s` (`-12.8%`)
  - recent clean baseline controls for comparison:
    - baseline A: `mlxts=82.046`, `mlx-lm=92.201`, gap `-10.155`, ratio `0.890`
    - baseline B: `mlxts=81.543`, `mlx-lm=89.247`, gap `-7.704`, ratio `0.914`
- Conclusion:
  this was a clean, reference-backed hypothesis, but the benchmark said no.
  Hoisting the per-layer slice work out of the loop made the paired result
  worse, not better. That pushes the remaining suspicion back toward the
  retained source-layer cache/shared-KV handoff rather than the per-layer-input
  setup path.

### Experiment 32: Prefer owned cache pairs for retained source-layer handoff

- Status: `failed and reverted`
- Hypothesis:
  `mlx-lm` threads concrete `(keys, values)` pairs straight through Gemma 4's
  retained source/shared-KV handoff. Our path still updates the cache through a
  `TransformerCacheView`, then later borrows and materializes that pair for
  shared reuse. If retained source layers ask the cache for an owned pair up
  front, we might remove enough view/materialization churn to narrow the paired
  decode gap without making the semantic model surface less readable.
- Success criteria:
  - focused Gemma 4 tests stay green
  - transformers typecheck stays green
  - the canonical paired Gemma 4 benchmark improves on recent clean baselines
- Rollback rule:
  if the paired result does not beat the clean baseline controls, remove the
  handoff change immediately and keep only the benchmark result

#### Outcome

- Result:
  rejected after first canonical run
- Measurement:
  - focused Gemma 4 tests: passed
  - transformers typecheck: passed
  - canonical `5`-trial pair:
    - `mlxts`: `generation_tps=82.060`
    - `mlx-lm`: `generation_tps=93.981`
    - gap: `-11.921 tok/s`
    - ratio: `0.873`
  - recent clean baseline controls for comparison:
    - baseline A: `mlxts=82.046`, `mlx-lm=92.201`, gap `-10.155`, ratio `0.890`
    - baseline B: `mlxts=81.543`, `mlx-lm=89.247`, gap `-7.704`, ratio `0.914`
- Conclusion:
  this was the top remaining reference-shaped cache handoff idea, and it still
  did not help. `mlxts` stayed effectively flat while `mlx-lm` remained
  stronger in the same window, so the paired ratio got worse rather than
  better. That is a useful negative result: the remaining gap is probably not
  just `TransformerCacheView` churn at the retained source/shared boundary.

### Experiment 33: Bypass `TransformerCacheView` on shared-KV consumer layers

- Status: `failed and reverted`
- Hypothesis:
  Gemma 4 E2B spends its retained tail consuming shared `(keys, values)` pairs
  without mutating cache. `mlx-lm` feeds those shared tensors straight into
  attention, while our path still wraps them in a borrowed
  `TransformerCacheView` before SDPA. If shared consumers bypass the cache-view
  wrapper entirely, we might shave enough repeated wrapper churn off the shared
  tail to narrow the paired decode gap.
- Success criteria:
  - focused Gemma 4 tests stay green
  - transformers typecheck stays green
  - the canonical paired benchmark beats a same-window baseline control
- Rollback rule:
  if the result is only marginally positive or fails a reverse-order
  confirmation run, remove it and keep only the logged result

#### Outcome

- Result:
  rejected after bracketed confirmation
- Measurement:
  - focused Gemma 4 tests: passed
  - transformers typecheck: passed
  - first canonical `5`-trial pair:
    - experimental path: `mlxts=83.000`, `mlx-lm=93.280`, gap `-10.280`,
      ratio `0.890`
  - immediate same-window baseline control:
    - baseline: `mlxts=82.914`, `mlx-lm=93.537`, gap `-10.623`,
      ratio `0.886`
  - reverse-order confirmation:
    - experimental path: `mlxts=80.958`, `mlx-lm=92.780`, gap `-11.822`,
      ratio `0.873`
- Conclusion:
  this looked barely positive in one healthy window, but not enough to trust,
  and the reverse-order confirmation erased it completely. That is exactly the
  kind of maybe-win the research loop is meant to protect us from. The result
  is useful anyway: removing the borrowed shared-KV wrapper alone is not the
  tray. If a deeper Gemma 4 shared-consumer optimization exists, it probably
  needs to package the whole shared decode-attention stage, not just skip one
  wrapper object.

### Experiment 34: Native shared-consumer decode-attention stage

- Status: `failed and reverted`
- Hypothesis:
  More than half of Gemma 4 E2B's retained tail is shared-KV consumer work:
  query projection, q-norm, RoPE, SDPA against precomputed shared keys/values,
  then output projection. If we package that whole read-only shared-consumer
  stage into one Bun-safe native helper, we might finally remove enough
  JS-visible seams to close the gap without touching the readable model code.
- Success criteria:
  - native helper builds cleanly with the pointer-only config-buffer ABI shape
  - focused Gemma 4 tests and transformers/core typecheck stay green
  - the canonical paired benchmark beats a same-window baseline control
- Rollback rule:
  if the experimental path loses cleanly to the same-window baseline, remove
  the helper and keep only the benchmark result and ABI lesson

#### Outcome

- Result:
  rejected after same-window control
- Measurement:
  - core typecheck: passed
  - transformers typecheck: passed
  - focused Gemma 4 tests: passed
  - canonical `5`-trial experimental pair in a soft machine window:
    - experimental path: `mlxts=70.504`, `mlx-lm=82.793`, gap `-12.289`,
      ratio `0.851`
  - immediate same-window baseline control:
    - baseline: `mlxts=81.195`, `mlx-lm=77.670`, gap `+3.525`,
      ratio `1.045`
- Conclusion:
  this was the right ambitious seam to test, and the benchmark still said no.
  The helper was stable, the Bun ABI shape was fine, and the math checked out,
  but the real decode result was dramatically worse than the same-window
  baseline. That makes this a strong negative result, not an ambiguous one.
  The important thing we keep is the implementation lesson: a pointer-only
  config-buffer native helper can be built and validated safely, but this
  particular shared-consumer attention tray did not help Gemma 4 parity.

### Experiment 35: Grow full-cache buffers by extending the live backing array

- Status: `kept as the current winner`
- Hypothesis:
  The cleanest remaining Gemma 4 cliff was not inside the steady sliding ring;
  it was at the first generated token after a full-cache chunk boundary. Our
  `KVCache` growth path was allocating a brand-new larger zero buffer and
  copying the entire retained prefix into it, while `mlx-lm` extends the live
  full-cache buffer with a zero chunk and keeps the old prefix in place. If we
  match that growth shape, we should reduce the `prompt_tokens=1024 -> first
  decode token` tax without changing semantic cache surfaces.
- Success criteria:
  - focused cache tests and transformers typecheck stay green
  - the `1023 -> 1024` short-run decode cliff shrinks materially
  - the canonical paired benchmark beats a same-weather reverted baseline
- Rollback rule:
  if the cliff does not shrink or a same-weather reverted control is comparable
  or better, remove the growth-path change immediately

#### Outcome

- Result:
  accepted for the current baseline
- Measurement:
  - focused cache tests: passed
  - transformers typecheck: passed
  - pre-patch boundary probe:
    - `prompt_tokens=1023`, `generation_tokens=8`, `trials=5`:
      `mlxts=88.406`, `mlx-lm=77.596`, gap `+10.810`, ratio `1.139`
    - `prompt_tokens=1024`, `generation_tokens=8`, `trials=5`:
      `mlxts=73.046`, `mlx-lm=76.698`, gap `-3.652`, ratio `0.952`
    - `mlxts` cliff across the `1023 -> 1024` boundary:
      `-15.360 tok/s`
  - patched same-window boundary probe in a softer machine state:
    - `prompt_tokens=1023`, `generation_tokens=8`, `trials=5`:
      `mlxts=73.684`, `mlx-lm=65.236`, gap `+8.448`, ratio `1.129`
    - `prompt_tokens=1024`, `generation_tokens=8`, `trials=5`:
      `mlxts=69.023`, `mlx-lm=63.626`, gap `+5.397`, ratio `1.084`
    - patched `mlxts` cliff across the same boundary:
      `-4.661 tok/s`
  - patched canonical `1024 / 128 / 3` pair in the same soft window:
    - `mlxts=65.952`, `mlx-lm=73.919`, gap `-7.967`, ratio `0.892`
  - same-weather reverted control immediately after removing the patch:
    - `mlxts=48.208`, `mlx-lm=64.073`, gap `-15.865`, ratio `0.752`
- Conclusion:
  this is the first experiment in a while with a strong, mechanically
  explainable win that also survives a same-weather reverted control. The exact
  absolute TPS numbers were depressed by machine state, but the shape of the
  result was clear: the `1024` boundary tax is real in `mlxts`, and extending
  the full-cache buffer in place-like fashion removes a large part of it. This
  growth-path change stays in the branch as the new baseline for the next
  experiment round.

### Experiment 36: Stack a native attention-output projection seam on top of the cache-growth win

- Status: `failed and reverted`
- Hypothesis:
  With the full-cache growth cliff reduced, the strongest remaining small seam
  from earlier rounds was still the attention output tail:
  `transpose -> reshape -> output projection`. Re-testing that seam on top of
  the Experiment 35 baseline might finally let the earlier near-win stack into
  a real canonical improvement.
- Success criteria:
  - focused core fast-kernel tests and Gemma 4 attention tests stay green after
    a real native rebuild
  - core and transformers typecheck stay green
  - the canonical paired benchmark improves relative to the Experiment 35
    baseline strongly enough to justify a same-weather control
- Rollback rule:
  if the canonical run is only flat or worse, remove the helper immediately and
  keep the Experiment 35 baseline clean

#### Outcome

- Result:
  rejected after canonical run
- Measurement:
  - forced native rebuild: passed
  - focused core fast and Gemma 4 attention tests: passed
  - core and transformers typecheck: passed
  - short paired sanity run (`1024 / 32 / 1`):
    - `mlxts=84.353`, `mlx-lm=92.121`, gap `-7.768`, ratio `0.916`
  - canonical paired run (`1024 / 128 / 3`):
    - `mlxts=81.230`, `mlx-lm=92.332`, gap `-11.102`, ratio `0.880`
- Conclusion:
  the helper stayed correct and the seam was still readable, but the canonical
  result did not justify keeping it. This is another useful negative result:
  even on top of the cache-growth win, the isolated output-projection native
  seam is not enough to beat the stronger baseline. It was removed before the
  next round.

### Experiment 37: Disable EOS early stopping in the `mlx-lm` reference helper

- Status: `kept`
- Hypothesis:
  Our parity loop always measures a fixed number of greedy decode steps, but
  the Python `mlx-lm` helper was using `stream_generate()` with the tokenizer's
  normal EOS ids still enabled. On synthetic long prompts, that can stop the
  reference run early and report nonsense `generation_tps`, especially once the
  prompt gets large enough for argmax to stumble into EOS. If we make the
  helper match `mlx-lm`'s own benchmark behavior by clearing EOS ids first, the
  paired metric should become stable again and reveal the true Gemma 4 gap.
- Success criteria:
  - the benchmark helper and benchmark-common tests stay green
  - long-context reference runs stop producing impossible `generation_tps`
  - the canonical paired benchmark is re-measured under the fixed reference
    harness before any new model-path experiment is attempted
- Rollback rule:
  if disabling EOS does not materially stabilize the paired metric, revert the
  helper and keep treating long synthetic runs as untrustworthy

#### Outcome

- Result:
  accepted as the new benchmark baseline
- Measurement:
  - benchmark-common tests: passed
  - transformers typecheck: passed
  - before the helper fix, long synthetic reference runs could report impossible
    decode throughput, e.g. `prompt_tokens=32000`, `generation_tokens=64`:
    - `mlx-lm=36697.248`, `mlxts=64.505`
  - after disabling EOS in the helper, the same style of long-context runs
    became sane again:
    - `prompt_tokens=10000`, `generation_tokens=64`, `trials=1`:
      `mlx-lm=78.407`, `mlxts=80.281`, gap `+1.874`, ratio `1.024`
    - `prompt_tokens=32000`, `generation_tokens=64`, `trials=1`:
      `mlx-lm=71.967`, `mlxts=72.813`, gap `+0.846`, ratio `1.012`
  - canonical paired confirmation under the fixed harness:
    - `prompt_tokens=1024`, `generation_tokens=128`, `trials=3`:
      `mlx-lm=81.152`, `mlxts=82.250`, gap `+1.098`, ratio `1.014`
  - immediate repeat canonical confirmation under the same fixed harness:
    - `prompt_tokens=1024`, `generation_tokens=128`, `trials=3`:
      `mlx-lm=81.333`, `mlxts=82.039`, gap `+0.706`, ratio `1.009`
- Conclusion:
  this is not a Gemma 4 model-path speedup, but it is a decisive parity
  research win. The earlier long-context and canonical gap numbers were being
  distorted by a reference-side benchmark bug, not just by `mlxts` runtime
  overhead. Once the helper matches `mlx-lm`'s own benchmark posture and forces
  a fixed-length decode window, Gemma 4 on the current branch is already at or
  slightly above paired parity in repeated healthy-window confirmations, and it
  also stays strong as decode length grows to `1000` and `10000` tokens in the
  single-run checks recorded here. Future experiments should use this fixed
  harness; otherwise the research loop will optimize against a moving lie.
