# Runtime Review: Gemma 4 Sampling And KEqV Cleanup

## Summary

This patch makes two small hot-path fixes without changing the repo's readable
surface structure.

The first fix removes a real double-application bug in GPU-side sampling when
repetition penalty and probability filtering were both enabled. The second fix
removes duplicate `kProjection` work in Gemma 4's `attentionKEqV` path by
sharing one projected-key tensor across key and value preparation.

The goal here is correctness and contained hot-path waste removal, not another
structural runtime experiment. The semantic model code stays the same.

## Files Reviewed

- `packages/transformers/src/infrastructure/sampling/index.ts`
- `packages/transformers/src/infrastructure/sampling/runtime.ts`
- `packages/transformers/src/families/gemma4/runtime/attention.ts`
- `packages/transformers/src/families/gemma4/attention.ts`

## Tensor Lifetime Audit

The sampling fix does not add any new ownership surfaces. `SamplerState` still
owns the temporary `repetitionAdjusted` tensor inside `sampleTokenTensor()`,
and the filtered branch now passes that owned tensor forward instead of
recomputing the same penalty transform from the original logits.

The Gemma 4 attention fix keeps tensor lifetimes explicit. The new
`prepareKeyValueHeads()` helper returns owned key/value tensors, and
`Gemma4TextAttention.buildFreshKeyValues()` frees both in the same lexical
scope after either:

- retaining them into a standalone cache view, or
- appending them into the live cache and borrowing the returned view

The `attentionKEqV` branch no longer allocates a second projected-key tensor
just to prepare values. No borrowed cache ownership rules changed.

## Memory / Performance Evidence

These fixes touch generation hot paths, but they are intentionally narrow.
Only the Gemma 4 `attentionKEqV` change affects the default greedy benchmark
path directly. The sampling fix affects non-greedy filtered generation
(`topK` / `topP` / `minP` plus repetition penalty), so the canonical greedy
Gemma 4 benchmark is not expected to show a headline jump from that change
alone.

Fresh sequential measurements on cached `google/gemma-4-E2B-it` from this
branch:

- `bench:generation`
  - command:
    `bun run bench:generation --model google/gemma-4-E2B-it --prompt-tokens 1024 --generation-tokens 128 --trials 3`
  - averages:
    - `prompt_tps=8648.120`
    - `generation_tps=79.387`
    - `peak_memory=9.894 GB`
    - `evals_per_token=1.00`
- `bench:generation:parity`
  - command:
    `bun run bench:generation:parity --model google/gemma-4-E2B-it --prompt-tokens 1024 --generation-tokens 128 --trials 3 --mlx-lm-python .tmp/venvs/mlx-lm-bench/bin/python`
  - captured reference:
    - `mlx-lm prompt_tps=1025.554`
    - `mlx-lm generation_tps=85.231`
    - `mlx-lm peak_memory=9.889 GB`
  - mlxts averages:
    - `prompt_tps=8161.994`
    - `generation_tps=79.656`
    - `peak_memory=9.894 GB`
    - `evals_per_token=1.00`

Interpretation:

- the Gemma 4 greedy decode headline remains roughly where the cleaned baseline
  already was; these changes do not introduce a regression in the one-eval
  steady-state decode invariant
- the sampling fix is still worth carrying because it removes real extra work
  on filtered generation paths that the canonical greedy benchmark does not
  exercise
- the KEqV fix removes a wasted matmul from the alternative-attention path
  without changing the semantic family surface

Targeted regression coverage added with this patch:

- `SamplerState combines repetition penalty with top-k filtering` now pins the
  filtered-sampling call path so the shared filtered branch cannot silently
  drop or bypass the already-penalized logits tensor
- `Gemma4TextAttention reuses the key projection for attentionKEqV
  full-attention layers` now asserts that the KEqV path only calls
  `kProjection.forward()` once
- `Gemma4TextAttention keeps the cache-backed attentionKEqV path valid after
  key/value tensors are released` now exercises the cache-backed ownership path
  and checks that cached keys and values still match the eager reference pair

## Independent Review

An external fresh-eyes review was requested from Kimi around this exact scope.
The independent findings matched the implementation choices here:

- the double repetition-penalty path was confirmed as a real bug
- the duplicate `kProjection` work in the `attentionKEqV` path was confirmed as
  real
- the sliding-window step-mask suspicion was called a false positive for
  steady-state decode because the current fast path already returns `null`
- the repetition-history growth concern and the MLP transpose concern were both
  judged real but better treated as separate follow-up work rather than folded
  into this patch

That review did not surface a blocker against the implemented fix set.

## Remaining Risks / Follow-ups

- The repetition-history growth issue in `SamplerState` is still real for long
  filtered generations, but it was deliberately left out of this patch because
  it needs a clearer product decision about history bounds or a dedicated
  ring-buffer design rather than a rushed hot-path tweak.
- Gemma 4 `runtime/mlp.ts` still has a smaller hot-path inefficiency around
  weight transposes inside the compiled closure. That is a follow-up candidate,
  not part of this correctness-first patch.
- The canonical greedy Gemma 4 parity gap remains open. These fixes reduce
  specific wasted work, but they do not change the broader conclusion that
  Gemma 4 is still the only major supported family materially behind `mlx-lm`
  on decode throughput.
