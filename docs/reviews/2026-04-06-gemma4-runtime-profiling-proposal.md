# Runtime Review: Gemma 4 Contract Refactor and Compile Spike

## Status

This review remains useful evidence for the compile-first branch, but it now
belongs to the earlier Gemma 4 runtime track. Read it together with the later
runtime retrospective and readability-restructure proposal before treating any
of its next-step framing as current guidance.

## Summary

This branch now does six deliberate things:

1. internal-only runtime profiling for the current Gemma 4 decode path
2. a private cache/attention contract refactor so runtime code no longer
   reconstructs cache-return ownership independently in each attention family
3. a bounded compile-first sweep over Gemma-family gated activations and
   deterministic sampling helpers, now expressed through semantic helper names
   instead of family-local `compiled*` wrappers
4. a compile ergonomics cleanup so package code keeps semantic names and any
   shape-memoized transform reuse stays local to the consumer instead of
   becoming a new public core primitive
5. a private cache-view follow-up that uses borrowed full-buffer internal views
   on the hottest sliding-cache path, so attention code no longer pays owned
   return churn when it only needs the view for the current forward step
6. a repo-wide runtime lock for heavy MLX commands so benchmarks, soak runs,
   acceptance runs, and long training/proof commands cannot contend on one
   machine

It also keeps two small hot-path cleanups that were already justified
independently:

- Gemma 4 attention borrows externally-owned shared masks instead of retaining
  and freeing them per layer
- `fast.rope()` propagates metadata so downstream hot-path reads avoid
  unnecessary shape and dtype queries

This is still not a native-cache remediation branch. The new compile step
matters because it attacks the broader Bun + FFI boundary cost before the repo
pays the maintenance cost of raw-pointer or custom-C++ cache work.

The benchmark scripts still reset the runtime profile after the first token has
already been built. That makes the profile directionally useful and not exact
per-token attribution. It is good enough for branch ordering, not for claiming
that one bucket owns a fixed percentage of the total gap.

## Files Reviewed

- `packages/core/src/array.ts`
- `packages/core/src/fast.ts`
- `packages/core/src/index.ts`
- `packages/core/src/runtime-profile.ts`
- `packages/core/src/transforms-compile.ts`
- `packages/core/src/transforms.ts`
- `packages/core/src/typed-array-copy.ts`
- `examples/nanogpt/src/bench/memory.ts`
- `examples/nanogpt/src/run/acceptance.ts`
- `examples/nanogpt/src/run/soak.ts`
- `examples/nanogpt/src/run/supervisor.ts`
- `packages/nn/src/activations.ts`
- `packages/nn/src/losses.ts`
- `packages/transformers/scripts/benchmark-generation.ts`
- `packages/transformers/scripts/benchmark-generation-parity.ts`
- `packages/transformers/src/families/gemma3/attention.ts`
- `packages/transformers/src/families/gemma3/mlp.ts`
- `packages/transformers/src/families/gemma4/attention.ts`
- `packages/transformers/src/families/gemma4/block.ts`
- `packages/transformers/src/families/gemma4/mlp.ts`
- `packages/transformers/src/families/llama-like/attention.ts`
- `packages/transformers/src/families/llama-like/mlp.ts`
- `packages/transformers/src/infrastructure/cache-ops.ts`
- `packages/transformers/src/infrastructure/cache.ts`
- `packages/transformers/src/infrastructure/cache-view.ts`
- `packages/transformers/src/infrastructure/gated-activations.ts`
- `packages/transformers/src/infrastructure/runtime-profile.ts`
- `packages/transformers/src/infrastructure/sampling.ts`

## Tensor Lifetime Audit

The profiling additions do not change ownership semantics for ordinary
`MxArray` instances. All runtime-sensitive paths remain on the tracked-array
model with per-call `OutSlot` ownership at the FFI boundary.

The private `TransformerCacheView` contract makes cache-return ownership
explicit in one place instead of being manually reconstructed in Gemma 3,
Gemma 4, and llama-like attention code. The public `TransformerCache` surface
is unchanged in this branch.

The shared-mask cleanup in Gemma 4 attention removes per-layer
`retainArray()`/free churn for externally-owned masks, but it does not weaken
ownership safety. The model-level mask creator already owns the mask lifetime,
and the attention layer now frees only masks it creates locally in the same
lexical scope.

The `fast.rope()` metadata propagation seeds shape and dtype metadata on the
returned tracked wrapper. It does not change disposal, aliasing, or evaluation
semantics.

The GEGLU transform reuse is process-lifetime on purpose. It is a deliberate
shared runtime primitive for this model family, not a request-scoped transform.
Its outputs are still ordinary tracked `MxArray` values and stay under the same
explicit disposal rules as the rest of the forward pass.

The shape-memoized `crossEntropy` reuse path does not change array ownership
semantics. It keeps compiled transform variants local to the loss helper
instead of introducing a new public transform concept in `@mlxts/core`.

The borrowed cache-view follow-up does change the private internal cache-view
contract slightly. A borrowed `TransformerCacheView` may now alias mutable
cache state and should not outlive a later mutation of the same cache layer.
That is acceptable for the actual attention hot path, because the view is used
and disposed inside the same forward step. Callers that need stable ownership
across later cache mutations must call `materializeOwnedPair()`.

The runtime lock lives in script entrypoints rather than tensor code. It does
not change MLX array ownership; it only prevents multiple heavy MLX programs
from running concurrently on one machine unless they are intentionally nested
under the same inherited lock token.

## Memory / Performance Evidence

Directional baseline from the earlier clean branch:

- `bench:generation`
  - `generation_tps ≈ 75.536`
- `bench:generation:parity`
  - `generation_tps ≈ 76.495`

Current branch after the cache-view refactor, selective compile sweep, local
deterministic-sampling transform reuse, and borrowed full-buffer internal cache
views:

- `bun run bench:generation --model google/gemma-4-E2B-it --prompt-tokens 1024 --generation-tokens 128 --trials 1`
  - `prompt_tps=8496.515`
  - `generation_tps=79.598`
  - `peak_memory=9.907 GB`
  - `evals_per_token=1.00`

- `bun run bench:generation:parity --model google/gemma-4-E2B-it --prompt-tokens 1024 --generation-tokens 128 --trials 1`
  - `prompt_tps=8219.995`
  - `generation_tps=81.901`
  - `peak_memory=9.907 GB`
  - `evals_per_token=1.00`

- `bun run bench:generation:parity --model google/gemma-4-E2B-it --prompt-tokens 1024 --generation-tokens 128 --trials 3`
  - `prompt_tps=8180.411`
  - `generation_tps=80.737`
  - `peak_memory=9.907 GB`
  - `evals_per_token=1.00`

This is a real directional improvement of about `6–7%` over the prior Gemma 4
decode plateau without any native-cache work.

The newer cache-view follow-up after that compile win improves the profiled
steady-state host-side buckets again, but the 128-token Gemma headline still
lands in roughly the low `81 tok/s` range rather than jumping decisively
higher. That means the seam change is worth keeping, but it should be treated
as a structural cleanup plus bucket improvement, not as the fix that closes
Gemma parity.

Control-model sanity on the same branch:

- `bun run bench:generation:parity --model meta-llama/Llama-3.2-1B-Instruct --prompt-tokens 1024 --generation-tokens 128 --trials 1`
  - `prompt_tps=6580.291`
  - `generation_tps=175.581`
  - `peak_memory=2.937 GB`
  - `evals_per_token=1.00`

This matters because it suggests the compile-first runtime program is helping
the broader stack, while the remaining Gemma gap is becoming more
model/cache-specific.

Smaller decode-only profiled parity run:

- `MLXTS_RUNTIME_PROFILE=1 bun run bench:generation:parity --model google/gemma-4-E2B-it --prompt-tokens 1024 --generation-tokens 32 --trials 1`
  - `generation_tps=85.175`
  - `peak_memory=9.907 GB`
  - steady-state decode profile:
    - `out_slot_ms_per_token=0.0333`
    - `ffi_ms_per_token=0.3833`
    - `wrapper_ms_per_token=0.5127`
    - `free_ms_per_token=0.1369`
  - selected steady-state cache counters:
    - `cache.sliding_single_token=11.250/token`
    - `cache.write_range=28.125/token`
    - `cache.return_prefix_view=5.625/token`
    - `cache.return_full_buffer=22.500/token`
    - `cache.buffer_replaced=28.125/token`

The evidence supports three conclusions:

1. there is a real broader host-side per-op tax in the Bun + FFI stack
2. a bounded compile pass can remove a meaningful slice of that tax before any
   native-cache work begins
3. the cache path is still a meaningful structural contributor inside the
   remaining tax, but the remaining Gemma work is now likely to require either
   native cache help or a similarly deeper change than another small
   TypeScript-only cache contract tweak

The evidence does not prove that cache alone owns a fixed percentage of the
gap, and the profile should not be treated that way.

The runtime lock, compile-ergonomics cleanup, and deterministic top-k/top-p
follow-up are primarily safety, readability, and sampling cleanup. The bucket
improvement described above comes from putting the private cache-view seam to
work on the hot sliding-cache path, not from those script and naming changes.

## Independent Review

The direction of this branch was informed by independent external reviews from
Claude, Codex, and GPT 5.4 Pro. The reviewers converged on three important
points:

- the next remediation should not be another broad speculative experiment
- the cache contract still matters, but it should not automatically outrank a
  working compile primitive that can collapse broader FFI cost first
- the remaining work should be driven by a reviewed proposal that explicitly
  weighs compile, cache-contract work, native-binding options, and broader
  ownership changes

The proposal under `docs/proposals/` has been updated to reflect that order:
contract cleanup first, compile spike second, then cache-native work only if
the remaining headroom still justifies it.

Local verification completed on this branch:

- `bun run --filter @mlxts/transformers typecheck`
- `bun test packages/transformers/src/infrastructure/cache.test.ts packages/transformers/src/infrastructure/cache-view.test.ts`
- `bun test packages/core/src/transforms.test.ts packages/nn/src/activations.test.ts packages/nn/src/losses.test.ts packages/nn/src/integration.test.ts packages/transformers/src/infrastructure/gated-activations.test.ts packages/transformers/src/infrastructure/sampling.test.ts`
- `bun run validate`

## Remaining Risks / Follow-ups

- The compile spike is still bounded. It proves ordering, not that compilation
  alone will close the entire Gemma 4 gap.
- The next compile expansion, if any, should remain selective: MLP-local or
  block-local motifs before any attempt to compile larger decoder regions.
- The deeper native research remains historically important, but it is not the
  current default mainline direction. The shallow native seam experiments were
  informative and unstable. Any future native resume should happen only behind
  a clearer backend seam and after the readable inference surface is restored.
- The machine currently lacks a working live `mlx-lm` Python environment for
  parity runs, so fresh direct reference capture should be restored before the
  eventual implementation branch is judged complete.
