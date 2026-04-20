# Runtime Research Retrospective: Compile, Cache, And Native Seam Work

## Summary

This document preserves what the recent runtime work actually taught us before
the repo is cleaned back toward a more readable mainline.

The goal is not to justify every experiment. The goal is to keep the useful
evidence:

- what measurably helped
- what failed
- what the failures imply about the right execution boundary
- what should remain part of the architecture after cleanup

This retrospective should be read together with
[`docs/proposals/2026-04-08-readable-runtime-restructure.md`](../proposals/2026-04-08-readable-runtime-restructure.md),
which turns these lessons into the next repo shape.

## Files Reviewed

This retrospective summarizes work across:

- `packages/core/src/transforms-*`
- `packages/core/native/mlxts_core_ops.cpp`
- `packages/core/src/ffi/*`
- `packages/core/src/ffi/closure-bridge.ts`
- `packages/core/src/transforms-base.ts`
- `packages/nn/src/activations.ts`
- `packages/nn/src/losses.ts`
- `packages/nn/src/activations/index.ts`
- `packages/nn/src/activations/runtime.ts`
- `packages/nn/src/losses/index.ts`
- `packages/nn/src/losses/runtime.ts`
- `packages/transformers/src/infrastructure/cache*.ts`
- `packages/transformers/src/infrastructure/sampling.ts`
- `packages/transformers/src/infrastructure/cache/index.ts`
- `packages/transformers/src/infrastructure/cache/ops.ts`
- `packages/transformers/src/infrastructure/cache/runtime.ts`
- `packages/transformers/src/infrastructure/cache/view.ts`
- `packages/transformers/src/infrastructure/gated-activations/index.ts`
- `packages/transformers/src/infrastructure/gated-activations/runtime.ts`
- `packages/transformers/src/infrastructure/generation-defaults.ts`
- `packages/transformers/src/infrastructure/generation/defaults.ts`
- `packages/transformers/src/infrastructure/generation/helpers.ts`
- `packages/transformers/src/infrastructure/generation/runtime.ts`
- `packages/transformers/src/infrastructure/sampling/index.ts`
- `packages/transformers/src/infrastructure/sampling/runtime.ts`
- `packages/transformers/src/families/gemma4/*`
- `packages/transformers/src/families/gemma4/runtime/attention.ts`
- `packages/transformers/src/families/gemma4/runtime/mlp.ts`
- `packages/transformers/src/families/gemma4/runtime/model.ts`
- related runtime benchmark scripts and runtime-oriented docs

It is intentionally higher level than a line-by-line review. The point is to
capture the decisions and lessons before the structure changes again.

## Tensor Lifetime Audit

The lasting mainline changes from this research were reviewed against the repo's
visible-lifetime rule.

- compile-first helper work kept tensor-producing intermediates explicit in the
  surviving TypeScript paths
- the cache-view seam clarified borrowed versus owned cache tensors rather than
  hiding that ownership behind family-specific guesses
- the unstable native cache seam was not left as an active default path, so the
  benchmark-instability investigation did not ship as hidden production
  lifetime behavior

## Memory / Performance Evidence

The meaningful performance evidence from this research was:

- selective compile moved Gemma 4 decode from the earlier mid-70 tok/s plateau
  into the low-80 tok/s range
- compile materially reduced broad host-side overhead, which narrowed the
  remaining problem from "generic TypeScript overhead everywhere" to a smaller
  set of cache and attention-boundary concerns
- native cache-only and fused native cache-plus-attention experiments could
  work in tiny repros, but remained unstable under the real Gemma 4 parity
  benchmark

The exact benchmark numbers varied by branch state and benchmark shape, but the
architectural conclusion held: compile-first produced real value, and the
shallow native seam did not prove stable enough for mainline use.

## Independent Review

This retrospective reflects multiple independent review passes during the
runtime investigation, including fresh-eyes critique of the compile ordering,
cache boundary choices, and the later readability-first restructuring that
followed from these results.

## What We Attempted

The runtime work happened in four broad waves.

### 1. Compile-first hot-path cleanup

We expanded selective compile across repeated pure tensor motifs:

- activation helpers
- Gemma-family gated helpers
- sampling preprocess helpers
- parts of the Gemma 4 attention path
- later, multi-output compile experiments for larger Gemma 4 motifs

This was the right first lever because it attacked broad Bun/FFI graph-building
overhead without changing model semantics.

### 2. Cache ownership cleanup

We introduced a private cache-view seam inside `@mlxts/transformers` so model
families no longer guessed ownership independently.

This made two things explicit:

- when the cache owns the returned view
- when the caller owns the returned tensor

That seam was worth adding even though it did not close the full parity gap.

### 3. Native cache helper exploration

We tried to move the hot mutable cache state machine lower:

- private native in-place helpers
- opaque native cache handle experiments
- native-backed cache classes
- native fetch/update paths

These experiments proved that the cache algorithm itself could be expressed
natively and that tiny targeted scenarios could work.

### 4. Deeper native seam exploration

When the native cache-only seam still looked fragile, we tried a fused native
cache-plus-attention helper so the cache state would be consumed natively
instead of handing lazy cache tensors back to the JS decode graph.

Again, tiny repros could be made to work. The real Gemma 4 parity benchmark
still remained unstable.

## What Worked

### Compile-first was real

The compile-first program delivered a real decode win. The earlier Gemma 4
plateau in the mid-70 tok/s range moved into the low-80s.

The exact headline varied by branch state and benchmark shape, but the
important result held:

- compile materially reduced broad host-side overhead
- the repo should keep compile as a first-class runtime lever for repeated pure
  tensor motifs

That conclusion survives cleanup.

### The cache-view seam was worth adding

The private cache-view seam made ownership rules clearer and reduced some cache
path churn.

That work should survive because it improved architecture, not just the
benchmark.

### The runtime lock was the right preventive rule

Heavy MLX commands now have a hard contention gate instead of relying on agent
discipline. That should remain part of the repo's safety posture.

### The experiments narrowed the problem

By the end of the compile-first work, the remaining Gemma 4 gap no longer
looked like generic TypeScript overhead everywhere. The remaining cost looked
much more concentrated around:

- cache update and view semantics
- attention-adjacent fused native paths
- the boundary between Bun FFI and lazy MLX arrays

Even the failed native work was useful because it falsified a shallow native
seam.

## What Did Not Work

### Native cache alone was not a stable enough seam

The native cache experiments could be made to work in isolated repros, but the
real multi-step Gemma 4 benchmark still produced Bun crashes.

That means "move the cache into C++ and hand lazy arrays back to JS" is not a
safe enough boundary for this stack, at least not in the form we tried.

### Fused native cache-plus-attention was still not benchmark-safe

The deeper fused native helper was directionally better than native cache
alone, but the real parity benchmark still remained unstable.

So the lesson is not "native is impossible." The lesson is:

**the chosen native seam was still wrong or still too shallow.**

### Mainline readability degraded too much

The optimization work drifted into model-family files heavily enough that parts
of the repo stopped reading like a clear ML implementation and started reading
like execution-strategy experiments.

That matters because the project goal is not benchmark chasing alone. The repo
must remain a place where a human can understand training and inference by
reading the code.

## What The Failures Likely Mean

The strongest current inference is:

**the unstable part is the boundary, not the native cache math.**

More specifically, the research suggests that returning lazy MLX arrays from
native-owned mutable state back into the wider Bun-driven decode graph is where
things become fragile under the full Gemma 4 benchmark.

That does not prove the next answer is a full native decode path. It does prove
that the repo should not keep patching the same shallow seam while mainline
readability gets worse.

## Architectural Conclusions To Keep

These conclusions should survive the cleanup:

1. compile belongs early in the optimization order for repeated pure
   tensor-to-tensor motifs
2. cache ownership needs an explicit private seam
3. runtime strategy is not model identity
4. TypeScript should keep the readable orchestration and model semantics
5. optimized execution strategy should live beneath semantic surfaces, not
   spread through the teaching surface of the repo
6. native work remains in-bounds, but only behind a cleaner backend boundary

## What Should Happen Next

The repo should now pivot from "keep optimizing in place" to
"preserve the lessons, then rebuild the runtime structure cleanly."

That means:

1. keep this retrospective as the factual record of the recent work
2. clean the mainline back toward readable inference and training surfaces
3. preserve compile and native research behind clearer backend boundaries
4. only resume deep performance work once the readable reference surfaces and
   backend seams are explicit again

## Remaining Risks / Follow-ups

The main follow-up risk is interpretive, not immediate correctness:

- the repo should not forget that the native failures taught us something real
  about seam depth, even though the readability-first cleanup parked deeper
  performance work
- future performance work should restart from the cleaner semantic/backend
  boundary rather than resurrecting the old shallow native-cache-only path
- historical docs from this period should remain clearly marked as history so
  they are not mistaken for the active mainline direction
