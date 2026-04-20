# Proposal: Gemma 4 Decode Parity Refactor

## Status

This is now a historical proposal. Keep it as the record of the earlier Gemma 4
runtime direction, but read it with
[`docs/proposals/2026-04-08-readable-runtime-restructure.md`](./2026-04-08-readable-runtime-restructure.md)
and
[`docs/reviews/2026-04-08-runtime-runtime-research-retrospective.md`](../reviews/2026-04-08-runtime-runtime-research-retrospective.md)
for the current mainline direction.

## Summary

`google/gemma-4-E2B-it` is still materially slower in steady-state decode on
`mlxts` than on `mlx-lm` on the same Apple Silicon machine. Dense correctness is
already fixed. The remaining work is no longer a correctness task; it is a
runtime architecture task.

This proposal exists to prevent the next step from turning into another
speculative performance branch. It frames the remaining gap as a deliberate
refactor decision that must be reviewed against upstream references, Bun
runtime constraints, and the repo's package-boundary rules before any deeper
remediation begins.

This proposal now sits alongside the broader system document
[`docs/proposals/2026-04-06-runtime-execution-architecture.md`](./2026-04-06-runtime-execution-architecture.md),
which captures how the Gemma 4 findings should change runtime design across
the stack.

It should now also be read with
[`docs/reviews/2026-04-08-runtime-runtime-research-retrospective.md`](../reviews/2026-04-08-runtime-runtime-research-retrospective.md)
and
[`docs/proposals/2026-04-08-readable-runtime-restructure.md`](./2026-04-08-readable-runtime-restructure.md).
Those later documents record that the shallow native seam explored after this
proposal was informative but not stable enough to remain the active mainline
path.

This is not a pure measurement branch. It is a proposal branch plus two scoped
hot-path cleanups that were already justified independently:

- Gemma 4 attention now borrows externally-owned shared masks instead of
  retaining and freeing them per layer
- `fast.rope()` now propagates known metadata so downstream hot-path code avoids
  unnecessary shape and dtype queries

## Problem Statement

Current clean-base parity numbers on cached `google/gemma-4-E2B-it`:

- `mlxts` decode: about `76.5 tok/s`
- `mlx-lm` decode: about `90.6 tok/s`
- peak memory: effectively tied at about `9.9 GB`

The gap is therefore about `16–18%` in steady-state decode.

Known facts:

- Gemma 4 dense correctness is fixed.
- The main fused kernels are already in use:
  - scaled dot product attention
  - RMSNorm
  - RoPE
- The decode loop already uses the intended async schedule and one scalar sync
  per token.
- Small hygiene fixes did not materially close the headline gap.

The remaining issue is therefore not "wrong math." It is the runtime cost of
how the graph is built, wrapped, updated, and observed during steady-state
decode.

## Decision Factors

Every proposed remediation should be reviewed against these three factors.
These are the factors that second and third audits should explicitly use.

### 1. Performance Leverage

How much of the actual Gemma 4 decode gap can the change plausibly remove, and
is that claim backed by measurements rather than intuition?

### 2. Architectural Fit

Does the change preserve the repo's intended shape?

That means:

- MLX-C first, JS fallback last
- explicit ownership over magic abstractions
- no public API widening for an unproven hot-path idea
- improvements generalized only when the evidence is real
- runtime-sensitive lifetimes still visible in code

### 3. Maintenance and Audit Cost

Does the change create ongoing ABI, review, or correctness burden that is
larger than the performance benefit?

This is especially important for:

- Bun FFI ownership changes
- custom `mlx-c` bindings
- new public contracts in `@mlxts/core`, `@mlxts/nn`, or `@mlxts/transformers`

## Current Evidence

### Measured clean-base behavior

The current branch adds internal-only runtime profiling to the clean Gemma 4
parity baseline. The measurement work is intentionally behind
`MLXTS_RUNTIME_PROFILE=1` and does not widen the product surface.

Post-restart sequential benchmark on the current branch:

- `bun run bench:generation:parity --model google/gemma-4-E2B-it --prompt-tokens 1024 --generation-tokens 128 --trials 3`
- `mlxts` decode average: `76.495 tok/s`
- `mlx-lm` live reference was unavailable on this machine in that run because
  the Python `mlx_lm` package was not installed in the active environment
  (`ModuleNotFoundError`), so the `mlx-lm` bar remains the previously captured
  local reference for this machine until a fresh reference run is restored

Smaller profiled parity-oriented run on the current branch:

- `MLXTS_RUNTIME_PROFILE=1 bun run bench:generation:parity --model google/gemma-4-E2B-it --prompt-tokens 1024 --generation-tokens 32 --trials 1`
- decode: `79.711 tok/s`
- approximate post-first-token decode profile:
  - `out_slot_ms_per_token ≈ 0.0402`
  - `ffi_ms_per_token ≈ 0.6645`
  - `wrapper_ms_per_token ≈ 0.6175`
  - `free_ms_per_token ≈ 0.1476`
- top measured FFI labels include:
  - `geluApprox`
  - `matmul`
  - `fast_rms_norm`
  - `fast_rope`
  - `reshape`
  - `multiply`
  - `add`
  - `transpose`

The important conclusion is not the exact decimal split. It is the shape:

- the host-side wrapper and FFI tax is real
- the gap is distributed across many operations, not a single obviously broken
  kernel call

The current profiler window is directionally useful, not exact attribution. The
profile is reset after the first token has already been built, while the
printed normalization still divides by the full generated-token count. That is
good enough for branch selection and bad enough that it should not be treated
as proof that one bucket owns an exact percentage of the gap.

### Current cache-path counters

The same profiled run recorded steady-state decode counters in the Gemma 4
cache path:

- `cache.sliding_single_token ≈ 11.250/token`
- `cache.write_range ≈ 28.125/token`
- `cache.buffer_replaced ≈ 28.125/token`
- `cache.return_full_buffer ≈ 22.500/token`
- `cache.return_prefix_view ≈ 5.625/token`

These counters support "there is real structural churn here." They do not prove
that cache alone owns a fixed percentage of the whole gap. The cache churn is
happening inside a broader per-op host-side tax.

## Reference Audit

### `mlx-c`

Current local pinned version: `v0.6.0`.

Relevant existing low-level ops already exposed in the C headers include:

- `mlx_slice_update`
- `mlx_slice_update_dynamic`
- `mlx_scatter`
- `mlx_put_along_axis`
- `mlx_take_along_axis`

There is no obvious "append token to KV cache and return stable borrowed view"
primitive in the current exposed `mlx-c` surface. That does not rule out a
custom binding, but it means the next step should not assume an upstream C
primitive already exists for the exact cache contract we want.

### `mlx-lm`

`mlx-lm` remains the primary performance reference.

Relevant design patterns:

- full KV caches and rotating caches preserve stable cache objects
- single-token rotating updates use indexed assignment on those cache objects
- compiled transform reuse is used selectively for small composite math such as:
  - `geglu`
  - `logit_softcap`
- `mlx-lm` does not compile the full attention or full decoder pass

The strongest relevant difference for Gemma 4 remains the cache update and
return contract, especially on saturated sliding-window decode.

That difference should be named plainly:

- in `mlx-lm`, saturated single-token cache updates mutate stable cache objects
  and then return the existing cache references directly
- in `mlxts`, the same logical path still goes through array-returning
  `sliceUpdate()` calls, frees the previous buffers, and then often returns
  fresh retained wrappers to the caller

So the implementation target is not simply "reduce retainArray a bit." It is:

**eliminate as much wrapper churn as possible in the saturated single-token
cache-owning decode path while preserving explicit ownership and reviewability.**

### `mlx-swift`

`mlx-swift` is not the benchmark reference, but it is valuable for implementation
discipline and MLX usage patterns. It should inform:

- how we document MLX best practices for TypeScript
- where composition and ergonomics should live versus where hot-path code should
  stay explicit
- how to separate reusable guidance from model-family-specific code

Its `skills/` material should later be reviewed and distilled into TypeScript
guidance after the parity work stabilizes.

### `mlx-swift-lm`

`mlx-swift-lm` is a design cross-check, not a Gemma 4 reference implementation.
It does not currently expose a Gemma 4 implementation, but its cache code is
still useful because:

- it uses stable cache objects
- it keeps update semantics local to the cache abstraction
- it follows the same selective-compile philosophy as `mlx-lm`

## Candidate Refactor Directions

These are the real options. They should be reviewed as refactor candidates, not
ad hoc optimizations.

### Option A: Cache-Contract Refactor

Refactor `packages/transformers/src/infrastructure/cache.ts` so the steady-state
update-and-fetch path stops manufacturing as many fresh owned handles when the
logical cache buffers can remain stable.

What this means:

- keep the public `TransformerCache` surface stable unless review explicitly
  approves a different private boundary
- tighten the internal write/return contract for:
  - full caches
  - saturated rotating caches
  - mixed-pattern layer caches
- make the exact saturated single-token target explicit:
  - fewer returned wrappers
  - fewer retain/free cycles
  - fewer replace-and-return steps per cache-owning layer per token
- align the logical behavior more closely with `mlx-lm` and `mlx-swift-lm`

Strengths:

- directly targets the hottest structural difference still visible in the cache
  counters
- stays private to transformer internals if done carefully
- fits the current evidence best

Risks:

- cache correctness is easy to damage subtly
- the private ownership boundary is not specified enough yet
- may only close part of the gap because the broader host-side per-op tax still
  exists
- `sliceUpdate()` still returns a new array, so a pure TypeScript refactor may
  still have an irreducible wrapper floor unless the internal representation or
  native surface changes

### Option A1: Borrowed Return Contract

Change the cache/model interaction so cache-owned steady-state buffers can be
borrowed by attention instead of always being returned as owned wrappers.

Strengths:

- targets the exact return-side churn called out by the reviews
- may stay private to the cache/attention boundary

Risks:

- requires a precise ownership contract change before implementation
- attention disposal logic must change in lockstep

### Option A2: Raw-Pointer Cache Internals

Keep cache internal state as raw native pointers rather than as JS-owned
`MxArray` wrappers, and only wrap pointers when crossing back into the model
boundary.

Strengths:

- can eliminate most internal cache wrapper churn without immediately requiring
  a custom C++ layer
- keeps the experimental surface narrow and private to cache internals

Risks:

- deeper than a normal cache refactor
- raises the local lifetime burden significantly
- must be reviewed against the repo's explicit-ownership rules very carefully

### Option A3: Stable-Wrapper Pointer Swap

Preserve stable JS-visible cache wrapper objects and swap the underlying native
pointer they own when the cache updates.

Strengths:

- conceptually mirrors the stable-object behavior in Python and Swift
- could eliminate most return-side wrapper churn

Risks:

- this is a foundational ownership trick, not a normal refactor
- requires strong proof that old MLX graph nodes remain valid after pointer
  replacement and explicit free

### Option B: Private Native Cache Helper

Add one custom binding below `@mlxts/core` to express a cache-specific
write-plus-return behavior that the current `mlx-c` surface does not expose
cleanly.

Strengths:

- aligns with MLX-C-first thinking when an exact hot-path primitive is missing
- could reduce TS/Bun wrapper churn more aggressively than a pure TypeScript
  refactor

Risks:

- raises ABI and maintenance burden immediately
- difficult to justify before the cache-contract refactor is better specified
- should not be chosen unless we can point to a concrete existing gap in the
  `mlx-c` surface, not just frustration with Bun overhead

### Option B1: Thin Custom C++ In-Place Cache Write

Add one small native helper that performs the hot cache write in place and keeps
the same cache buffer identity stable across updates.

Strengths:

- most direct path to Python-style cache mutation semantics
- small enough to keep the native surface narrow if carefully scoped

Risks:

- still introduces custom C++ and ABI maintenance burden
- should only happen after the TypeScript/private-contract route is specified

### Option B2: Native Cache Module

Move the entire cache state machine into a native module and let TypeScript
coordinate it rather than represent its internal state directly.

Strengths:

- highest long-term performance ceiling
- aligns with the principle that TypeScript stitches and orchestrates while the
  hot state machine can live lower in the stack

Risks:

- largest implementation and review surface
- too large for the first remediation branch unless the design review explicitly
  chooses it

### Option C: Local-Lifetime Intermediate Refactor

Introduce a more deliberate ownership mode for short-lived intermediates used in
the Gemma 4 decode hot path, potentially bypassing some registry bookkeeping for
strictly lexical lifetimes.

Strengths:

- directly addresses the measured wrapper and registry tax
- likely relevant if the gap proves broader than cache churn alone

Risks:

- deepest semantic change of all options
- very easy to drift into public API sprawl
- earlier broad attempts in this direction already proved too speculative

This option should remain private and proposal-only unless measurements and
reviews explicitly force it.

### Option D: Small Compiled Subgraphs

Reduce graph node count for selected composite helpers without changing the
overall cache or ownership model.

Examples:

- small composite activations
- small normalization-plus-elementwise helpers

Strengths:

- matches upstream MLX style when used sparingly
- can reduce per-token op count without changing ownership semantics
- especially relevant for repeated small composite helpers such as `geglu`,
  where upstream already uses selective compilation

Risks:

- not the best fit for the strongest current cache evidence
- compiling large forward regions would be architecturally intrusive and would
  move against upstream practice

## Proposed Recommendation

The current best recommendation is:

1. approve the measurement and proposal branch first
2. finish the Phase 1 cache/attention contract cleanup so borrowed-versus-owned
   cache returns are explicit and private
3. run **Option D: Small Compiled Subgraphs** as a bounded Phase 1.5 spike
   before any raw-pointer or native-cache work
4. only if the compile spike leaves a material remaining gap, write the narrow
   pre-implementation design note for **Option A**
5. choose **Option A: Cache-Contract Refactor** or **Option B: Private Native
   Cache Helper** from that point based on the refreshed numbers
6. treat **Option C** as a last-resort architectural branch, not the default
   next move

This ordering reflects the current evidence:

- a bounded compile spike already produced a meaningful directional Gemma 4
  decode improvement without widening product surface
- the broader per-op Bun + FFI tax is real enough that it should be attacked
  before the repo pays native-cache complexity
- the cache path is still the clearest structural mismatch with `mlx-lm`
- if compile closes most of the remaining gap, the cache program becomes
  smaller or unnecessary
- if compile does not close enough of the gap, the remaining headroom may still
  justify native work, but only behind a clearer backend seam than the earlier
  shallow native experiments
- native binding work should be justified by a concrete missing primitive, not
  used as the first response to host-side overhead
- TypeScript-only cache work is still worth trying first, but the proposal does
  not rule out raw-pointer or custom-C++ approaches if the reviewed design says
  they are the cleanest scalable answer

## Implementation Order And Decision Gates

### Phase 1: Contract And Measurement Refactor

This phase is mandatory and already underway.

Scope:

- make the cache/attention ownership boundary explicit
- keep the public `TransformerCache` surface stable
- tighten the profiling story and benchmark review docs

Exit condition:

- clean contract boundary exists
- direction-grade measurements are refreshed

### Phase 1.5: Compiled Subgraph Spike

This phase comes before raw-pointer or native-cache work.

Start with the same kind of bounded helpers upstream already compiles:

- `geglu`
- `logit_softcap`

Then only widen if the numbers justify it:

- Gemma 4 MLP-local compiled motifs
- block-local compiled motifs

The purpose is to answer one question cheaply:

**how much of the remaining decode gap is really Bun/FFI boundary cost that can
be collapsed without changing cache ownership or adding native cache code?**

Exit conditions:

- if Gemma 4 decode is now within about `5%` of the live `mlx-lm` bar, stop
  the cache-native program and treat the remaining gap as the current ceiling
- if the gap remains material and the cache still looks like the strongest
  structural mismatch, continue into Option A or Option B

### Phase 2: Cache-Structural Branch

Only after Phase 1.5 fails to close enough of the gap do we choose between:

- Option A: cache-contract work, potentially including raw-pointer internals
- Option B: deeper native backend research behind a cleaner seam than the
  earlier shallow cache-only experiments

At this point the compile branch has already exhausted the strongest
non-native/non-cache lever.

The later runtime research changed one practical detail: shallow native cache
seams were explored and proved too unstable to become the default next
mainline step. That means any resumed native work should now be treated as a
deeper backend-seam research branch, not an automatic follow-up to compile.

## Required Design Note Before Implementation

Before any implementation branch starts, write one short design note for the
chosen Option A variant that answers these questions explicitly:

1. Which private boundary is changing?
   - cache internals only
   - cache/attention boundary
   - or a deeper core/native boundary
2. What is the steady-state `updateAndFetch()` return contract?
   - caller owns returned arrays
   - cache owns them and caller borrows them
   - or another precise private contract
3. Given current `sliceUpdate()` semantics, what is the irreducible wrapper
   floor per cache-owning layer per token if no native helper is added?
4. If that floor is still too high, which exact `mlx-c` primitives are being
   exhausted first, and why is a custom helper then justified?
5. How do full-cache, saturated sliding-cache, and mixed-pattern-cache paths
   each change under the proposal?

## Review Checklist For Fresh-Eyes Agents

Any agent reviewing this proposal should answer these questions explicitly.

### Reference correctness

- Does the proposal accurately describe the current `mlx-lm` cache behavior?
- Does it overstate what `mlx-c` already exposes?
- Does it use `mlx-swift` and `mlx-swift-lm` as design references rather than
  as false benchmark references?

### Architecture and package boundaries

- Does the proposal preserve `@mlxts/core` and `@mlxts/transformers` boundaries?
- Does it avoid widening public APIs prematurely?
- Does it keep tensor lifetimes visible and auditable?

### Performance and evidence quality

- Are the benchmark and profiling windows actually measuring steady-state decode?
- Are all benchmark runs sequential and reproducible?
- Does the proposal distinguish between:
  - per-op host-side tax
  - cache-local structural churn
  - missing native primitives
- Does it separate branch-selection evidence from precise gap-allocation claims?

### Refactor quality

- Is the proposed cache refactor internally coherent across:
  - full caches
  - rotating caches
  - mixed per-layer caches
- Does it have an explicit fallback if TypeScript-only changes are insufficient?

## What Should Not Happen Next

- No broad public `untracked` or `unchecked` API work.
- No large-scale runtime refactor without a reviewed proposal.
- No blind `mlx-c` binding sweep.
- No assumption that one small tweak is "the fix" without a full design pass.
- No parallel benchmark runs.
- No pretending that the current profile buckets are exact per-token attribution.

## Exit Criteria Before Implementation

Implementation should not begin until:

1. this proposal has been reviewed by at least two independent agents
2. the review explicitly scores the preferred branch against the three decision
   factors
3. the Option A design note above exists and is reviewed
4. the chosen branch has a clear fallback path
5. the measurement method is accepted as direction-grade and the completion
   criteria require a fresh live `mlx-lm` reference before final sign-off

## Expected Deliverables For The Implementation Phase

Once a refactor branch is approved, that branch should produce:

- a narrow design note for the chosen implementation path
- before/after sequential Gemma 4 parity numbers
- a runtime review artifact under `docs/reviews/`
- follow-up documentation on:
  - avoiding these patterns in future hot paths
  - how to measure similar issues cleanly
  - when to choose TS refactor versus compile versus native binding
