# Proposal: Runtime Execution Architecture After The First Gemma 4 Win

## Status

This document is still useful architectural context, but its branch-ordering
advice is historical.

Use it together with
[`docs/proposals/2026-04-08-readable-runtime-restructure.md`](./2026-04-08-readable-runtime-restructure.md)
and
[`docs/reviews/2026-04-08-runtime-runtime-research-retrospective.md`](../reviews/2026-04-08-runtime-runtime-research-retrospective.md),
which record the later readability reset and the decision to park deeper native
runtime work while the repo-wide alignment pass continues.

## Purpose

This document turns the recent Gemma 4 parity work into a system-level runtime
plan for `mlxts`.

It should now be read together with
[`docs/proposals/2026-04-08-readable-runtime-restructure.md`](./2026-04-08-readable-runtime-restructure.md),
which adds a stronger requirement that the main inference and training surfaces
remain readable while runtime strategy moves beneath them, and with
[`docs/reviews/2026-04-08-runtime-runtime-research-retrospective.md`](../reviews/2026-04-08-runtime-runtime-research-retrospective.md),
which records that the shallow native cache seam later explored from this plan
was not benchmark-stable enough to remain the active mainline direction.

It answers four questions:

1. what we just proved
2. how runtime responsibilities should be split across the stack
3. how this should compose across model families, inference, and training
4. what the next implementation branch should be

This is the document I want reviewed next. It is deliberately broader than the
Gemma 4 parity proposal because the recent work is no longer "just a Gemma 4
bug." It is evidence about how `mlxts` should approach hot-path runtime design
in general.

The canonical surface-level map for this program now lives in
[`docs/runtime-optimization-matrix.md`](../runtime-optimization-matrix.md).

## What We Proved

The recent branch proved two things that matter beyond Gemma 4.

First, the cache and attention ownership boundary needed to be made explicit.
We now have a private cache-view contract inside `@mlxts/transformers` so
attention code does not reconstruct ownership rules independently by family.

Second, compile had to move earlier in the program. We already had a working
`compile()` primitive in `@mlxts/core`, and a bounded Gemma 4 compile spike was
worth real throughput:

- earlier clean Gemma 4 decode plateau:
  - `bench:generation ≈ 75.5 tok/s`
  - `bench:generation:parity ≈ 76.5 tok/s`
- current branch after cache-view cleanup plus selective GEGLU transform reuse:
  - `bench:generation ≈ 81.2 tok/s`
  - `bench:generation:parity ≈ 81.6 tok/s`
- smaller profiled parity run:
  - `≈ 85.2 tok/s`
  - lower core FFI time per token than the earlier profiled branch

That is a real directional win of about `6–7%` without any native-cache work.

The follow-up cache-contract cleanup after that win also mattered, but in a
different way. Using the private cache-view seam to return borrowed full-buffer
views on the hot sliding-cache path lowered the measured steady-state host-side
cost again, but it did not produce a second decisive Gemma headline jump on its
own. The practical read is that the seam was worth putting to work, but the
remaining Gemma gap is now likely beyond what TypeScript-only cache-view cleanup
can close by itself.

The practical conclusion is:

**compile is now a first-class runtime lever in `mlxts`, not an optional
afterthought.**

That does not mean the repo should start reading like `compiledFoo`,
`compiledBar`, or "compiled cache." The semantic surface should still read in
terms of math and model behavior. Compile is the runtime strategy underneath
those names, not the name of the thing itself.

It does not remove the need for cache work entirely. The cache counters are
still structurally hot. But this document is now historical at one important
point: the later shallow native-cache seam exploration was useful research, yet
it was not stable enough to remain the next default mainline step. Native work
remains in-bounds, but only behind a cleaner backend seam than the one that was
attempted after this document was written.

## The Runtime Model We Are Choosing

We are not choosing "TypeScript does everything" and we are not choosing "move
everything into C++."

We are choosing this split:

- TypeScript orchestrates
- MLX/`mlx-c` does the tensor math
- compiled subgraphs collapse repeated pure host-side graph-building overhead
- native helpers or native modules own hot mutable state when TypeScript
  orchestration becomes the bottleneck

The mental model is:

**checkpoint truth, runtime strategy, and backend implementation are different
things.**

### Checkpoint truth

Model config continues to mean only:

- architecture
- layer counts
- KV-sharing pattern
- sliding-window rules
- rope settings
- vocabulary and embedding shapes

This remains family truth loaded from the checkpoint.

### Runtime strategy

Runtime strategy means:

- whether a repeated pure helper should run eager or compiled
- whether cache state is managed or native-backed
- whether cache views are owned or borrowed at the internal boundary
- later, whether KV is dense, compressed, quantized, paged, or tiered

This is not model identity. It is execution choice.

### Backend implementation

Backend implementation means the concrete mechanism:

- eager composed ops in TypeScript
- compiled MLX closure
- thin native helper
- full native cache module

That selection stays private until there is a validated winner worth carrying
as product surface.

## Responsibilities By Layer

### `@mlxts/core`

`@mlxts/core` owns:

- `MxArray`
- FFI boundary correctness
- transform primitives like `compile()`
- hot-path metadata propagation
- optional private native helpers when `mlx-c` is insufficient

`@mlxts/core` does **not** own model-family cache logic. If we add native cache
helpers, they should be narrow generic primitives or private transformer-facing
helpers, not a new public cache framework in core.

### `@mlxts/nn`

`@mlxts/nn` owns:

- modules
- parameter trees
- parameter update semantics
- reusable layer-level structure

`@mlxts/nn` can use compiled transform reuse when it is truly reusable and safe
across training and inference.

The key safety rule is:

**compiled transform reuse in reusable module code must be built from pure
tensor composites whose dynamic inputs are explicit function arguments.**

If a compiled helper captures weight arrays implicitly from a module instance,
it becomes risky for training because `Module.update()` can replace those array
handles. That kind of compile should stay out of generic `@mlxts/nn` surfaces
unless invalidation rules are explicit.

### `@mlxts/transformers`

`@mlxts/transformers` owns:

- family model structure
- family-specific cache policies
- attention/cache composition
- generation pipeline
- benchmark and parity evidence
- private execution strategy selection for inference hot paths

This is where the private runtime composition belongs for now:

- private cache-view contract
- private semantic helper selection by family, with compile hidden underneath
- later, private native cache adapter selection

This keeps runtime-sensitive choices close to the model families they affect
without prematurely turning them into general product APIs.

### `@mlxts/train`

`@mlxts/train` should remain mostly unaffected by cache-native work.

Training responsibilities stay:

- loops
- checkpointing
- schedules
- batch flow

Training usually does not use decoder KV caches in the same way as incremental
generation, so native cache backends are primarily an inference concern.

What training *does* care about from this program is:

- reusable compiled helpers that are training-safe
- clearer rules about when compile is allowed in module code
- no hidden stale-weight capture bugs

So training shares the compile discipline, not the cache-native machinery.

## How This Composes Across Model Families

### Llama-like and Mistral-style families

These families use the most standard cache/attention pattern:

- attention produces fresh KV
- cache updates state
- attention consumes the active KV view

They should move onto the same private cache-view contract and later the same
thin native helper path if that path wins.

### Gemma 3

Gemma 3 uses the same broad cache/attention pattern with mixed local/global
masking and q/k normalization. It should share:

- private cache-view contract
- compiled helper rules
- later native cache adapter if validated

### Gemma 4

Gemma 4 remains the proving ground because it exercises more of the hard
runtime cases at once:

- mixed full and sliding caches
- KV-shared layers
- per-layer input gating
- repeated gated GELU motifs

That is why the first wins and the first native work should continue here.

### Phi and other future families

Future decoder families should inherit the architecture, not reinvent it:

- family code decides model math
- runtime strategy plugs in beneath it
- family code does not hand-roll cache ownership rules again

## Inference Versus Training

### Inference

Inference is where all three runtime levers matter:

- explicit cache ownership
- compile for repeated pure subgraphs
- native state handling when mutation semantics are the bottleneck

The near-term inference program should therefore be:

1. keep the private cache-view boundary
2. expand compile selectively and measure
3. restore and protect readable inference reference surfaces
4. treat deeper native cache or decode work as research until a benchmark-stable
   backend seam is validated

### Training

Training should adopt only the parts that are actually general:

- compiled helper motifs that are pure and safe across parameter updates
- architecture rules about explicit ownership and visible control flow

Training should **not** be forced through an inference-native cache story it
does not need.

## Runtime Composition Pattern

The private runtime composition I want is:

- model config
- private execution strategy
- cache storage backend
- attention compute path

Not every axis needs multiple implementations immediately, but this is the
shape we should think in.

For example:

- today:
  - dense managed cache
  - eager attention path
  - selective compiled helpers
- next:
  - clearer semantic inference surfaces
  - runtime helpers beneath them
  - same selective compiled helpers
- later:
  - compressed or quantized KV backend
  - matching attention path that knows how to consume it

This is also how a future TurboQuant-style path should enter the system. It is
not a second model config. It is a runtime/backend pairing between:

- cache representation
- attention compute path

## The Program From Here

### Phase 1: complete and keep

Keep the current foundation:

- profiling
- private cache-view contract
- compile as a first-class runtime lever

This phase is already delivering value and should remain in the branch history.

### Phase 1.5: expand compile selectively

This is the next implementation phase.

The order should be:

1. keep the current compiled GEGLU helper
2. compile the next safe repeated pure motif only if it passes two tests:
   - dynamic inputs are explicit
   - weight invalidation semantics stay honest
3. measure after each expansion

Good candidates:

- additional Gemma 4 MLP-local composites
- per-layer gate composites
- possibly small block-local motifs if they remain pure and explicit

Bad candidates right now:

- full decoder pass
- cache mutation paths
- any compiled region that captures live weight handles implicitly without a
  clear invalidation rule

### Decision Gate

After selective compile expansion:

- if Gemma 4 is within about `5%` of the live `mlx-lm` bar, stop the cache
  native program and document the remaining ceiling
- if the gap remains material, first restore readable inference surfaces and
  keep deeper native work on the research track until the backend seam is
  proven stable enough to earn mainline status

### Phase 2: readable inference boundary cleanup

The next mainline branch should keep the top-level decode story readable.

It should:

- preserve private cache-view ownership rules
- keep TypeScript in charge of orchestration and readable control flow
- move execution strategy beneath semantic model-family and generation surfaces
- quarantine unstable deeper native work as research rather than default
  product direction

### Phase 3: deeper native work only if earned

Deeper native cache or decode work is still in-bounds, but only if:

- compile and readable-surface cleanup still stall short of the target
- the proposed backend seam is benchmark-stable
- the native work lives beneath the semantic surfaces rather than consuming them

That is the right threshold for a bigger native commitment.

## What We Are Not Doing

We are not doing these things now:

- no duplicate model configs for runtime variants
- no public runtime/backend selector yet
- no broad `untracked` or `unchecked` public ownership APIs
- no "compile everything" refactor
- no forcing training through inference-specific cache machinery
- no building future TurboQuant slots before we need them

The architecture should leave room for these things later without prebuilding
their public surface today.

## What Should Change In Repo Guidance

This program changes the standing guidance in a few concrete ways.

### 1. Compile moves earlier

The repo already said composite helpers should compile before they become
native primitives. That rule now needs to be read more broadly:

- for repeated pure decode motifs, try compile before native helper work
- but keep compile selective, explicit, and measurement-driven

### 2. Runtime strategy is not model identity

The docs should say this plainly:

- model config describes checkpoint truth
- runtime strategy describes how we run it
- backend implementation is the concrete mechanism

### 3. Private runtime selection before public runtime selection

We should not widen public API until:

- at least one alternative runtime path wins
- we know which choices we actually intend to carry

### 4. Native hot-state ownership is allowed

The project should explicitly allow this principle:

**TypeScript orchestrates; hot mutable state may move lower when the evidence
justifies it.**

That is the real architecture lesson from this work so far.

## Questions I Want Reviewed Next

These are the questions I would want Claude to react to next:

1. Is the runtime split above the right one for `mlxts` as a whole?
2. Is selective compile the correct immediate next branch?
3. Is the training boundary drawn correctly, especially around compiled helpers
   and weight invalidation?
4. If selective compile stalls, is a deeper native backend seam the right next
   research branch, and how deep should that seam be before it earns mainline?
5. Are there any model families or package layers where this plan is too
   inference-specific or not composable enough?

## Recommendation

The next implementation branch should be:

**selective compile expansion on top of the current private cache-view
foundation, followed by one decision gate before any native cache work.**

That is the most truthful next move given what we now know.
