# Proposal: Readable Runtime Surfaces With Optimized Backends Underneath

## Summary

The recent runtime work improved performance, but it also pushed too much
execution strategy directly into the model-family teaching surface of the repo.

That trade is no longer acceptable.

The new direction is:

- preserve the runtime lessons
- clean the repo back toward readability
- separate readable reference surfaces from optimized backend implementations

This proposal is not a retreat from performance work. It is a structural reset
so performance work can continue without obscuring the code that is supposed to
teach how inference and training actually work.

For the current repo phase, deeper Gemma 4 performance work is parked. The
active mainline priority is repo-wide readability, composability, and surface
alignment. Runtime research remains background context until the semantic
surfaces across the repo are clean again.

It builds directly on
[`docs/reviews/2026-04-08-runtime-runtime-research-retrospective.md`](../reviews/2026-04-08-runtime-runtime-research-retrospective.md).

## The Core Rule

Model code should explain the model.

Runtime backends should hide execution tricks.

That means the repo should expose two clearly different layers:

### 1. Readable reference surfaces

These are the files a human should read first to understand the system.

They should show control flow in semantic terms:

- inference:
  - prepare inputs
  - prefill cache
  - run decode step
  - sample token
  - finish
- training:
  - get batch
  - compute loss
  - compute gradients
  - normalize gradients
  - apply update
  - materialize
  - report

These surfaces should prefer clear function names and explicit step boundaries
over aggressive fusion.

### 2. Optimized backend implementations

These are the layers that decide how a semantic step is executed:

- plain eager TypeScript/MLX composition
- compiled transform reuse
- private native helper
- later, deeper native execution backend if warranted

The readable surface should call semantic helpers. Those helpers may delegate
to an optimized backend, but the high-level flow must stay legible.

That structure should also show up in the filesystem. When one concern grows
past a few files, the repo should prefer a small role-based subfolder such as
`runtime/`, `cache/`, `generation/`, or `sampling/` instead of leaving a large
flat directory that mixes semantic entrypoints and helper detail.

## The Design Goal

The project was meant to help a human understand ML top to bottom. That means:

- inference should be understandable by opening one small cluster of files
- training should be understandable by opening one small cluster of files
- performance strategy should be modular and swappable
- the same semantic step should be able to run through a TypeScript or native
  implementation without rewriting the control flow

The right architecture is therefore not "everything in TypeScript" and not
"everything in C++." It is:

- semantic orchestration in TypeScript
- execution strategy behind interfaces or helper seams
- multiple implementations where the same semantic step needs different
  runtime strategies

## What The Mainline Should Read Like

### Inference

Inference should remain readable through files like:

- `packages/transformers/src/generation.ts`
- family `model.ts`
- family `block.ts`
- family `attention.ts`

But these files should read primarily in terms of:

- decode step structure
- attention structure
- cache policy choice
- MLP structure

They should not become a pile of compile variants, fusion islands, and ad hoc
backend strategy switches.

This cleanup is now underway:

- Gemma 4 runtime strategy has been moved beneath `attention.ts`, `block.ts`,
  `mlp.ts`, and `model.ts`
- `packages/transformers/src/generation.ts` has been reduced back toward the
  public inference surface, with decode-loop runtime mechanics living
  underneath it
- shared sampling infrastructure now follows the same split, with
  `sampling/index.ts` focused on semantic sampling flow and runtime strategy
  moved beneath it
- `@mlxts/nn` math surfaces now follow the same rule, with
  `activations/index.ts` and `losses/index.ts` reading semantically while
  compile-backed transform reuse lives in lower runtime helpers

### Training

Training already shows more of the right shape through:

- `packages/train/src/loop.ts`
- `packages/train/src/step.ts`
- `packages/nanogpt/src/train.ts`

This direction should be strengthened, not replaced. Training should continue
to read as explicit staged flow rather than a hidden trainer framework.

### Composition Style

The code should compose through small step functions that feel pipeline-like
without becoming a framework:

- explicit state in, explicit state out
- semantic names
- small helpers that can be drilled into
- stable boundaries between orchestration and execution

This gives the readability benefits of pipeline or reactive code without
turning the repo into RxJS, Effect, or a callback maze.

## What Stays

The following work should remain part of the mainline direction:

- the runtime lock for heavy MLX commands
- the private cache-view ownership seam
- compile as an early optimization lever for repeated pure tensor motifs
- semantic helper names for math and model behavior
- the runtime optimization matrix as a planning/control document

These improve the architecture without forcing the model-family code to become
runtime-strategy prose.

## What Should Be Quarantined Or Reverted

The following kinds of work should not keep spreading through mainline:

- family-local compile/fusion strategies that make model files hard to read
- unstable native seam research in default runtime paths
- large attention files whose main job becomes execution strategy branching
- product-facing code that teaches backend internals instead of model semantics

Some of this work may still be useful. The point is that it should live behind
backend seams or in research-oriented code until it has earned a clean stable
home.

## Proposed Runtime Structure

### Semantic layer

This layer expresses the model or loop in readable terms.

Examples:

- `runDecodeStep`
- `runAttentionStep`
- `runMlpStep`
- `updateCacheState`
- `computeLoss`
- `prepareGradients`
- `applyOptimizerStep`

### Strategy layer

This layer decides how a semantic step runs.

Examples:

- eager helper
- compiled helper
- native helper
- later, backend-selected implementation

This layer is allowed to care about shape reuse, transform lifetime, or native
dispatch, but it should sit below the readable surface.

### Backend layer

This layer contains implementation-specific machinery:

- compile plumbing
- transform reuse maps
- native FFI wrappers
- later, deeper native execution paths

This layer is where performance tricks belong.

## What To Do Next

### Phase 1: preserve the lessons

- keep the retrospective
- keep the benchmark evidence that led us here
- keep the runtime lock and cache-view seam

### Phase 2: restore the readable inference surface

- identify the family files where execution strategy drift is worst
- move compile/native strategy decisions out of those files where possible
- keep semantic top-level helpers and control flow legible again

Gemma 4 should be the first cleanup target because that is where the drift is
currently the highest.

This phase now includes the generation boundary too:

- `generation.ts` should remain the readable public inference surface
- lower-level decode runtime mechanics should live beneath it

### Phase 3: codify the semantic/backend split

- introduce clearer internal helper seams so one semantic step can have
  multiple runtime implementations
- keep the default readable path obvious
- keep backend selection private

### Phase 4: resume performance work from the cleaner structure

Only after the readable surface is restored should we continue deeper runtime
work such as:

- broader compiled backend choices
- cleaner native helper seams
- later, deeper native execution boundaries if still justified

## Immediate Cleanup Sequence

The next implementation branch after this proposal should follow this order:

1. keep the retrospective and benchmark evidence in place so the recent work is
   not lost
2. identify which current runtime changes are foundational and which are
   readability-breaking experiments
3. keep foundational work in mainline:
   - runtime lock
   - cache-view ownership seam
   - simple semantic compile helpers
4. quarantine or revert work that makes the main family files read primarily in
   terms of execution strategy instead of model semantics
5. reshape the inference path so a reader can follow generation through a small
   set of semantic functions again
6. only after that, reintroduce backend strategy behind narrower helper seams

Gemma 4 should lead this cleanup because it accumulated the most runtime
strategy drift and therefore gives us the clearest proving ground for the new
structure.

## What Counts As "Readable Again"

Inference is readable again when:

- `generation.ts` reads like generation control flow
- family `model.ts`, `block.ts`, and `attention.ts` read like model semantics
- backend choices are visible as helper selection, not spread throughout the
  whole file
- a new reader can follow a decode step without first learning the performance
  experiment history of the repo

Training is readable again when:

- `@mlxts/train` and example training surfaces read like explicit staged loops
- optimizer and gradient logic remain drillable through small semantic helpers
- training-specific compile work stays behind helper seams rather than becoming
  the dominant vocabulary of the loop

## Inference And Training Are Different, But Aligned

Training and inference should share the same architectural rule but not the
same low-level machinery.

Training shares:

- readable explicit flow
- semantic helper naming
- compile-first for repeated pure motifs
- backend separation

Training does not need to inherit inference-specific cache machinery.

Inference is where mutable cache state and attention/runtime boundary work live.

## Success Criteria

This restructure is successful when:

1. a human can understand inference by reading a small number of semantic files
2. a human can understand training by reading a small number of semantic files
3. backend strategy can change without rewriting those semantic files
4. performance research continues behind clearer seams
5. the repo becomes more teachable without giving up the ability to beat
   `mlx-lm`

## Relationship To Existing Documents

This proposal refines the earlier runtime execution architecture document:

- it keeps the checkpoint truth versus runtime strategy split
- it keeps compile-first as an optimization rule
- it keeps native work in-bounds
- but it adds a stronger readability requirement

In short:

**optimization is still required, but readability is now a non-negotiable
architectural constraint.**
