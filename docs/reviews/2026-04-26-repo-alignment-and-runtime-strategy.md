# Repo Alignment Review: Product Vision and Runtime Strategy

## Summary

This review realigns the repo docs with the current product direction: `mlxts`
is a TypeScript-native ML stack for serious local ML on Apple Silicon, not an
inference demo and not a Python wrapper. The package seams should make training,
fine-tuning, serving, multimodal work, benchmarks, and agentic usage feel
coherent to web developers and ML practitioners.

The repo already encodes most of this doctrine in `AGENTS.md`, `MEMORY.md`,
`PLAN.md`, `docs/design-reasoning.md`, and the Phase 9 serving docs. The main
gap was that public orientation docs lagged the actual package state and did
not yet name the serving/runtime strategy seam clearly enough for future
techniques such as paged KV, TurboQuant-style KV compression, FlashAttention-
style backends, speculative decoding, MTP, and scheduler variants.

## Files Reviewed

- `AGENTS.md`
- `MEMORY.md`
- `PLAN.md`
- `README.md`
- `docs/architecture.md`
- `docs/design-reasoning.md`
- `docs/inference-optimizations.md`
- `docs/product-surfaces.md`
- `docs/runtime-optimization-matrix.md`
- `docs/runtime-safety.md`
- `packages/serve/README.md`
- `packages/agent/README.md`
- package and example layout under `packages/` and `examples/`

## Changes Made

- Added `docs/serving-runtime-strategy.md` as the shared strategy map for
  package boundaries, runtime strategy axes, future backend flags, native-code
  policy, evidence requirements, and the current seven-step execution order.
- Updated `README.md` so the public repo shape includes `@mlxts/serve`,
  `@mlxts/agent`, `@mlxts/transformers`, `@mlxts/lora`, `@mlxts/align`, and
  `@mlxts/quantize`, and so examples are described as workbook/proof surfaces
  rather than product substitutes.
- Updated `docs/product-surfaces.md` with first-class Serving, Agent, and
  Examples/Workbooks surfaces.
- Updated `docs/architecture.md`, `PLAN.md`, and `AGENTS.md` to point at the
  new serving/runtime strategy document.

## Alignment Findings

The package split is directionally correct. `@mlxts/core` owns MLX runtime
truth, `@mlxts/transformers` owns autoregressive model architecture and
generation, `@mlxts/serve` owns endpoint/protocol/scheduler concerns,
`@mlxts/agent` owns tool loops, and examples stay thin over package APIs.

The runtime doctrine is already strong. Existing docs say runtime strategy is
not model identity, performance is observable, cache updates must be O(1), and
advanced backends should live below semantic model code. The new strategy doc
pulls that into one place for serving and operator flags.

The serving docs are honest about the current state. They distinguish admission
micro-batching, static greedy full-cache batching, and future continuous
token-level batching. That distinction must remain non-negotiable when adding
Qwen and Gemma scheduler support.

The examples posture is mostly right but needed sharper product language.
Examples are not second-class; they are workbooks, proofs, and agent-readable
flows. But reusable behavior still belongs in packages.

## Remaining Gaps

Qwen and Gemma serving are not done just because the endpoint works. Qwen still
needs hybrid-cache-aware batching and long-prefill ergonomics. Gemma still
needs layer-pattern/sliding/global cache batching. Both need benchmark evidence
before broad serving claims.

The benchmark story is strong but should get more scheduler evidence before
fairness or throughput claims: queue time, admitted/running rows, active batch
size, waiting rows, cache-hit type, and cancellation phase.

Responses API support is useful but deliberately text-only. Fuller Responses
tool/function behavior and Anthropic Messages should land through the shared
protocol-neutral request model, not as copied protocol stacks.

The examples portfolio is useful but not yet a polished workbook system. Future
work should make examples consistently agent-readable: clear intent, command
flow, evidence, expected runtime, and where reusable behavior lives.

The repo currently has stale completed sub-agent threads occupying the agent
limit. Future complex work should clear stale threads before trying to launch
fresh second-opinion agents.

## Current Seven-Step Execution Order

1. Keep repo alignment current as product surfaces evolve.
2. Use `docs/serving-runtime-strategy.md` as the strategy map for runtime flags,
   cache backends, scheduler choices, and native helper decisions.
3. Profile Qwen memory and long-prefill behavior against `mlx-lm` with paired
   evidence.
4. Implement Qwen hybrid-cache batching only through honest cache/scheduler
   seams.
5. Implement Gemma sliding/global cache batching on the same scheduler model.
6. Expand protocol support through the shared request model: fuller Responses,
   then Anthropic Messages.
7. Resume MoE and multimodal breadth once cache/scheduler strategy is stable.

## Validation

This was a documentation and architecture-alignment slice. No runtime-sensitive
production code changed, no live model serving was run, and no heavy MLX
benchmark was needed.

Validation run for this slice:

- `git diff --check`
- `bun run typecheck`
- `bun run check:runtime-review`

`biome check` ignores the Markdown paths in this repo configuration, so it did
not provide a formatter signal for this documentation-only slice.
