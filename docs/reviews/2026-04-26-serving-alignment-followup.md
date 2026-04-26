# Docs Review: Serving Alignment Follow-Up

## Summary

After the sampled continuous-batching tranche, the implementation was ahead of
several planning docs. This follow-up realigns roadmap language with current
serving truth: Qwen 3.6 text and Gemma 3/4 layer-pattern requests now have
cache-generic continuous scheduling for buffered, streaming, greedy, and
model-native sampled paths, while prefix cache, paged KV, Anthropic, embeddings,
production metrics, and advanced backend flags remain future work.

## Files Reviewed

- `PLAN.md`
- `docs/ecosystem-structure.md`
- `docs/gates-and-milestones.md`
- `docs/inference-optimizations.md`
- `docs/runtime-optimization-matrix.md`
- `docs/serving-runtime-strategy.md`
- `packages/serve/README.md`
- `.reference/mlx-lm`
- `.reference/vllm-mlx`
- `.reference/omlx`
- `.reference/rapid-mlx`
- `.reference/text-generation-inference`

## Independent Review

Descartes reviewed serving against the reference repos and identified the next
gaps as scheduler depth, cache architecture, metrics, protocol parity, and
long-context/multi-request proof.

Halley reviewed roadmap/package alignment and found stale serving claims in
`PLAN.md`, overclaims in `docs/ecosystem-structure.md`, fragmented gate
guidance, missing serving rows in the optimization matrix, and the need for a
typed strategy seam before advanced backend flags.

## Changes Made

- Updated Phase 9 status and serving deliverables to reflect current Qwen/Gemma
  continuous scheduling and sampled defaults.
- Marked paged cache, prefix cache, Anthropic, embeddings, dynamic load/unload,
  and production metrics as future work instead of current package capability.
- Added a change-specific validation gate table so future agents can choose
  focused proofs without skipping required broader gates.
- Added serving strategy, metrics, prefix/paged cache, TurboQuant, and protocol
  parity rows to the runtime optimization matrix.
- Updated the serving/runtime execution order so typed strategy configuration,
  scheduler hardening, cache backends, metrics, and protocol expansion happen in
  the right sequence.

## Validation

This is documentation-only. No runtime-sensitive production files changed and no
heavy MLX benchmark was needed.

Validation for this slice should include:

- `git diff --check`
- `bun run typecheck`
- `bun run check:runtime-review`
