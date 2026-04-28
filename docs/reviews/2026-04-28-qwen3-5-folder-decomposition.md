# Qwen 3.5 Folder Decomposition Review

## Summary

Moved Qwen 3.5 / 3.6 family files into role-named subfolders without changing
model semantics: multimodal wrapper and vision helpers under `multimodal/`,
hybrid cache state under `cache/`, and gated-delta / rotary helpers under
`linear-attention/`. Text-core model, attention, block, MLP, config, weights,
types, and load files remain at the family root.

## Files Reviewed

- `packages/transformers/src/families/qwen3_5/attention.ts`
- `packages/transformers/src/families/qwen3_5/block.ts`
- `packages/transformers/src/families/qwen3_5/cache/batch-cache.ts`
- `packages/transformers/src/families/qwen3_5/cache/index.ts`
- `packages/transformers/src/families/qwen3_5/config.ts`
- `packages/transformers/src/families/qwen3_5/linear-attention/gated-delta-recurrence.ts`
- `packages/transformers/src/families/qwen3_5/linear-attention/gated-delta.ts`
- `packages/transformers/src/families/qwen3_5/linear-attention/rotary.ts`
- `packages/transformers/src/families/qwen3_5/load.ts`
- `packages/transformers/src/families/qwen3_5/model.ts`
- `packages/transformers/src/families/qwen3_5/multimodal/conditional-support.ts`
- `packages/transformers/src/families/qwen3_5/multimodal/conditional.ts`
- `packages/transformers/src/families/qwen3_5/multimodal/preprocessing.ts`
- `packages/transformers/src/families/qwen3_5/multimodal/vision-support.ts`
- `packages/transformers/src/families/qwen3_5/multimodal/vision.ts`
- `packages/transformers/src/index.ts`

## Tensor Lifetime Audit

No tensor-producing expressions, cache ownership semantics, recurrent-state
retention, disposal paths, or model forward logic were changed intentionally.
The changed production lines are relative import/export path updates caused by
the file moves.

## Memory / Performance Evidence

This tranche makes no performance claim. `bench:generation` was not run because
the change is a pure file move. `bench:generation:parity` was exercised through
the required real Qwen/Gemma regression decode smoke:

- Qwen `bench:generation:parity` smoke: prompt `232.319` tok/s, generation
  `26.811` tok/s, evals/token `1.00`, peak memory `17.184` GB.
- Gemma `bench:generation:parity` smoke: prompt `7532.730` tok/s, generation
  `75.619` tok/s, evals/token `1.00`, peak memory `9.893` GB.

The required `bun run regression:qwen-gemma -- --profile real` command passed
after local E2E load was removed from the machine:

- Qwen mixed long/short fairness rung:
  `mean_post_ttft_completion_tps=13.301`, `completion_tps=0.929`,
  short request `serverStreamTtftMs=5270.0`, passing the long-context
  fairness budgets without lowering thresholds.
- Gemma mixed long/short fairness rung:
  `mean_post_ttft_completion_tps=69.299`, `completion_tps=59.370`, passing.

The stale message-protocol route expectation was repaired in
`packages/serve/scripts/regression-serve-matrix.ts` after independent review:
chat, Responses, and Anthropic protocol-health rungs now expect
`single:prompt_prefix_cache` with zero continuous scheduler counters. This
preserves the intended prompt-prefix-cache capability instead of forcing an
artificial continuous route. Focused coverage was added in
`packages/serve/scripts/regression-serve-matrix.test.ts`, and
`packages/serve/README.md` now matches the runtime boundary.

The repaired protocol-health rungs passed inside the full real profile:

- Qwen chat: `routes=single:prompt_prefix_cache=1`,
  `mean_post_ttft_completion_tps=34.026`.
- Qwen Responses: `routes=single:prompt_prefix_cache=1`,
  `mean_post_ttft_completion_tps=34.382`.
- Qwen Anthropic: `routes=single:prompt_prefix_cache=1`,
  `mean_post_ttft_completion_tps=33.074`.
- Gemma chat/Responses/Anthropic also reported `single:prompt_prefix_cache=1`
  and passed protocol-health budgets.

Focused validation passed:

- `bun test packages/transformers/src/families/qwen3_5 packages/transformers/src/load.test.ts packages/transformers/src/index.test.ts` passed: 96 tests.
- `bun test packages/serve/scripts/regression-serve-matrix.test.ts` passed: 11 tests.
- `bun run regression:qwen-gemma -- --profile real`
- `bun run validate`

## Independent Review

Halley independently reviewed the route-gate blocker. The review confirmed that
message-shaped chat, Responses, and Anthropic requests intentionally route
through `single:prompt_prefix_cache` in `packages/serve/src/engine/index.ts`,
and that forcing continuous routing would measure an artificial non-production
path. Halley recommended the exact protocol-health budget repair implemented in
this diff and noted the matching README wording drift.

## Remaining Risks / Follow-ups

No threshold was lowered and no runtime path was changed to route around the
serving budgets. The long-context fairness rungs are sensitive to unrelated
local GPU/CPU load, so future reruns should keep other E2E workloads off the
machine while the real profile is active.
