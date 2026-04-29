# Qwen Conditional Continuous Routing

## Summary

This tranche makes top-level Qwen 3.5 / 3.6 conditional checkpoints eligible
for text-only continuous serving by exposing the same family-owned hybrid batch
cache that Qwen text checkpoints already expose.

Serve still rejects media inputs from continuous routing. Image prompts keep
their prepared-embedding and per-batch RoPE-delta path; the batch-cache path is
only for text token batches through the existing Qwen text decoder.

## Files Reviewed

- `packages/transformers/src/families/qwen3_5/multimodal/conditional.ts`
- `packages/transformers/src/families/qwen3_5/multimodal/conditional.test.ts`
- `packages/serve/src/engine/engine.test.ts`

## Runtime Sensitivity Notes

`conditional.ts` is runtime-sensitive because it is the forward path for
top-level Qwen conditional checkpoints loaded by real Qwen 3.6 snapshots. The
change adds a model-owned `Qwen3_5TextBatchCache` factory and allows
`Qwen3_5ForConditionalGeneration.forward()` to receive either the single Qwen
text cache or the Qwen text batch cache.

The conditional wrapper continues to own image prompt preparation and RoPE
deltas for single-cache multimodal decode. Batch-cache forwards without
explicit `positionIds` leave position handling to the Qwen text attention path,
which already uses `Qwen3_5TextBatchCache.offsetTensor()` for batched RoPE.

## Tensor Lifetime Audit

The new batch-cache factory mirrors the text-only Qwen factory. If seed-cache
restore fails, the newly created batch cache is disposed before rethrowing.

`forward()` does not retain extra tensors on the batch-cache path. Explicit
`positionIds` remain caller-owned through the existing retain helper; generated
shifted position ids are still freed in the existing `finally` block.

## Memory / Performance Evidence

- Live pre-fix cmux smoke with `mlx-community/Qwen3.6-27B-4bit` loaded and
  served successfully, but route telemetry reported `route=single
  reason=unsupported_model_type model_type=qwen3_5`.
- Live post-fix cmux smoke with the same checkpoint reported `route=continuous
  eligible=yes reason=eligible model_type=qwen3_5`, then emitted scheduler
  queued/admitted/first-token/finished phases. A repeated identical chat request
  reported `prompt_tokens_details.cached_tokens=16`, and server telemetry logged
  `cache hit read_tokens=16 write_tokens=0`.
- `bun test packages/transformers/src/families/qwen3_5/multimodal/conditional.test.ts packages/serve/src/engine/engine.test.ts`:
  passed, `60` tests.
- `bun test packages/transformers/src/families/qwen3_5/multimodal/conditional.test.ts packages/serve/src/engine/engine.test.ts packages/serve/src/engine/routing.test.ts`:
  passed after the explicit media-routing guard, `67` tests.
- `bun run validate`: passed.
- `bun run regression:qwen-gemma -- --profile real`: passed. Qwen decode smoke
  reported `generation_tps=29.024`; Qwen serve protocol and continuous rungs
  routed as `continuous:eligible`, including `max_continuous_batch=8` on the
  `128x16@8` rung and prompt-cache hits on chat/OpenResponses/Anthropic message
  rungs. The Qwen mixed `32768x128+128x32` fairness rung passed with
  `max_stream_chunk_gap_ms=648.1`. Gemma decode and serve rungs also passed.
- `bench:generation`: not rerun before the post-fix serving smoke; the change
  does not alter Qwen text model math, attention kernels, sampling, or
  single-request generation.
- `bench:generation:parity`: not rerun for the same reason. Paired Qwen text
  parity remains governed by the existing Qwen benchmark ladder; this tranche
  restores the conditional wrapper's text-only continuous-routing eligibility.

## Independent Review

Zeno reviewed the route miss and agreed with exposing the Qwen text batch-cache
surface on `Qwen3_5ForConditionalGeneration` instead of adding a serve-side
model-type allowlist. The review specifically requested a guard that top-level
Qwen media/content requests still route as `media_input` and do not enter
continuous scheduling; that guard is now covered in `engine.test.ts`.

## Out-of-scope Drift Noticed

None. The pre-fix repeated tiny direct-chat miss did not reproduce after the
conditional wrapper reached the continuous path; the warmed post-fix request
hit the prompt cache and reported cached tokens.

## Remaining Risks / Follow-ups

The broader continuous memory-reservation tranche remains separate.
