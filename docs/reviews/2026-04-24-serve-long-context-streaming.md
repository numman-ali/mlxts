# Runtime Review: Serve long-context streaming

## Summary

Qwen 3.6 `65536/128` endpoint runs showed that long-context serving is not just
a model-memory problem. Buffered JSON requests repeatedly timed out after several
minutes because the client received no bytes while prefill ran. Streaming also
timed out until the serving bridge and streaming prefill path were made
cooperative.

The fix has two parts:

- `/v1/completions` SSE now emits an initial comment frame and the SSE bridge
  emits periodic comment heartbeats while awaiting the next model event.
- Transformer streaming prefill now yields to the event loop between prefill
  chunks, allowing heartbeat frames and cancellation plumbing to run during long
  prefill instead of only after first-token decode.

## Files Reviewed

- `packages/serve/src/server-sse-heartbeat.ts`
- `packages/serve/src/server-streaming.ts`
- `packages/serve/src/server-responses-streaming.ts`
- `packages/transformers/src/infrastructure/generation/helpers.ts`
- `packages/transformers/src/infrastructure/generation/runtime-streaming.ts`

## Tensor Lifetime Audit

The new cooperative prefill helper mirrors the existing chunked prefill ownership
shape: chunk token arrays, logits, sliced prompt embeddings, and sliced position
IDs are scoped to each chunk and freed before yielding back to the scheduler.
Returned tail embeddings/position IDs remain owned by the caller, matching the
existing `PrefilledPrompt` contract. The serving heartbeat helper only enqueues
SSE comment frames and owns no tensors.

## Memory / Performance Evidence

- Focused regression suite:
  `bun test packages/transformers/src/generation.test.ts packages/transformers/src/infrastructure/generation/helpers.test.ts packages/serve/src/server.test.ts packages/serve/src/server-streaming.test.ts packages/serve/scripts/benchmark-serve-options.test.ts packages/serve/scripts/benchmark-serve-completions.test.ts`
  passes with `62` tests.
- Hot-path guardrail:
  `bun run bench:generation:parity --model mlx-community/Qwen3.6-27B-4bit --prompt-tokens 1024 --generation-tokens 128 --trials 1 --skip-mlx-lm-reference`
  completed with `generation_tps=29.216`, `prompt_tps=251.023`,
  `peak_memory=19.934 GB`, `active_delta=0.018 GB`,
  `active_slope_mb_per_token=0.14`, and `evals_per_token=1.00`.
- Qwen 3.6 long-context streaming:
  `bun run bench:serve --model mlx-community/Qwen3.6-27B-4bit --model-id qwen-local --prompt-tokens 65536 --generation-tokens 128 --concurrency 1 --trials 1 --no-warmup --stream --greedy --ignore-eos --max-concurrent-requests 1 --max-batch-size 8 --batch-window-ms 2 --max-prompt-tokens 65536 --max-total-tokens 65664 --gpu-memory-utilization 0.85 --request-timeout-ms 3600000`
  completed with `wall_ms=351716.2`, `mean_ttft_ms=346501.8`,
  `mean_prompt_to_first_token_tps=189.136`,
  `mean_post_ttft_completion_tps=24.356`, `peak_memory=31.406 GB`,
  `cache_memory=4.467 GB`, `active_delta=0.013 GB`, and `finish_reasons=length`.

## Independent Review

Pascal's independent serving/scheduler read-only review identified the same
boundary this fix preserves: HTTP/SSE should keep long requests observable, while
real concurrency belongs in a scheduler-owned generation engine. The fix here
keeps bytes flowing during long streaming prefill without claiming continuous
batching or changing cache semantics.

## Remaining Risks / Follow-ups

- This keeps long streaming requests alive and observable; it does not make
  huge-context buffered JSON a good user experience. Use streaming for very long
  prompts.
- Cooperative prefill yields between chunks, not inside a single MLX chunk. Very
  large `prefillStepSize` values can still create long silent intervals.
- This is not continuous batching. Scheduler-owned row admission remains the
  next serving-throughput tranche.
