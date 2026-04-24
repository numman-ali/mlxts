# Runtime Review: Serve stream flush cadence

## Summary

This slice fixes a real endpoint streaming gap found during the Qwen 3.6 serving
ladder. `/v1/completions --stream` produced many SSE chunks internally, but the
client observed the first non-empty chunk only when generation was effectively
finished. The model path was not the culprit: a synthetic Bun server reproduced
the same behavior when the async generator yielded through microtasks without a
macrotask break.

The serving stream writers now yield one macrotask after each processed
generation stream event so Bun's HTTP layer can flush already-enqueued SSE bytes
while model generation continues. `ReadableStream.start()` also launches the
writer in the background instead of returning the long-running writer promise.

## Files Reviewed

- `packages/serve/src/server.ts`
- `packages/serve/src/server-streaming.ts`
- `packages/serve/src/server-responses.ts`
- `packages/serve/src/server-responses-streaming.ts`

## Tensor Lifetime Audit

This change stays in the HTTP/SSE bridge above the tensor-owning generation
runtime. It does not create, retain, dispose, or evaluate any `MxArray` handles.
The only new runtime behavior is an event-loop yield after serving has processed
an already-yielded `GenerationStreamEvent`.

The model generator still owns decode and cache lifetimes. Cancellation still
flows through the existing abort signal and async iterator return path.

## Memory / Performance Evidence

Pre-fix Qwen endpoint streaming evidence:

- `bun run bench:serve --model mlx-community/Qwen3.6-27B-4bit --model-id qwen-local --prompt-tokens 1024 --generation-tokens 512,1024 --concurrency 1 --trials 3 --stream --greedy --ignore-eos --max-concurrent-requests 1 --max-batch-size 8 --batch-window-ms 2 --max-prompt-tokens 1024 --max-total-tokens 2048 --gpu-memory-utilization 0.85`
- `1024/512`: `stream_chunks=64`, but `mean_ttft_ms=21859.9` against `wall_ms=21860.1`
- `1024/1024`: `stream_chunks=128`, but `mean_ttft_ms=39652.2` against `wall_ms=39652.4`

Focused validation:

- `bun test packages/serve/src/server.test.ts packages/serve/src/server-streaming.test.ts`: 31 pass, 0 fail
- `bun run --filter '@mlxts/serve' typecheck`: pass
- Synthetic Bun server reproduction now receives the first streamed completion
  bytes in about `48 ms` while the microtask-heavy generator is still active,
  instead of after the whole stream drains.
- Post-fix live Qwen check:
  `bun run bench:serve --model mlx-community/Qwen3.6-27B-4bit --model-id qwen-local --prompt-tokens 1024 --generation-tokens 512 --concurrency 1 --trials 1 --no-warmup --stream --greedy --ignore-eos --max-concurrent-requests 1 --max-batch-size 8 --batch-window-ms 2 --max-prompt-tokens 1024 --max-total-tokens 1536 --gpu-memory-utilization 0.85`
  - `wall_ms=21957.6`
  - `mean_ttft_ms=4435.0`
  - `mean_prompt_to_first_token_tps=230.889`
  - `mean_post_ttft_completion_tps=29.163`
  - `stream_chunks=64`, `completion_tokens=512`, `finish_reasons=length`

## Independent Review

The earlier serving-reference audits still apply: reference stacks flush SSE
through a serving loop/collector boundary, while model execution continues in an
engine-owned loop. This patch keeps that responsibility boundary: it does not
move flushing into the model runtime, and it does not claim continuous batching.

## Remaining Risks / Follow-ups

- The macrotask yield is intentionally in the serving stream bridge. It may add a
  small amount of HTTP streaming overhead per emitted text batch, but the
  transformer stream batches decode output by default, so this is not a
  per-token sleep on normal Qwen/Gemma streaming.
- Rerun the full Qwen streaming ladder with multiple trials before claiming
  broad streaming quality. The one-rung post-fix check proves the failure mode is
  fixed, not that every long-output/concurrency case is done.
- This fixes progressive byte delivery, not scheduler-level backpressure or true
  continuous batching.
