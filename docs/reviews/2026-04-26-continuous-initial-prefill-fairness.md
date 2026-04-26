# Runtime Review: Continuous Initial Prefill Fairness

## Summary

The full-KV continuous scheduler now treats long initial prompts as scheduler
work instead of an opaque admission wall. When the first admission group
contains a prompt longer than `prefillStepSize`, those rows enter the same
chunked prefilling queue used for later arrivals.

The scheduler also avoids FIFO head-of-line blocking among initial prefilling
rows: rows that are already ready to sample their first token are admitted
before long rows continue chunking, and long prefilling rows rotate after each
chunk. This lets a short initial row produce a first token while a long initial
row is still prefilling.

## Files Reviewed

- `packages/transformers/src/infrastructure/generation/continuous-batch.ts`
- `packages/transformers/src/infrastructure/generation/continuous-batch-types.ts`

## Tensor Lifetime Audit

This change does not introduce new tensor-producing primitives. It changes the
order in which existing scheduler helpers run:

- short initial rows call the existing tail-sampling path via
  `sampleNextBatchToken()`;
- long initial rows call the existing `prefillPromptChunk()` chunk path;
- active row merging still uses the existing `BatchKVCache.extend()` and
  `combineCurrentTokens()` ownership rules.

The scheduler still frees `#currentToken`, active cache state, and prefilling
row caches in the existing `finally` path. Rotating prefilling request objects
does not duplicate or retain native array handles.

## Memory / Performance Evidence

Focused checks:

- `bun test packages/transformers/src/infrastructure/generation/continuous-batch.test.ts packages/serve/src/transformers-engine.test.ts packages/serve/src/model-server.test.ts packages/serve/scripts/benchmark-serve.test.ts`
- Result: 54 tests passed, 0 failed.
- This tranche changes scheduler ordering, not model math. The generation
  benchmark surface is still `bench:generation` /
  `bench:generation:parity`; the paired Qwen parity evidence immediately
  before this tranche measured `26.996` vs `27.123 tok/s` on the 10k/128 rung
  with identical `19.219 GB` peak memory, and this scheduler change does not
  alter Qwen/Gemma model routing or decode kernels.

The new transformer scheduler test proves a long first initial prompt and short
second initial prompt no longer force the short row to wait for all long-row
prefill chunks. The short row emits its first token before the long row reports
its first `2/5` prefill chunk, and the model never receives the full long
prompt as one prefill forward.

The new serve-engine test proves the same behavior through
`createTransformersGenerationEngine()` with a 4096-token long prompt and a
1-token short prompt. The short request's continuous `first_token` scheduler
event is emitted before the long request's first prefill-progress event, and
the test model never sees a 4096-token forward.

Real endpoint smoke:

- `bun run bench:serve --model mlx-community/Llama-3.2-1B-Instruct-4bit --model-id llama-local --rungs 4096x16@2 --trials 1 --no-warmup --greedy --ignore-eos --stream --report-json .tmp/llama-continuous-initial-prefill-fairness-v2.json --max-concurrent-requests 1 --max-batch-size 2 --batch-window-ms 10 --max-prompt-tokens 4096 --max-total-tokens 4112 --gpu-memory-utilization 0.85 --request-timeout-ms 3600000`
- Result: route `continuous:eligible=2`, `continuous_admissions=2`, `continuous_scheduler_phases=10`, max continuous batch `2`, `20.510` completion tok/s end-to-end, `165.838` post-TTFT tok/s, max stream chunk gap `13.7 ms`, peak memory `1.790 GB`.
- The report records two prefill progress events per request on the 4096-token concurrent rung, proving the initial long prompts are scheduler-visible chunks rather than one silent prefill wall.

## Independent Review

Explorer sub-agent Pauli identified initial full prefill as the next bounded
serving fairness seam before widening Qwen or Gemma continuous batching.
Explorer sub-agent Rawls reviewed the first local patch and caught the remaining
FIFO head-of-line issue: initial rows were chunked, but a short second row could
still wait behind all chunks for a long first row. The final implementation
addresses that by admitting ready prefilling rows first and rotating long rows
after each chunk.

Explorer sub-agent Nash recommended Qwen hybrid static batching as the next
major serving tranche after this bounded scheduler fix. That work should remain
transformer-first and should not widen Qwen serving routes until
`generateBatchTokens()` parity is proven for Qwen's hybrid cache.

## Remaining Risks / Follow-ups

This is full-KV continuous scheduling only. It does not add Qwen hybrid cache
batching, Gemma layer-pattern continuous batching, sampling support, paged KV,
or prefix cache.

When multiple initial rows are all long, this favors responsiveness and
observable progress over a single batched full-prefill call. If future endpoint
benchmarks show a throughput regression on homogeneous long-prompt batches, the
next refinement should batch same-sized prefill chunks across prefilling rows
while preserving ready-row priority.
