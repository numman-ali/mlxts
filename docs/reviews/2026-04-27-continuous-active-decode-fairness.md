# Runtime Review: Continuous Active Decode Fairness

## Summary

Continuous scheduling now protects active decode from long prompt-prefill
head-of-line blocking. When active rows are already producing tokens and pending
prefill work is long, the scheduler decodes a bounded quantum before resuming
prefill in smaller active-aware chunks. Short pending prompts still bypass that
guard so ordinary staggered requests can join the active batch instead of losing
batching.

Serving exposes the policy through CLI/programmatic knobs and benchmark reports:
`active_prefill_step_size` and `active_decode_steps_per_prefill_chunk`.

## Files Reviewed

- `packages/transformers/src/infrastructure/generation/continuous-batch.ts`
- `packages/transformers/src/infrastructure/generation/continuous-batch-helpers.ts`
- `packages/transformers/src/infrastructure/generation/continuous-batch-lifecycle.ts`
- `packages/transformers/src/infrastructure/generation/continuous-batch-policy.ts`
- `packages/transformers/src/infrastructure/generation/continuous-batch-types.ts`
- `packages/transformers/src/infrastructure/generation/continuous-batch.test.ts`
- `packages/transformers/src/families/qwen3_5/model.test.ts`
- `packages/serve/src/serve-runtime-strategy.ts`
- `packages/serve/src/model-server.ts`
- `packages/serve/src/server-info.ts`
- `packages/serve/src/transformers-engine.ts`
- `packages/serve/src/transformers-engine-continuous.ts`
- `packages/serve/src/cli-options.ts`
- `packages/serve/src/cli.ts`
- `packages/serve/src/cli.test.ts`
- `packages/serve/src/server.test.ts`
- `packages/serve/src/model-server.test.ts`
- `packages/serve/src/transformers-engine.test.ts`
- `packages/serve/scripts/benchmark-serve-options.ts`
- `packages/serve/scripts/benchmark-serve.ts`
- `packages/serve/scripts/benchmark-serve-options.test.ts`
- `packages/serve/scripts/benchmark-serve.test.ts`
- `packages/serve/scripts/regression-serve-matrix.ts`
- `packages/serve/scripts/regression-serve-matrix.test.ts`
- `packages/serve/README.md`
- `MEMORY.md`

## Tensor Lifetime Audit

The scheduling change does not add new tensor-producing model math. It changes
the host-side order of existing scheduler work:

- `#decodeOneStep()` still owns and frees the emitted token tensor exactly as
  before.
- `#prefillOneChunk()` still delegates chunked prefill to the existing helper
  and only changes the chunk size selected while active rows exist.
- The new decode quantum is a host-side counter. The short-pending-prefill guard
  reads request token counts and existing prefill cursors; it does not allocate
  MLX arrays.

The risk to watch is scheduler policy, not tensor lifetime: aggressive decode
priority can reduce batching or delay long-request TTFT. The final policy keeps
short pending prompts admission-friendly and bounds decode priority before long
prefill resumes.

## Memory / Performance Evidence

Focused tests passed:

- `bun test packages/transformers/src/infrastructure/generation/continuous-batch.test.ts packages/serve/src/cli.test.ts packages/serve/src/server.test.ts packages/serve/src/model-server.test.ts packages/serve/src/transformers-engine.test.ts`
- `bun test packages/serve/scripts/benchmark-serve-options.test.ts packages/serve/scripts/benchmark-serve.test.ts packages/serve/scripts/regression-serve-matrix.test.ts`
- `bun test packages/transformers/src/families/qwen3_5/model.test.ts`
- `bun run typecheck`
- `bun run lint`
- `bun run check:assertions`
- `bun run check:file-lines`
- `bun run check:tensor-lifetimes`
- `bun run check:runtime-review`
- `bun run check:coverage`

Targeted real Qwen/Gemma probes passed:

- Qwen mixed long/short one-off:
  `bun run bench:serve --model mlx-community/Qwen3.6-27B-4bit --model-id qwen-local --mixed-rungs 32768x128+128x32 --trials 1 --no-warmup --report-json .tmp/qwen36-mixed-long-short-active-decode.json --request-timeout-ms 3600000 --max-concurrent-requests 1 --max-batch-size 8 --batch-window-ms 2 --greedy --request-stagger-ms 100 --stream --ignore-eos`
  completed with max visible stream gap `574.0 ms`, short request duration
  `12.480 s`, short request max stream gap `574.0 ms`, peak memory
  `19.277 GB`, and active delta `0.014 GB`.
- Qwen staggered short-prompt batching control:
  `bun run bench:serve --model mlx-community/Qwen3.6-27B-4bit --model-id qwen-local --rungs 128x32@2 --trials 1 --no-warmup --report-json .tmp/qwen36-staggered-active-short-fix.json --request-timeout-ms 3600000 --max-concurrent-requests 1 --max-batch-size 8 --batch-window-ms 2 --greedy --request-stagger-ms 100 --ignore-eos`
  restored `continuous_admission_rows=3` and `max_generation_batch=2`.
- Gemma mixed control:
  `bun run bench:serve --model google/gemma-4-E2B-it --model-id gemma-local --mixed-rungs 5000x128+128x32 --trials 1 --no-warmup --report-json .tmp/gemma4-mixed-long-short-active-decode.json --request-timeout-ms 3600000 --max-concurrent-requests 1 --max-batch-size 8 --batch-window-ms 2 --greedy --request-stagger-ms 100 --stream --ignore-eos`
  completed with max visible stream gap `56.9 ms`, peak memory `9.840 GB`,
  and flat active memory.

Formal real regression passed:

- `bun run regression:qwen-gemma -- --profile real --report-dir .tmp/qwen-gemma-active-decode-real --request-timeout-ms 3600000`

That command includes `bench:generation:parity` decode smokes for cached Qwen
and Gemma before the endpoint matrix. Standalone `bench:generation` was not run
separately for this slice because the production diff is scheduler policy and
serving option/report plumbing, not model math or single-request generation
kernel code.

Key formal Qwen mixed result:

- Report:
  `.tmp/qwen-gemma-active-decode-real/serve/qwen36-mixed-long-short-staggered-stream.json`
- Aggregate max visible stream gap: `625.4 ms`, now guarded at `<= 1000 ms`
  instead of the prior loose `6000 ms`.
- Short `128x32` request: duration `14.548 s`, TTFT `7.769 s`,
  post-TTFT decode `4.573 tok/s`, max visible stream gap `625.4 ms`.
- Before this fix, the same formal shape had short duration about `92.212 s`,
  post-TTFT decode about `0.367 tok/s`, and max visible stream gap about
  `5348 ms`.
- Long `32768x128` request still completed, peak memory was `19.277 GB`, and
  active memory delta was effectively flat.

Gemma formal mixed control also passed:

- Report:
  `.tmp/qwen-gemma-active-decode-real/serve/gemma4-mixed-long-short-staggered-stream.json`
- Max visible stream gap: `56.5 ms`
- Peak memory: `9.840 GB`

## Independent Review

Hilbert reviewed the scheduler shape and identified the root cause: the loop was
advancing one long prefill chunk before every active decode step, so a short
active request paid the long-prefill chunk wall for each output token. Hilbert
recommended a bounded active decode quantum rather than draining active rows
indefinitely.

Newton independently checked reference scheduler patterns and highlighted the
second key detail: streaming cadence only improves if the scheduler yields after
protected decode work instead of always doing prefill before every yield. Newton
also called out the long-TTFT risk, which is why the final implementation keeps
the quantum bounded and preserves fast admission for short pending prompts.

## Remaining Risks / Follow-ups

The policy is still a fixed heuristic. It is now much better for the observed
Qwen long-prefill/short-output shape, but future scheduler work should evolve
toward token-budget or latency-SLO-aware scheduling rather than only fixed
quanta.

The max server-side silent event gap can still be several seconds during huge
Qwen prefill because server progress events are coarser than visible SSE output.
The regression now guards client-visible stream cadence directly, which is the
operator-visible issue this change fixes.
