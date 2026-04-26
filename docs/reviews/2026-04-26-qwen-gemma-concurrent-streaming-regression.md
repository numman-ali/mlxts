# Serving Regression Evidence: Qwen/Gemma Concurrent Streaming

## Summary

The real serving regression matrix now covers concurrent streamed completions for
Qwen 3.6 text and Gemma 4. The new rungs prove the endpoint can stream two
simultaneous greedy requests through the continuous scheduler, not just route a
single streamed request or batch buffered requests.

## Files Reviewed

- `packages/serve/scripts/regression-serve-matrix.ts`

## Evidence

Focused regression test:

- `bun test packages/serve/scripts/regression-serve-matrix.test.ts`

Real model matrix:

- `bun run packages/serve/scripts/regression-serve-matrix.ts --real-models --qwen-model mlx-community/Qwen3.6-27B-4bit --gemma4-model google/gemma-4-E2B-it --report-dir .tmp/qwen-gemma-regression-stream-concurrent --request-timeout-ms 3600000`

Observed Qwen 3.6 concurrent streaming `128x32@2`:

- `routes=continuous:eligible=2`
- `continuous_admissions=1`
- `continuous_admission_rows=2`
- `continuous_scheduler_phases=7`
- `max_generation_batch=2`
- `stream_chunks=64`
- `max_stream_chunk_gap_ms=45.9`
- `peak_memory=16.244 GB`
- `active_delta=0.000 GB`

Observed Gemma 4 concurrent streaming `128x32@2`:

- `routes=continuous:eligible=2`
- `continuous_admissions=1`
- `continuous_admission_rows=2`
- `continuous_scheduler_phases=7`
- `max_generation_batch=2`
- `stream_chunks=34`
- `max_stream_chunk_gap_ms=87.1`
- `peak_memory=9.671 GB`
- `active_delta=0.000 GB`

## Guardrail Added

`ServeRegressionBudget.expectEveryRequestStreamed` makes streaming evidence
per-request instead of aggregate-only. When enabled, every request row in the
benchmark report must include TTFT, streamed chunks, streamed bytes, and a
`stop` or `length` finish reason.

## Remaining Risks

This is still greedy-only. Sampled/model-native-default requests remain on the
single-request route until sampled batched decode is implemented.

This does not yet add concurrent long-context streaming rungs. Keep those as
separate capability smoke work because they are much more expensive on local
hardware.
