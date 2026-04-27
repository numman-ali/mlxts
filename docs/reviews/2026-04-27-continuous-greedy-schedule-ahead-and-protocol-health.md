# Runtime Review: Continuous Greedy Schedule-Ahead And Protocol Health

## Summary

Continuous greedy decode now schedules the next token before synchronizing the
current token to the host when the row set is guaranteed to continue unchanged.
This narrows the serving gap to the in-process async lookahead path for the
safe exact-length benchmark case while keeping EOS, sampled generation, and
repetition-penalty rows on the conservative path.

The real Qwen/Gemma serving regression also now covers protocol-health rungs
for OpenAI chat completions, OpenAI Responses, and Anthropic Messages. These
text protocols get explicit admission cushions because chat templates add
tokens beyond the benchmark prompt target; completions throughput rungs remain
tightly bounded token-array runs.

## Files Reviewed

- `packages/transformers/src/infrastructure/generation/continuous-batch.ts`
- `packages/transformers/src/infrastructure/generation/continuous-batch-decode.ts`
- `packages/transformers/src/infrastructure/generation/continuous-batch.test.ts`
- `packages/serve/scripts/regression-serve-matrix.ts`
- `packages/serve/scripts/regression-serve-matrix.test.ts`
- `packages/serve/src/server.test.ts`

## Tensor Lifetime Audit

The schedule-ahead path is limited to greedy decode with no EOS tokens,
temperature `0`, no repetition penalty, and rows that all have at least one
more required token. That means the scheduled next token is still semantically
valid before host-side callbacks observe the current token.

The new `nextToken` ownership is explicit. If every row finishes, it is freed
before the current emitted token and cache are disposed. If rows are filtered,
the scheduled token is filtered with `tokenRows()` and the original token is
freed. If the schedule-ahead model call throws, or if `tokenTensorToIds()`,
telemetry, filtering, or a user `onToken` callback throws after schedule-ahead
has allocated a token, the catch path frees both the scheduled token and the
emitted token before the scheduler rejects the active requests and disposes the
cache through the existing failure path.

The unit test `recovers when a token callback fails after greedy lookahead is
scheduled` covers the callback failure path, and `recovers when greedy
lookahead scheduling fails` covers the schedule-ahead model-call failure path.
Both verify the scheduler can accept a new request afterward.

## Memory / Performance Evidence

Focused validation passed:

- `bun test packages/serve/scripts/regression-serve-matrix.test.ts packages/serve/scripts/benchmark-serve-completions.test.ts packages/serve/src/protocols/anthropic-messages.test.ts packages/serve/src/server.test.ts packages/transformers/src/infrastructure/generation/continuous-batch.test.ts`
  passed `85` tests.
- `bun run typecheck` passed across all packages.

Targeted Qwen endpoint probe after schedule-ahead:

- `bun run bench:serve --model mlx-community/Qwen3.6-27B-4bit --model-id qwen-local --rungs 1024x128@1 --trials 1 --no-warmup --stream --greedy --ignore-eos --report-json .tmp/qwen36-serve-greedy-schedule-ahead.json --max-concurrent-requests 1 --max-batch-size 8 --batch-window-ms 2 --gpu-memory-utilization 0.85 --request-timeout-ms 3600000`
  measured `28.718 tok/s` post-TTFT, peak memory `16.509 GB`, flat active
  memory, and route `continuous:eligible`.

Generation benchmark coverage:

- `bun run bench:generation:parity --model mlx-community/Qwen3.6-27B-4bit --prompt-tokens 1024 --generation-tokens 128 --trials 1 --memory-sample-interval 16 --skip-mlx-lm-reference`
  ran inside the formal regression and measured `28.960 tok/s`.
- `bench:generation` is still the local decode-smoke surface for isolated
  generation changes; this tranche used `bench:generation:parity` through
  `regression:qwen-gemma` because the changed behavior is tied to real
  Qwen/Gemma checkpoint serving evidence.

Formal real Qwen/Gemma regression:

- `bun run regression:qwen-gemma -- --profile real --report-dir .tmp/qwen-gemma-protocol-schedule-ahead-final --request-timeout-ms 3600000`
  passed.
- Qwen decode smoke `1024x128` measured `28.797 tok/s`, peak memory
  `17.184 GB`, active memory delta `0.018 GB`, active slope
  `0.14 MB/token`, and `evals_per_token=1.00`.
- Gemma 4 decode smoke `1024x128` measured `81.432 tok/s`, peak memory
  `9.893 GB`, active memory delta `-0.005 GB`, active slope
  `-0.04 MB/token`, and `evals_per_token=1.00`.
- Qwen serving `1024x128@1` streaming measured `28.368 tok/s` post-TTFT,
  peak memory `16.509 GB`, flat active memory, and route
  `continuous:eligible`.
- Gemma 4 serving `1024x128@1` streaming measured `80.659 tok/s` post-TTFT,
  peak memory `9.772 GB`, flat active memory, and route
  `continuous:eligible`.

Protocol-health rungs passed against real checkpoints:

- Qwen chat: `16` completion tokens, `15` stream chunks, finish `stop`,
  route `continuous:eligible`.
- Qwen Responses: `16` completion tokens, `15` stream chunks, finish
  `length`, route `continuous:eligible`.
- Qwen Anthropic Messages: `16` completion tokens, `15` stream chunks,
  finish `max_tokens`, route `continuous:eligible`.
- Gemma 4 chat: `16` completion tokens, `16` stream chunks, finish `stop`,
  route `continuous:eligible`.
- Gemma 4 Responses: `16` completion tokens, `16` stream chunks, finish
  `length`, route `continuous:eligible`.
- Gemma 4 Anthropic Messages: `16` completion tokens, `16` stream chunks,
  finish `max_tokens`, route `continuous:eligible`.

Fairness rungs passed:

- Qwen `32768x128+128x32` staggered streaming measured `14.287 tok/s`
  post-TTFT, peak memory `19.277 GB`, flat active memory,
  `max_stream_chunk_gap_ms=641.0`, and `continuous_admissions=2`.
- Gemma 4 `5000x128+128x32` staggered streaming measured `69.212 tok/s`
  post-TTFT, peak memory `9.840 GB`, flat active memory,
  `max_stream_chunk_gap_ms=47.3`, and `continuous_admissions=2`.

## Independent Review

Singer independently reviewed the serving performance gap and identified that
the continuous scheduler still synchronized the emitted token and ran callbacks
before scheduling the next greedy model forward. That matched the measured
endpoint gap and directly informed the schedule-ahead seam.

Feynman reviewed serving protocol coverage and called out that real
Qwen/Gemma protocol-health rungs were missing for chat completions, Responses,
and Anthropic Messages, and that Anthropic stream flushing needed focused
coverage. The regression now includes those real protocol rungs, and
`server.test.ts` includes an Anthropic microtask-heavy streaming flush test.

Darwin reviewed the resulting diff and found three issues before final
validation: Anthropic finish reasons were not accepted by stream lifecycle
budgets, the finish-reason allow-list needed to be protocol-aware, and the
new schedule-ahead tensor needed error-path cleanup. All three were fixed and
covered by focused tests before the real regression was rerun.

## Remaining Risks / Follow-ups

The schedule-ahead optimization is intentionally narrow. EOS-capable rows,
sampled rows, and repetition-penalty rows still use the conservative decode
path until each case has its own correctness proof.

Anthropic protocol-health reports currently record prompt tokens as `0`
because the Anthropic wire usage object exposes output usage during streaming
but not the internal chat-template prompt token count in the final benchmark
metric. This is acceptable for lifecycle health, but throughput comparisons
should continue using completions token-array rungs or protocol-specific usage
normalization work.

The Qwen mixed long-prefill fairness rung still takes minutes. It passes the
current guardrails, but future serving work should keep pushing TTFT and
long-prefill ergonomics without weakening the flat-memory and fairness budgets.
