# Runtime Review: Continuous Decode Lookahead

## Summary

Continuous batch decode now schedules the next sampled token with MLX async eval
instead of synchronously materializing it at the end of every scheduler step.
The decode loop also reuses the emitted token tensor as the next model input
when active rows are unchanged, avoiding a host round-trip that converted token
tensors to JavaScript numbers and then rebuilt an equivalent MLX tensor.

The paired benchmark helper also gained an explicit
`--mlx-lm-allow-extra-weights` option. This keeps normal `mlx-lm` parity strict
by default while allowing known Gemma 4 checkpoints with extra unused shared-KV
checkpoint tensors to be compared transparently. The helper filters only that
known extra tensor pattern and still runs MLX-LM weight loading in strict mode,
so missing or mismatched expected weights remain benchmark blockers.

## Files Reviewed

- `packages/transformers/src/infrastructure/generation/continuous-batch.ts`
- `packages/transformers/src/infrastructure/generation/continuous-batch-helpers.ts`
- `packages/transformers/scripts/benchmark-common.ts`
- `packages/transformers/scripts/benchmark-common.test.ts`
- `packages/transformers/scripts/benchmark-mlx-lm.py`

## Tensor Lifetime Audit

`sampleNextBatchToken()` now calls `mxAsyncEval(nextToken)`. The returned token
is still owned by the scheduler and is synchronised later when
`tokenTensorToIds()` reads it with `item()` or `toTypedArray()`. This mirrors
the existing single-request generation pattern where the next token is scheduled
asynchronously, then read on the next loop iteration.

`nextInputTensor()` now retains the emitted token tensor when no rows are
filtered. The retained view is disposed by the existing `using nextInput`
scope, while the original emitted token is freed after the next token has been
scheduled. When rows are filtered, the helper still creates an owned gathered
token tensor with `tokenRows()`.

Sampled generation remains safe: categorical sampling still materializes inside
the sampler before RNG key handles are released. The additional async eval on
an already materialized sampled token is harmless and keeps the helper behavior
uniform.

## Memory / Performance Evidence

Focused tests passed:

- `bun test packages/transformers/src/infrastructure/generation/continuous-batch.test.ts packages/serve/scripts/regression-serve-matrix.test.ts packages/transformers/scripts/benchmark-common.test.ts`
- `bun run bench:generation --model mlx-community/Qwen3.6-27B-4bit --prompt-tokens 128 --generation-tokens 32 --trials 1 --memory-sample-interval 16 --skip-mlx-lm-reference`
  measured `generation_tps=30.173`, peak memory `15.731 GB`, flat active
  memory, and `evals_per_token=1.00`.

Paired generation parity:

- Qwen strict reference:
  `bun run bench:generation:parity --model mlx-community/Qwen3.6-27B-4bit --prompt-tokens 128 --generation-tokens 128 --trials 1 --memory-sample-interval 16 --require-mlx-lm-reference --mlx-lm-python .tmp/venvs/mlx-lm-bench/bin/python`
  measured `mlxts=29.651 tok/s`, `mlx-lm=29.580 tok/s`, peak memory
  `15.731 GB` vs `15.681 GB`.
- Qwen `1024x128` strict reference measured `mlxts=29.122 tok/s`,
  `mlx-lm=29.303 tok/s`, peak memory `17.184 GB` vs `17.022 GB`.
- Qwen `10000x128` strict reference measured `mlxts=24.866 tok/s`,
  `mlx-lm=22.504 tok/s`, peak memory `19.219 GB` for both.
- Gemma 4 with explicit extra-weight filtering:
  `bun run bench:generation:parity --model google/gemma-4-E2B-it --prompt-tokens 1024 --generation-tokens 128 --trials 1 --memory-sample-interval 16 --require-mlx-lm-reference --mlx-lm-allow-extra-weights --mlx-lm-python .tmp/venvs/mlx-lm-bench/bin/python`
  measured `mlxts=83.626 tok/s`, `mlx-lm=81.568 tok/s`, peak memory
  `9.893 GB` vs `9.884 GB`.

Serving probes:

- Before this change, the saved Qwen continuous streaming report
  `.tmp/qwen-gemma-active-decode-final/serve/qwen36-completions-stream.json`
  measured `1024x128@1` post-TTFT decode at `24.827 tok/s`.
- After async scheduling:
  `bun run bench:serve --model mlx-community/Qwen3.6-27B-4bit --model-id qwen-local --rungs 1024x128@1 --trials 1 --no-warmup --stream --greedy --ignore-eos --report-json .tmp/qwen36-serve-async-lookahead.json --max-concurrent-requests 1 --max-batch-size 8 --batch-window-ms 2 --gpu-memory-utilization 0.85 --request-timeout-ms 3600000`
  measured post-TTFT decode at `25.769 tok/s`.
- After tensor input reuse:
  `bun run bench:serve --model mlx-community/Qwen3.6-27B-4bit --model-id qwen-local --rungs 1024x128@1 --trials 1 --no-warmup --stream --greedy --ignore-eos --report-json .tmp/qwen36-serve-tensor-reuse.json --max-concurrent-requests 1 --max-batch-size 8 --batch-window-ms 2 --gpu-memory-utilization 0.85 --request-timeout-ms 3600000`
  measured post-TTFT decode at `25.911 tok/s`.
- Qwen `128x32@2` streaming continuous control:
  `.tmp/qwen36-serve-continuous-at2-tensor-reuse.json` measured
  `25.211 tok/s` post-TTFT, `max_generation_batch=2`, and flat active memory.
- Qwen buffered `1024x128@1` control:
  `.tmp/qwen36-serve-buffered-tensor-reuse.json` completed with `13.944`
  completion tok/s including prefill and flat active memory.
- Gemma 4 `1024x128@1` streaming control:
  `.tmp/gemma4-serve-tensor-reuse.json` measured `71.724 tok/s` post-TTFT,
  peak memory `9.772 GB`, and flat active memory.

Formal real Qwen/Gemma regression:

- `bun run regression:qwen-gemma -- --profile real --report-dir .tmp/qwen-gemma-decode-lookahead-real --request-timeout-ms 3600000`
  passed the focused model and serving guardrails.
- Qwen decode smoke `1024x128` measured `29.089 tok/s`, peak memory
  `17.184 GB`, active memory delta `0.018 GB`, and `evals_per_token=1.00`.
- Gemma 4 decode smoke `1024x128` measured `80.242 tok/s`, peak memory
  `9.893 GB`, active memory delta `-0.005 GB`, and `evals_per_token=1.00`.
- Qwen serving `1024x128@1` streaming continuous measured `25.261 tok/s`
  post-TTFT, `max_generation_batch=1`, flat active memory, and route
  `continuous:eligible=1`.
- Qwen serving concurrent streaming controls passed at `128x32@2`,
  `128x16@4`, and `128x16@8`, with `max_generation_batch` matching the
  requested concurrency and zero static/admission batches.
- Gemma 4 serving concurrent streaming controls passed at `128x32@2`,
  `128x16@4`, and `128x16@8`, with `max_generation_batch` matching the
  requested concurrency and zero static/admission batches.
- The mixed long/short fairness rungs passed:
  Qwen `32768x128+128x32` streaming peaked at `19.277 GB` with flat active
  memory and `max_stream_chunk_gap_ms=596.0`; Gemma 4 `5000x128+128x32`
  streaming peaked at `9.841 GB` with flat active memory and
  `max_stream_chunk_gap_ms=64.0`.

## Independent Review

Erdos reviewed the Qwen/Gemma hot path against `.reference/mlx-lm` before the
change and identified the likely serving-specific gap: continuous batching
materialized every next token synchronously, while mlx-lm-style generation uses
async lookahead.

Noether independently reviewed the benchmark ladder and confirmed the existing
scripts are the right evidence surfaces: `bench:generation:parity` for paired
reference claims, `bench:serve` for endpoint behavior, and
`regression:qwen-gemma` for real-model serving guardrails.

Confucius reviewed the uncommitted diff and found one real blocker: the first
implementation of `--mlx-lm-allow-extra-weights` used MLX-LM `strict=False`,
which would have allowed missing expected parameters as well as extra
checkpoint tensors. The helper now filters only the known extra Gemma 4
shared-KV checkpoint tensors and then loads weights strictly. Confucius found no
correctness regression in the continuous-batch async/tensor changes.

## Remaining Risks / Follow-ups

The endpoint still does not expose the same post-TTFT number as the in-process
decode benchmark on Qwen. Current evidence suggests this is mostly endpoint and
streaming-loop overhead rather than model math, because total request time lines
up closely with the paired in-process prefill plus decode total. If we want to
close that remaining serving gap, the next seam should be scheduler yield/SSE
flush cadence and per-token stream work, not another blind model hot-path edit.

The Gemma 4 reference flag deliberately filters only known extra upstream
checkpoint tensors before strict `mlx-lm` reference loading. It must remain
explicit in commands and review artifacts so filtered-reference evidence cannot
be mistaken for the default strict parity posture.
