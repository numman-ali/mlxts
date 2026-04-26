# Runtime Review: Qwen Staggered Continuous Mask

## Summary

Real Qwen 3.6 staggered continuous serving exposed a valid single-token
left-padded decode shape that failed inside MLX `arange`:
`queryLength=1`, `totalKeyLength=133`, `pastLength=132`. The cache semantics
were correct; the fragile part was building a one-element query position vector
with `arange(start, start + 1)`.

The mask helper now builds one-token query ranges with `array([start])` and
keeps `arange` for multi-token ranges. The real serving regression matrix now
includes a Qwen staggered continuous endpoint rung so delayed row admission is
covered through HTTP telemetry.

## Files Reviewed

- `packages/transformers/src/infrastructure/masks.ts`
- `packages/serve/scripts/regression-serve-matrix.ts`

## Tensor Lifetime Audit

`positionRange()` returns a new owned `MxArray` exactly like the previous
`arange()` call sites. Existing `using` declarations still own and dispose the
returned query-offset tensors in causal, fast-boolean, and left-padded mask
builders.

The fix does not alter cache padding, offset, or filtering semantics. Qwen
full-attention delayed admission still flows through `BatchKVCache.extend()`,
which pads shorter histories on the left and preserves per-row logical offsets.
Qwen linear-attention state still requires exhausted linear padding before
extension.

## Memory / Performance Evidence

Focused checks:

- `bun test packages/transformers/src/infrastructure/masks.test.ts packages/serve/scripts/regression-serve-matrix.test.ts`
- In-process real Qwen staggered scheduler probe against `mlx-community/Qwen3.6-27B-4bit`
- `bun run bench:generation:parity --model mlx-community/Qwen3.6-27B-4bit --prompt-tokens 128 --generation-tokens 32 --trials 1 --memory-sample-interval 16 --skip-mlx-lm-reference`
- `bun run bench:serve --model mlx-community/Qwen3.6-27B-4bit --model-id qwen-local --rungs 128x32@2 --trials 1 --report-json .tmp/qwen36-continuous-staggered-candidate.json --request-timeout-ms 3600000 --max-concurrent-requests 1 --max-batch-size 8 --batch-window-ms 2 --request-stagger-ms 100 --greedy --ignore-eos`
- `bun run packages/serve/scripts/regression-serve-matrix.ts --real-models --qwen-model mlx-community/Qwen3.6-27B-4bit --gemma4-model google/gemma-4-E2B-it --report-dir .tmp/qwen-gemma-regression-qwen-staggered-continuous/serve-rerun --request-timeout-ms 3600000`

Focused tests passed: 20 pass, 0 fail.

The tiny in-process `bench:generation:parity` guard, which is the paired
`bench:generation` surface for local-only generation checks, completed with
`generation_tps=30.854`, `peak_memory=15.731 GB`,
`active_delta=0.001 GB`, `active_slope=0.02 MB/token`, and
`evals_per_token=1.00`. The external `mlx-lm` reference was intentionally
skipped for this guard because this tranche fixes a delayed-admission mask
crash, not a publishable decode-speed claim.

The in-process real Qwen staggered scheduler probe completed both delayed rows
with 32 generated tokens.

Manual staggered Qwen endpoint evidence:

- `routes=continuous:eligible=2`
- `completion_tokens=64`
- `continuous_admissions=2`
- `continuous_admission_rows=3`
- `continuous_scheduler_phases=9`
- `max_continuous_batch=2`
- `max_generation_batch=2`
- `static_batches=0`
- `peak_memory=15.904 GB`
- `active_delta=0.000 GB`

The full real serve matrix rerun passed. Qwen streaming remained
`single:streaming=1`, Qwen non-streaming simultaneous continuous passed with
minimum scheduler evidence, Qwen staggered continuous passed with two
admissions and batch size 2, and Gemma 4 streaming/non-streaming controls
remained healthy.

## Independent Review

Popper independently audited the failure and recommended keeping the
one-token `positionRange()` helper. The review concluded that
`array([pastLength], "int32")` is semantically equivalent to
`arange(pastLength, pastLength + 1, 1, "int32")` for `queryLength === 1`,
and that cache padding/offset semantics should not change.

## Remaining Risks / Follow-ups

This closes the delayed-admission mask failure for greedy non-streaming Qwen
continuous serving. Qwen streaming continuous remains deliberately out of
scope and should need separate SSE lifecycle, cancellation, and cadence
evidence before routing.
