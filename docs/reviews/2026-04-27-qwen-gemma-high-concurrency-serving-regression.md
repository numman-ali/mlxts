# Runtime Review: Qwen/Gemma High-Concurrency Serving Regression

## Summary

The real serving regression matrix now covers higher-concurrency streaming
continuous routes for Qwen 3.6 and Gemma 4 at `128x16@4` and `128x16@8`.
These rungs protect the behavior that matters for real local serving: each
request must stream, each server request must emit stream evidence, the route
must stay `continuous:eligible`, static/admission batching must stay out of the
path, token-pressure telemetry must be present, and the scheduler must reach the
expected max generation batch size.

The budgets also add client-observed per-request TTFT and server-observed
scheduler queued-time ceilings so higher concurrency cannot pass on aggregate
throughput while one request silently starves.

## Files Reviewed

- `packages/serve/scripts/regression-serve-matrix.ts`
- `packages/serve/scripts/regression-serve-matrix.test.ts`
- `docs/serving-runtime-strategy.md`
- `PLAN.md`
- `MEMORY.md`

## Tensor Lifetime Audit

This change is regression harness and documentation only. It does not add model
math, cache tensor mutation, sampling kernels, MLX array allocation, or FFI
handle ownership. The touched code reads benchmark report JSON and applies
host-side assertions over recorded metrics.

## Memory / Performance Evidence

Focused unit coverage passed:

- `bun test packages/serve/scripts/regression-serve-matrix.test.ts`

The full real Qwen/Gemma profile passed after the matrix update:

- `bun run regression:qwen-gemma -- --profile real --report-dir .tmp/qwen-gemma-regression-high-concurrency`

Key high-concurrency endpoint results from that run:

- Qwen `128x16@4`: `18.936` completion tok/s, `15.108` post-TTFT tok/s,
  `2.385s` mean TTFT, `2.441s` max client TTFT, `2.291s` max scheduler queued,
  `13` scheduler phases, `4` admission rows, max generation batch size `4`,
  and `17.136 GB` peak memory.
- Qwen `128x16@8`: `19.038` completion tok/s, `7.923` post-TTFT tok/s,
  `4.829s` mean TTFT, `4.948s` max client TTFT, `4.674s` max scheduler queued,
  `25` scheduler phases, `8` admission rows, max generation batch size `8`,
  and `18.636 GB` peak memory.
- Gemma `128x16@4`: `91.587` completion tok/s, `37.615` post-TTFT tok/s,
  `299.9ms` mean TTFT, `300.0ms` max client TTFT, `235.1ms` max scheduler
  queued, `13` scheduler phases, `4` admission rows, max generation batch size
  `4`, and `9.907 GB` peak memory.
- Gemma `128x16@8`: `147.835` completion tok/s, `35.522` post-TTFT tok/s,
  `443.0ms` mean TTFT, `469.2ms` max client TTFT, `407.8ms` max scheduler
  queued, `25` scheduler phases, `8` admission rows, max generation batch size
  `8`, and `10.362 GB` peak memory.

Budget thresholds are based on local real-endpoint probes against cached Qwen
and Gemma checkpoints:

- Qwen `128x16@4` streaming continuous:
  `18.261` completion tok/s, `11.394` post-TTFT tok/s, `1.897s` mean TTFT,
  `2.618s` max client TTFT, `684.8ms` max stream gap, `17` scheduler phases,
  `9` continuous admission rows, max generation batch size `4`, and
  `16.246 GB` peak memory.
- Qwen `128x16@8` streaming continuous:
  `20.474` completion tok/s, `8.932` post-TTFT tok/s, `4.572s` mean/client
  TTFT, `256.4ms` max stream gap, `25` scheduler phases, `8` admission rows,
  max generation batch size `8`, and `18.636 GB` peak memory.
- Gemma `128x16@4` streaming continuous:
  `85.730` completion tok/s, `37.875` post-TTFT tok/s, `350.1ms` mean/client
  TTFT, `60.3ms` max stream gap, `13` scheduler phases, `4` admission rows,
  max generation batch size `4`, and `9.907 GB` peak memory.
- Gemma `128x16@8` streaming continuous:
  `138.643` completion tok/s, `35.059` post-TTFT tok/s, `495.0ms`
  mean/client TTFT, `62.0ms` max stream gap, `25` scheduler phases, `8`
  admission rows, max generation batch size `8`, and `10.362 GB` peak memory.

The new matrix budgets are looser than these probes to avoid turning normal
thermal and local-machine variance into false failures while still catching
route drift, output buffering, scheduler starvation, or major throughput/memory
regressions.

## Independent Review

Schrodinger reviewed the regression design independently. That review confirmed
the threshold values matched the probes, recommended making the request TTFT
budget explicitly client-observed, and called out missing direct tests for the
new TTFT and scheduler queued-time assertions. The implementation now uses
`maxClientRequestTtftMs` and tests both client TTFT and server scheduler queue
failures.

The review also suggested exact admission counts for simultaneous `@4` and `@8`
rungs. This was intentionally not adopted for Qwen because the real `@4` probe
legitimately admitted rows in multiple waves while still proving continuous
routing, max generation batch size `4`, and all per-request stream evidence.
The guardrail therefore asserts minimum admissions/rows plus expected max batch
size instead of making scheduler-internal wave count a brittle contract.

## Remaining Risks / Follow-ups

These rungs prove greedy streaming continuous behavior at short prompt/output
sizes. They do not replace the longer mixed/staggered rungs for long-prefill
fairness, and they do not prove higher-concurrency model-native sampled
streams.

The next scheduler quality pass should use this evidence to decide whether
fairness policy needs active changes, rather than adding policy before a
regression shows starvation.
