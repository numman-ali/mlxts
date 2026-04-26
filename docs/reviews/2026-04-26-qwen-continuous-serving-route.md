# Runtime Review: Qwen Continuous Serving Route

## Summary

Qwen 3.5/3.6 text checkpoints with a model-owned hybrid batch cache can now use
the continuous scheduler for greedy non-streaming serving. The route is narrow:
Qwen streaming still stays on the single-request path, sampled/model-native
defaults still fall back, and Qwen-like test models without a model-owned batch
cache remain static-only.

The change turns the previously implemented `Qwen3_5TextBatchCache.extend()`
prerequisite into a served endpoint path only after adding Qwen-specific
continuous scheduler tests and real Qwen 3.6 endpoint evidence.

## Files Reviewed

- `packages/serve/src/transformers-engine-routing.ts`

## Tensor Lifetime Audit

The production change is a route-decision guard only. It does not create,
retain, filter, extend, or dispose tensor handles directly.

The widened route depends on the already-reviewed scheduler and Qwen hybrid
cache ownership paths from `docs/reviews/2026-04-26-cache-generic-continuous-batching.md`:
the scheduler owns active and prefilling caches, disposes them in `finally` and
failure paths, and `Qwen3_5TextBatchCache.extend()` prepares replacement linear
state before mutating the target cache.

The new Qwen tests exercise real-family continuous scheduling through
staggered admission and chunked waiting prefill, both of which force the hybrid
cache extension path used by serving.

## Memory / Performance Evidence

Focused checks:

- `bun test packages/serve/scripts/regression-serve-matrix.test.ts packages/serve/src/transformers-engine.test.ts packages/transformers/src/families/qwen3_5/model.test.ts`
- `bun run regression:qwen-gemma -- --profile real --report-dir .tmp/qwen-gemma-regression-qwen-continuous`

Focused tests passed: 46 pass, 0 fail.

Real Qwen/Gemma regression passed. Key evidence:

- Qwen decode smoke: `generation_tps=29.085`, `peak_memory=17.184 GB`,
  `active_delta=0.018 GB`, `active_slope_mb_per_token=0.14`,
  `evals_per_token=1.00`.
- Qwen streaming `1024x128@1` remains single-route fallback:
  `routes=single:streaming=1`, zero static/continuous counters,
  `post_ttft_completion_tps=28.662`, `peak_memory=17.184 GB`.
- Qwen non-streaming `128x32@2` now uses continuous scheduling:
  `routes=continuous:eligible=2`, `continuous_admissions=1`,
  `continuous_admission_rows=2`, `continuous_scheduler_phases=7`,
  `max_continuous_batch=2`, `max_generation_batch=2`, `static_batches=0`,
  `completion_tps=25.771`, `peak_memory=16.244 GB`, `active_delta=0.000 GB`.
- Gemma 4 streaming control `1024x128@1` remains single-route fallback:
  `routes=single:streaming=1`, zero static/continuous counters,
  `post_ttft_completion_tps=81.487`, `peak_memory=9.892 GB`.
- Gemma 4 non-streaming control `128x32@2` remains continuous:
  `routes=continuous:eligible=2`, `continuous_admissions=1`,
  `continuous_scheduler_phases=7`, `max_generation_batch=2`,
  `completion_tps=61.255`, `peak_memory=9.671 GB`.

## Independent Review

Herschel independently audited Qwen continuous readiness and identified the
route-gating blockers: Qwen was still excluded from continuous eligibility,
streaming would become eligible if we widened the model set blindly, and
Qwen-specific scheduler proof was missing.

Nietzsche independently reviewed the endpoint evidence harness and recommended
using the existing `bench:serve`/real regression counters for the proof:
`continuous:eligible=2`, zero static batches, one continuous admission,
seven scheduler phases, batch size 2, token-count coverage, and memory budgets.

The implementation follows both reviews by adding Qwen-specific scheduler
tests, narrowly gating Qwen continuous eligibility to non-streaming model-owned
hybrid caches, and updating the real regression budgets to assert the new route.

## Remaining Risks / Follow-ups

Qwen streaming remains deliberately single-route. Enabling streaming continuous
for Qwen needs its own SSE lifecycle and cancellation proof, not this
non-streaming route evidence.

This proves a simultaneous `128x32@2` endpoint rung. The next serving-quality
step is a staggered Qwen endpoint rung so delayed row admission is proven
through HTTP telemetry, not only transformer-level scheduler tests.
