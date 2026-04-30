# Serve Agent Prompt Cache Retention

## Summary

Serve now keeps four distinct prompt-boundary snapshots by default and replaces
duplicate exact entries instead of burning retention slots on repeated identical
prompts. This mitigates Pi-style two-agent Gemma/Qwen cache churn while leaving
family-owned snapshot/fork correctness unchanged.

## Files Reviewed

- `packages/serve/src/engine/prefix-cache-entry.ts`
- `packages/serve/src/engine/prefix-cache.ts`
- `packages/serve/src/runtime/strategy.ts`
- `packages/serve/src/engine/generation.ts`
- `packages/serve/src/engine/continuous.ts`
- `packages/serve/src/engine/index.ts`
- `packages/serve/src/model-loading/router.ts`
- `packages/serve/src/engine/prefix-cache.test.ts`
- `packages/serve/src/engine/engine.test.ts`

## Incident

One Pi/Gemma agent session could warm and reuse the prompt-prefix cache. Adding
a second divergent Pi session made warm repeats miss because the default retained
one prompt-boundary snapshot per served model. Gemma layer-pattern and Qwen
hybrid caches are exact-boundary caches, so they cannot fall back to an arbitrary
shared `AGENTS.md` prefix unless a family-owned snapshot supports that fork.

## Change

The serving runtime default now retains four prompt-boundary snapshots per served
model. This keeps a small set of active agent sessions warm without changing
family-owned cache semantics. Operators can still lower or raise
`--prompt-prefix-cache-max-entries`, and `--prompt-prefix-cache-max-bytes`
remains the explicit memory cap for large-context deployments.

Exact duplicate prompt-boundary stores now replace the existing retained entry's
snapshot rather than appending another entry. This keeps the default four entries
available for four distinct prompt boundaries.

The regression coverage now includes:

- divergent exact-boundary entries for Gemma-style `["full", "sliding"]` and
  Qwen-style `["linear-recurrent", "full"]` cache layer kinds
- Gemma-style A/B/A warm replay through the transformer generation engine
- Qwen-style A/B/A warm replay through the transformer generation engine
- one Gemma engine and one Qwen engine behind the same model router, proving
  model-local prompt caches stay isolated

## Tensor Lifetime Audit

The production change updates host-side retention policy only. No `MxArray`
ownership, cache snapshot/fork implementation, scheduler loop, or
tensor-producing expression changed.

The new tests use existing fake cache/model helpers and do not introduce new
runtime tensor lifetimes in production code.

## Memory / Performance Evidence

Focused tests:

- `bun test packages/serve/src/engine/prefix-cache.test.ts packages/serve/src/engine/engine.test.ts packages/serve/src/engine/routing.test.ts`
- `bun test packages/serve/src/cli.test.ts packages/serve/src/http/server.test.ts packages/serve/src/runtime/strategy.test.ts`
- `bun test packages/serve`
- `bun run validate`
- `bun run regression:qwen-gemma -- --profile quick`

Real cached model evidence:

- `bun run regression:qwen-gemma -- --profile real --report-dir .tmp/qwen-gemma-regression-cache-retention-dense` passed for `mlx-community/Qwen3.6-27B-4bit` and `google/gemma-4-E2B-it`.
  - Qwen decode smoke: `generation_tps=29.166`, `evals_per_token=1.00`, `active_delta=0.018 GB`.
  - Gemma decode smoke: `generation_tps=81.934`, `evals_per_token=1.00`, `active_delta=-0.005 GB`.
  - Qwen chat/responses/Anthropic protocol rungs each recorded `prompt_cache_hits=1` and `prompt_cache_read_tokens=278`.
  - Gemma chat/responses/Anthropic protocol rungs each recorded `prompt_cache_hits=1` and `prompt_cache_read_tokens=276`.
  - Qwen `32768x128+128x32` and Gemma `5000x128+128x32` mixed long/short fairness rungs passed.
- Targeted MoE chat-cache smokes passed because the generic `real` regression profile's Qwen active-memory budget is dense-model-sized.
  - `bun run bench:serve --model unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit --model-id qwen-moe-local --rungs 128x16@1 --trials 1 --report-json .tmp/qwen-gemma-regression-cache-retention-moe/serve/qwen-moe-chat-stream.json --request-timeout-ms 3600000 --max-concurrent-requests 1 --max-batch-size 8 --batch-window-ms 2 --protocol chat --max-prompt-tokens 512 --max-total-tokens 1024 --greedy --stream` recorded `prompt_cache_hits=1`, `prompt_cache_read_tokens=278`, `cache_read_tokens=139`, and `routes=continuous:eligible=1`.
  - `bun run bench:serve --model mlx-community/gemma-4-26b-a4b-it-4bit --model-id gemma-moe-local --rungs 128x16@1 --trials 1 --report-json .tmp/qwen-gemma-regression-cache-retention-moe/serve/gemma-moe-chat-stream.json --request-timeout-ms 3600000 --max-concurrent-requests 1 --max-batch-size 8 --batch-window-ms 2 --protocol chat --max-prompt-tokens 512 --max-total-tokens 1024 --greedy --stream` recorded `prompt_cache_hits=1`, `prompt_cache_read_tokens=284`, `cache_read_tokens=142`, and `routes=continuous:eligible=1`.

Both focused suites passed locally.

Memory tradeoff: the default can retain up to four prompt snapshots instead of
one. That is intentional for agent serving, where two or more divergent
conversations must not evict each other immediately. Large-context operators
should pair the count with `--prompt-prefix-cache-max-bytes`; snapshots over the
byte budget are disposed instead of retained. Duplicate exact-boundary stores no
longer multiply retained entries for the same prompt boundary.

## Independent Review

GPT-5.5 xhigh sub-agent review found no blocker for the default bump and
recommended duplicate exact-boundary replacement so retention capacity represents
distinct prompt boundaries. That follow-up is included in this tranche.

## Remaining Risks / Follow-ups

- Cold concurrent requests still miss until one request writes a completed
  snapshot. That is expected and not changed here.
- Shared `AGENTS.md` prefix reuse across divergent exact-boundary prompts still
  requires an exact retained snapshot at that boundary or a future family-owned
  cache backend that supports the fork.
- A future serving QA tranche should add automated real checkpoint A/B/A Pi
  smokes for dense Gemma, Gemma MoE, dense Qwen, Qwen MoE, and mixed Gemma/Qwen
  multi-model serving using the repo-local `serve-cache-qa` skill. This tranche
  adds deterministic synthetic A/B/A unit coverage plus real repeated-chat cache
  evidence.
- `bun run regression:qwen-gemma -- --profile real --qwen-model unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit --gemma4-model mlx-community/gemma-4-26b-a4b-it-4bit --report-dir .tmp/qwen-gemma-regression-cache-retention-moe` stops before generation because the current `regression-model-matrix.ts` default Qwen active-memory budget is `16.5 GB` and the cached Qwen A3B MoE load reports `20.742 GB`. The targeted MoE serving cache smokes above cover the cache behavior without changing regression budgets in this tranche.

## Out-of-scope Drift Noticed

- `prompt_cache_key` remains parsed metadata on OpenResponses and is not a
  cache partition/sharing key in the engine.
