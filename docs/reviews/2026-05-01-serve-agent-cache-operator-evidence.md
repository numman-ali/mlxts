# Serve Agent Cache Operator Evidence

## Summary

`regression:agent-cache` now writes forensic request evidence into the JSON
report for each cold, warm, and exact replay request. The compact AXI stdout
remains unchanged; the detailed report records client timing/cache fields plus
server route, prompt-prepare, prompt-cache, prefill, and stream summaries so Pi
or OpenAI-compatible client reports can be diagnosed from one artifact.

This tranche does not change cache lookup, snapshot/fork behavior, scheduling,
or protocol usage accounting.

## Files Reviewed

- `packages/serve/scripts/regression-agent-cache.ts`
- `packages/serve/scripts/regression-agent-cache.test.ts`
- `packages/serve/README.md`
- `.agents/skills/serve-cache-qa/SKILL.md`
- `docs/gates-and-milestones.md`
- `continuity.md`
- `MEMORY.md`

## Tensor Lifetime Audit

The changed script code is host-side report assembly over already-emitted
`ServeEvent` objects and benchmark client metrics. It allocates plain objects
and arrays only. No `MxArray`, cache snapshot, tensor-producing primitive,
model forward path, scheduler path, or disposal boundary changed.

## Memory / Performance Evidence

No serving runtime behavior changed. The report now retains per-request
diagnostic objects after each probe. The additional memory is proportional to
the five requests per model probe and existing event slices; real model memory,
prompt-prefix cache retention, and scheduler memory budgets are unchanged.

The expected operator tradeoff is a slightly larger JSON report in exchange for
enough evidence to distinguish cold-concurrent misses, prompt-shape drift, and
real warm-retention failures.

## Validation

Focused validation passed:

```bash
bun test packages/serve/scripts/regression-agent-cache.test.ts
bun test packages/serve/scripts/regression-agent-cache.test.ts packages/serve/src/engine/prefix-cache.test.ts packages/serve/src/engine/engine.test.ts
bun run --filter '@mlxts/serve' typecheck
bun run check:skills
bun run lint
bun run check:tensor-lifetimes
bun run check:runtime-review
```

Targeted Gemma MoE two-active-agent proof passed:

```bash
bun run regression:agent-cache -- --scenarios gemma-moe --max-concurrent-requests 2 --report-json .tmp/agent-cache-regression/operator-gemma-moe.json
```

Result: warm hits `2`, warm server read tokens `618`, warm client cached tokens
`309`, exact replay hits `1`, exact replay client cached tokens `167`. The JSON
report includes five request records: `cold-a`, `cold-b`, `warm-a`, `warm-b`,
and `exact-a`, each with route/cache/stream evidence keyed by response id.

Dense and same-server dense proof passed:

```bash
bun run regression:agent-cache -- --scenarios qwen-dense,gemma-dense,multi-dense --max-concurrent-requests 2 --report-json .tmp/agent-cache-regression/operator-dense.json
```

Results:

- `qwen-dense`: warm hits `2`, server read tokens `608`, client cached tokens
  `304`, exact replay client cached tokens `164`.
- `gemma-dense`: warm hits `2`, server read tokens `602`, client cached tokens
  `301`, exact replay client cached tokens `163`.
- `multi-dense`: warm hits `4`, server read tokens `1210`, client cached tokens
  `605`, exact replay client cached tokens `327`.

## Independent Review

Helmholtz performed a read-only Phase 9 serving/cache pass and recommended an
operator-evidence tranche before deeper cache-backend work. The review called
out that the server warm-cache invariant is already covered, while report
diagnostics were not rich enough to explain Pi/client-shaped misses without
manual log reconstruction.

## Out-of-Scope Drift Noticed

The current proof still does not claim arbitrary shared `AGENTS.md` prefix reuse
for exact-boundary Qwen hybrid or Gemma layer-pattern caches. That remains a
future family-owned cache backend capability, not a serve-side report change.

Paged KV, SSD-backed cache, quantized KV cache, and in-flight cold prefill
deduplication remain separate Phase 9 cache-backend work.

## Remaining Risks

The report can only correlate server events by response/request id when the
protocol response id matches the emitted generation id. If a future protocol
changes that id relationship, the report falls back to the phase-local model
event slice, which is still useful but less precise for concurrent cold A/B
requests.

Interactive `cmux` Pi smokes remain useful for client UI and prompt-shape
debugging, but the in-process regression remains the canonical server gate.
