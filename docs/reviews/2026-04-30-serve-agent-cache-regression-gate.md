# Serve Agent Cache Regression Gate

## Summary

This tranche turns the Pi-style prompt-prefix cache incident into a repeatable
serve regression command. `bun run regression:agent-cache` drives divergent
A/B chat sessions cold, replays both warm, and fails unless both warm replays
produce server prompt-cache hits plus OpenAI-compatible cached-token usage.

The default scenarios cover dense Qwen, dense Gemma, and same-server dense
Qwen+Gemma. `--include-moe` adds the proven Qwen/Gemma MoE single-model
scenarios, and `--include-moe-multi` adds same-server Qwen MoE + Gemma MoE.

## Files Reviewed

- `packages/serve/scripts/regression-agent-cache.ts`
- `packages/serve/scripts/regression-agent-cache.test.ts`
- `packages/serve/scripts/regression-serve-matrix.ts`
- `package.json`
- `packages/serve/package.json`
- `packages/serve/README.md`
- `.agents/skills/serve-cache-qa/SKILL.md`
- `AGENTS.md`
- `PLAN.md`
- `continuity.md`
- `MEMORY.md`

## Tensor Lifetime Audit

The new command is a regression harness over existing serving APIs. It does not
introduce tensor-producing primitives or model/runtime cache implementations.
Real checkpoint loads are disposed through `serveLoadedModels({ disposeModelsOnStop: true })`.
The command loads tokenizer/profile metadata before each model so local metadata
failures cannot strand a partially loaded checkpoint. If a multi-model scenario
fails before server startup, already-loaded models are disposed explicitly
before the error is rethrown. Multi-model scenarios load sequentially to avoid
transient memory spikes.

## Memory / Performance Evidence

- `python3 /Users/numman/.codex/skills/.system/skill-creator/scripts/quick_validate.py .agents/skills/serve-cache-qa` passed.
- `bun test packages/serve/scripts/regression-agent-cache.test.ts packages/serve/scripts/regression-serve-matrix.test.ts` passed (`16` tests).
- `bun run --filter '@mlxts/serve' typecheck` passed.
- `bun run regression:agent-cache -- --scenarios qwen-dense --report-json .tmp/agent-cache-regression/qwen-dense.json` passed: warm hits `2`, server read tokens `608`, client cached tokens `304`.
- `bun run regression:agent-cache -- --report-json .tmp/agent-cache-regression/dense.json` passed: dense Qwen, dense Gemma, and same-server dense Qwen+Gemma recorded warm hits `2/2/4`, server read tokens `608/602/1210`, and client cached tokens `304/301/605`.
- `bun run regression:agent-cache -- --scenarios qwen-moe,gemma-moe --report-json .tmp/agent-cache-regression/moe.json` passed: Qwen MoE and Gemma MoE recorded warm hits `2/2`, server read tokens `608/618`, and client cached tokens `304/309`.
- `bun run regression:agent-cache -- --scenarios multi-moe --report-json .tmp/agent-cache-regression/multi-moe.json` passed: same-server Qwen MoE + Gemma MoE recorded warm hits `4`, server read tokens `1226`, and client cached tokens `613`.

## Independent Review

Bernoulli reviewed the post-cache roadmap and recommended making the next
tranche a serve-cache product gate before starting larger cache backends or
Phase 10 implementation work. The review specifically called out dense,
targeted MoE, mixed Gemma+Qwen multi-model serving, and Pi-style A/B/A replay
evidence as the highest-value next move.

Huygens performed a blocker-only pre-commit review and identified partial-load
model disposal as the serious issue to fix before commit. The harness now loads
lightweight metadata before the model, removing the partial `Promise.all` leak
risk.

## Remaining Risks / Follow-ups

- This command proves warm retained prompt-boundary reuse. It does not claim
  cold concurrent requests can reuse an in-flight snapshot.
- Exact-boundary Qwen hybrid and Gemma layer-pattern caches still require an
  exact retained snapshot at the requested boundary. Arbitrary AGENTS-prefix
  LCP reuse remains future cache-backend work.
- The command starts in-process serving harnesses. Real interactive Pi terminal
  smokes through `cmux` remain useful when validating client-specific UI or
  model metadata behavior.
