# Agent Cache Concurrency Proof

## Summary

This tranche extends `bun run regression:agent-cache` with explicit
`--max-concurrent-requests` and `--prompt-prefix-cache-max-entries` options.
The harness now records both values in the JSON report and compact stdout, so
Pi-style cache evidence states whether the probe used the default serialized
serving lane or a two-active-request serving shape.

The serving cache behavior is unchanged. This is a regression harness and
documentation hardening tranche for the reported two-agent Gemma MoE cache miss
class.

## Files Reviewed

- `packages/serve/scripts/regression-agent-cache.ts`
- `packages/serve/scripts/regression-agent-cache.test.ts`
- `packages/serve/README.md`
- `.agents/skills/serve-cache-qa/SKILL.md`
- `docs/serving-runtime-strategy.md`

## Tensor Lifetime Audit

No production tensor or cache implementation changed. The command still loads
real checkpoints through existing package APIs, serves them in-process, stops
the server in `finally`, and clears the MLX memory cache between scenarios.

The new options only configure serving concurrency and retained prompt-boundary
capacity for the existing probe. Cache snapshot ownership, fork semantics,
continuous scheduling, and tensor disposal stay in the existing runtime code.

## Memory / Performance Evidence

- `bun test packages/serve/scripts/regression-agent-cache.test.ts` passed.
- `bun run check:skills` passed, including the malformed skill-description
  regression test.
- `bun run --filter '@mlxts/serve' typecheck` passed.
- `bun run regression:agent-cache -- --help` emitted compact AXI stdout with
  the new options.
- `bun run regression:agent-cache -- --max-concurrent-requests 0` exited `2`
  before the runtime lock with structured stdout.
- `bun run regression:agent-cache -- --scenarios gemma-moe --max-concurrent-requests 2 --report-json .tmp/agent-cache-regression/gemma-moe-concurrent.json`
  passed: warm hits `2`, warm server read tokens `618`, warm client cached
  tokens `309`, exact replay hits `1`, exact replay client cached tokens `167`.
- `bun run regression:agent-cache -- --include-moe --include-moe-multi --report-json .tmp/agent-cache-regression/full-concurrency-knob.json`
  passed all six scenarios:
  - `qwen-dense`: warm hits `2`, server read tokens `608`, client cached tokens
    `304`, exact replay client cached tokens `164`.
  - `gemma-dense`: warm hits `2`, server read tokens `602`, client cached tokens
    `301`, exact replay client cached tokens `163`.
  - `multi-dense`: warm hits `4`, server read tokens `1210`, client cached
    tokens `605`, exact replay client cached tokens `327`.
  - `qwen-moe`: warm hits `2`, server read tokens `608`, client cached tokens
    `304`, exact replay client cached tokens `164`.
  - `gemma-moe`: warm hits `2`, server read tokens `618`, client cached tokens
    `309`, exact replay client cached tokens `167`.
  - `multi-moe`: warm hits `4`, server read tokens `1226`, client cached tokens
    `613`, exact replay client cached tokens `331`.

## Independent Review

Locke reviewed the current serving cache QA posture and recommended a harness
hardening tranche rather than a serving behavior change. The review identified
the gap that the existing A/B/A probe used `maxConcurrentRequests: 1`, while
the user-reported incident involved two active Gemma MoE agents. The
recommendation was to add an explicit concurrency knob, report the tested
shape, and document the Gemma MoE concurrent proof command.

## Remaining Risks / Follow-ups

- Cold concurrent A/B requests can both miss when no completed retained
  snapshot exists yet. The required product behavior is warm A/B replay plus
  exact A replay hitting retained prompt boundaries.
- The in-process harness proves server cache behavior and OpenAI-compatible
  usage accounting. `cmux` Pi terminal smokes remain useful for client UI,
  model metadata, and workspace-specific prompt-shape debugging.
- Exact-boundary Gemma and Qwen caches still do not claim arbitrary shared
  `AGENTS.md` prefix reuse. They require a retained exact boundary or a future
  family-owned cache backend that supports shorter forks.
