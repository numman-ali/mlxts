# Serve Agent Cache QA Rerun

## Summary

Reran the Pi-style prompt-prefix cache regression after the reported two-agent
Gemma MoE cache miss. The current tree does not reproduce a warm-cache
regression: dense Qwen, dense Gemma, same-server dense Qwen+Gemma, Qwen MoE,
Gemma MoE, and same-server Qwen MoE+Gemma MoE all retain warm prompt-boundary
snapshots with `max_concurrent_requests: 2`.

The important distinction remains: two cold concurrent requests can both miss
because no completed snapshot exists yet. Warm replay and exact A replay must
hit retained prompt boundaries, and the gate proves that behavior.

## Files Reviewed

- `packages/serve/scripts/regression-agent-cache.ts`
- `packages/serve/src/engine/prefix-cache.ts`
- `packages/serve/src/engine/index.ts`
- `packages/serve/README.md`
- `.agents/skills/serve-cache-qa/SKILL.md`

## Evidence

Focused cache tests passed:

```bash
bun test packages/serve/scripts/regression-agent-cache.test.ts packages/serve/src/engine/prefix-cache.test.ts packages/serve/src/engine/engine.test.ts
```

Targeted Gemma MoE two-active-agent proof passed:

```bash
bun run regression:agent-cache -- --scenarios gemma-moe --max-concurrent-requests 2 --report-json .tmp/agent-cache-regression/gemma-moe-concurrent-rerun.json
```

Result: warm hits `2`, warm server read tokens `618`, warm client cached tokens
`309`, exact replay hits `1`, exact replay client cached tokens `167`.

Full concurrent matrix passed:

```bash
bun run regression:agent-cache -- --scenarios qwen-dense,gemma-dense,multi-dense,qwen-moe,gemma-moe,multi-moe --max-concurrent-requests 2 --report-json .tmp/agent-cache-regression/full-concurrent-rerun.json
```

Results:

- `qwen-dense`: warm hits `2`, server read tokens `608`, client cached tokens
  `304`, exact replay client cached tokens `164`.
- `gemma-dense`: warm hits `2`, server read tokens `602`, client cached tokens
  `301`, exact replay client cached tokens `163`.
- `multi-dense`: warm hits `4`, server read tokens `1210`, client cached tokens
  `605`, exact replay client cached tokens `327`.
- `qwen-moe`: warm hits `2`, server read tokens `608`, client cached tokens
  `304`, exact replay client cached tokens `164`.
- `gemma-moe`: warm hits `2`, server read tokens `618`, client cached tokens
  `309`, exact replay client cached tokens `167`.
- `multi-moe`: warm hits `4`, server read tokens `1226`, client cached tokens
  `613`, exact replay client cached tokens `331`.

## Diagnosis

No serving cache code change is indicated by this rerun. The likely operational
confusion is cold-concurrent behavior: if two agents start from no retained
prompt-boundary snapshot, both initial requests miss. Once their cold requests
complete, the configured four retained prompt-boundary snapshots per served
model keep divergent A/B sessions warm, including Gemma layer-pattern and Qwen
hybrid exact-boundary caches.

## Remaining Risks / Follow-ups

- Exact-boundary Gemma and Qwen caches still do not claim arbitrary shared
  `AGENTS.md` prefix reuse. That requires a retained exact boundary or a future
  family-owned backend that supports shorter forks.
- Interactive `cmux` Pi terminal smokes remain useful for client UI and
  workspace prompt-shape debugging, but the in-process regression is the
  canonical product gate for server cache behavior.
