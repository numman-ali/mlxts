# Runtime Review: Local Model Discovery

## Summary

`mlxts-serve --model-root` now discovers supported local autoregressive checkpoint directories and expands them into source-backed model entries before startup. Model-root serving defaults to lazy loading and routes through the existing `serveModels()` lazy pool, so startup can advertise a local model store without loading every checkpoint.

## Files Reviewed

- `packages/serve/src/model-loading/discovery.ts`
- `packages/serve/src/cli-model-options.ts`
- `packages/serve/src/cli-options.ts`
- `packages/serve/src/cli.ts`
- `packages/serve/src/cli-usage.ts`
- `packages/serve/src/index.ts`

## Runtime Sensitivity

The tranche changes source-backed serving setup, not token generation. Local model roots expand to explicit source entries before the server starts, and the existing lazy `serveModels()` pool remains the only loading path for discovered checkpoints.

Discovery stays local and shallow by design: the scanner accepts the root checkpoint, direct child checkpoints, and two-level `org/model` checkpoints only when a directory contains `config.json`, a model type supported by the transformer family registry, and safetensor weights. It records config metadata for logs/API use later, but it does not infer runtime strategy or widen model loading.

## Tensor Lifetime Audit

No MLX tensors or native handles are created. The scanner performs host filesystem reads only. The CLI expands model roots before model resolution, memory preflight, checkpoint loading, and engine construction.

## Memory / Performance Evidence

- `bun test packages/serve/src/model-loading/discovery.test.ts packages/serve/src/cli.test.ts` — `19 pass`
- `bun run --filter '@mlxts/serve' typecheck` — pass
- `bun run --filter '@mlxts/serve' test` — `396 pass`
- `bun run validate` — pass

## Independent Review

Avicenna recommended this as the next Phase 9d-sized product tranche after lazy loading: local-only model discovery, root plus `org/model` layout, explicit lazy loading, no broad config auto-dispatch, and no active-request abort policy in this commit. A follow-up pass caught that raw `config.json` plus safetensor detection would also advertise non-LM component directories; this tranche now filters discovery through the supported autoregressive model-type registry.

Cicero reviewed the final uncommitted diff and reported no high- or medium-severity findings. The review called out two low-severity tightenings before commit: add negative coverage for post-discovery duplicate model ids and missing pinned ids, and make this review artifact record the completed review outcome. Both are addressed in this tranche.

Carver reviewed the AXI CLI guidance requested during this tranche and recommended storing the skill at `.agents/skills/axi/SKILL.md`, updating root/product doctrine, and leaving a full serve CLI AXI rewrite as a separate focused migration instead of folding it into model-root discovery.

## Remaining Risks / Follow-ups

- Discovery does not parse Hugging Face hub cache internals such as `models--org--name/snapshots/<sha>`; users should point `--model-root` at local checkpoint folders or an `org/model` directory tree.
- Discovered config metadata is not yet surfaced in `/info` or startup logs.
- The existing `mlxts-serve` startup/help/error stream is not yet fully AXI-shaped; the repo now carries the AXI skill and doctrine, and the serve CLI needs a dedicated migration tranche for stdout/stderr and finite status/error contracts.
- Active-request abort policy for loaded models under KV pressure remains separate memory-management work.

## Out-of-scope Drift Noticed

- `continuity.md` still described lazy loading, idle TTL, and pinning as future memory work before this session's lazy-pool commits. This tranche updates that state rather than changing serving behavior outside model-root discovery.
