# Repo Memory

This file captures durable cross-session learnings for `mlxts` so future agent sessions do not need to rediscover the same sharp edges.

## How To Use This File

### Tier 1 — Must Read

- Read this section at the start of every session, immediately after `AGENTS.md`.
- Keep it short and durable. Promote only facts that should shape most future sessions.
- If Tier 1 grows, demote older or narrower items into Tier 2.

### Tier 2 — Lookup Log

- Do not read this section end to end by default.
- Search it when working on a related area.
- Append durable learnings here at the end of a session.

## Memory Update Rule

- Add durable learnings to Tier 2 using this template:
  - `(YYYY-MM-DD) [TAG] <learning> — refs: <file/doc/command>`
- Promote only cross-session critical items to Tier 1.
- Keep Tier 1 focused on durable repo facts and recurring sharp edges, not doctrine already stated in `AGENTS.md`.
- Archive obsolete items instead of silently deleting them.
- Do not put transient task notes or diary-style updates here.

## Tier 1 — Must Read

- `packages/nanogpt/` is a temporary validation fixture; prefer improving canonical `@mlxts/*` packages over deepening new permanent surfaces there.
- Heavy MLX commands are exclusive on one machine. Do not run benchmarks, soak runs, acceptance runs, or long training/proof commands in parallel.
- Runtime-sensitive production changes require a review artifact under `docs/reviews/`, plus `bun run check:runtime-review` before handoff.
- `bun run typecheck` and `bun run check:coverage` are required quality gates, not optional cleanup.
- Before writing a JS fallback for a GPU-facing operation, check whether `mlx-c` already exposes the op in `packages/core/native/build/_deps/mlx-c-src/mlx/c/ops.h`.
- Keep runtime strategy out of public model identity and semantic names. Do not widen `CausalLM` or create duplicate model configs just to represent cache/backend/runtime variants.

## Tier 2 — Lookup Log

- (2026-04-22) [PRODUCT] `packages/nanogpt/` remains a temporary validation fixture during package extraction; permanent contracts should land in reusable package-owned surfaces instead. — refs: `AGENTS.md`
- (2026-04-22) [OPS] Heavy MLX commands are exclusive on a shared machine; do not run benchmark, soak, acceptance, memory, or long-run training commands in parallel. — refs: `AGENTS.md`
- (2026-04-22) [QUALITY] Runtime-sensitive diffs need a paired review artifact under `docs/reviews/`, and the artifact's `Files Reviewed` section must name the exact changed runtime-sensitive files. — refs: `AGENTS.md`, `docs/runtime-safety.md`
- (2026-04-22) [ARCH] Model contracts describe behavior, not runtime strategy; MoE stays a block-level swap inside `CausalLM`, and runtime/backend differences should not fork model identity. — refs: `AGENTS.md`, `docs/design-reasoning.md`
- (2026-04-22) [MLX-C] For GPU-facing functionality, check `mlx-c` first before introducing JS workarounds; fallback JS should stay limited to genuinely host-side work. — refs: `AGENTS.md`, `packages/core/native/build/_deps/mlx-c-src/mlx/c/ops.h`
