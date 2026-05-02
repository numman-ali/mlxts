# Continuity

Compact continuity record for long-running `mlxts` work. Durable doctrine lives in
`AGENTS.md`, durable learnings in `MEMORY.md`, roadmap decisions in `PLAN.md`,
and evidence in `docs/reviews/`.

Last compacted: 2026-05-02.

## Current Focus

The immediate priority is to restore native runtime health, then finish Phase
10b as a complete generation product foundation rather than another set of
family proofs.

Current blocker:

- `bun test packages/core/src/ffi.test.ts` fails after dylib load/empty-array
  smoke because scalar/data arrays and default GPU stream creation return null
  pointers.
- `bun run check:phase10-proofs` fails with the same null array/stream pointer
  signatures across image, video, and audio proof tests.

Do not make new runtime/proof claims until this blocker is isolated.

## Accepted Direction

1. Restore native runtime proof gates.
2. Extract media artifact writers, report schemas, verifiers, BMP/WAV helpers,
   and proof metadata out of examples and into package-owned generation
   surfaces.
3. Split classic LTX-Video and LTX-2 into separate family folders with a small
   shared home, then run documented real-checkpoint proofs where access permits.
4. Design the generation product contract before implementing a top-level
   wrapper: request/response types, artifact descriptors, capability metadata,
   model-source policy, proof metadata, and unsupported-mode errors.
5. Design the future `mlxts` CLI as the agent-native entrypoint. `mlxts` with
   no arguments should show a compact live dashboard. Session hooks are not a
   v1 requirement.
6. Adapt serving after the generation contract exists. `@mlxts/serve` should
   likely own HTTP routes and resource policy, not family artifact writers or
   proof schemas.

## Active Capability Rule

For current Phase 10b work, do not let image-only proof coverage stand in for
the full generation product. Video and audio paths, artifact verification,
CLI/API shape, checkpoint access/licensing metadata, and unsupported-mode
behavior all matter.

## Documentation Routing

- Update `PLAN.md` only for accepted roadmap, phase-order, or mission changes.
- Keep `continuity.md` to active continuity state, blockers, next commands, and
  compact decisions.
- Put recurring sharp edges in `MEMORY.md`, as lookup pointers rather than
  evidence blocks.
- Put validation details and independent review in `docs/reviews/`.
- Keep audit findings in `docs/audits/`; route accepted outcomes to the doc
  where continuing work will naturally look.

Do not append large evidence ladders here. Link the review/audit artifact.

## Current Evidence Pointers

- Fresh posture audit:
  `docs/audits/2026-05-02-architectural-posture-audit.md`
- Fresh metrics and gate results:
  `docs/audits/2026-05-02-audit-metrics.md`
- Audit practice prompt:
  `docs/audits/README.md`
- Serving/Qwen evidence ladder:
  `docs/reviews/2026-04-24-qwen-serve-benchmark-ladder.md`
- LTX proof state:
  `docs/reviews/2026-05-01-ltx-video-proof-cli.md`,
  `docs/reviews/2026-05-02-ltx2-proof-assembly.md`,
  `docs/reviews/2026-05-02-ltx-video-proof-verifier.md`

## Next Useful Commands

Start with the blocker:

```bash
bun test packages/core/src/ffi.test.ts
bun run check:phase10-proofs
```

After native runtime is healthy, rerun the focused doc/governance checks touched
by the audit/doc updates:

```bash
bun run typecheck
bun run check:file-lines
bun run check:assertions
bun run check:tensor-lifetimes
bun run check:runtime-review
bun run check:per-package-agents
bun run check:cross-package-imports
bun run check:skills
bun run lint
```
