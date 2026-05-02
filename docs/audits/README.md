# Architectural Posture Audits

This folder holds periodic holistic repo audits — a distinct artifact type from the per-tranche reviews in [`docs/reviews/`](../reviews/).

## What this folder is

A **posture audit** is a read-only diagnosis of the repo's overall health at a moment in time. It covers:

- **Repo posture**: governance gates, AGENTS.md coverage, doc-layer hygiene, sub-agent workflow health
- **Architectural posture**: folder structure, line-cap pressure, cross-package edges, structural drift since the prior audit
- **Product surface coherence**: CLI / API / SDK alignment, paradigm splits, AXI compliance across consumer-facing surfaces
- **Quality bar**: `Add` vs `Prove` evidence stages, real-checkpoint gaps, completeness against industry-standard
- **Direction**: proposed phase plans for user review (not committed work)
- **The audit format itself**: this practice is iterated on, and the format evolves deliberately

A **per-tranche review** in `docs/reviews/` is a single feature's evidence artifact: what changed, what was tested, performance evidence, runtime safety review. Per-tranche reviews live next to features; posture audits live above them.

The two artifact types compose: per-tranche reviews capture point-in-time changes; posture audits identify drift and structural pressure across many tranches.

## Why we do this

Five-day windows of fast feature work accumulate structural pressure that is invisible in per-tranche review. Folder shapes drift. Files cluster near caps. CLIs duplicate boilerplate. Doctrine duplicates across files. Mechanical gates stay green while structure rots.

The posture audit:

- **Detects drift.** Every audit anchors on the previous one and reports finding-by-finding what is *resolved*, *persisting*, or *new pressure*.
- **Compounds cleanup.** Naming the drift makes it actionable. The next audit either finds it cleaner (the mechanism worked) or finds it persisting (signal that the recommendation lacked a mechanical gate behind it).
- **Forecasts structural pressure.** Files in the 450–499 LOC band before any cap violation. The audit catches the pressure before the gate does.
- **Tests product surface coherence.** The audit explicitly asks whether CLI / API / SDK align, whether paradigms have unified surfaces, whether the quality bar is uniform across families, and whether the user-facing story makes sense.
- **Acts as a forcing function on documentation.** When `continuity.md` regrows or `MEMORY.md` Tier 2 bloats, the audit names it. Without the audit, narrative docs drift unchecked.

## Cadence

- **Active phase work** (build-out tranches across multiple packages): every **3–5 days**
- **Stable maintenance**: weekly
- **Major version transitions**: before phase entry, after phase exit

Each audit reads the prior audit's findings before starting. Don't skip iterations — the value is the comparison, not the snapshot.

## Process

Each audit produces two artifacts:

1. `<date>-audit-metrics.md` — mechanical baseline (gates status, package weight, file-cap pressure, cross-package edges, AGENTS.md coverage, drift summary)
2. `<date>-architectural-posture-audit.md` — synthesized findings, recommendations, and direction proposals

Workflow:

1. **Mechanical baseline pre-pass** — single Explore agent runs gates + counters. Output: `<date>-audit-metrics.md`.
2. **Six parallel slice agents** (Opus, read-only, ~30–50 minutes each):
   - **Slice A**: `core` / `nn` / `optimizers` / `quantize`
   - **Slice B**: `train` / `data` / `tokenizers` / `lora` / `align`
   - **Slice C**: `transformers`
   - **Slice D**: `diffusion`
   - **Slice E**: `serve` / `agent` / `protocols`
   - **Slice F**: `examples` / `docs` / governance / AXI cross-cutting
3. **Synthesis** by lead agent — produces the audit doc with the stable section structure below.

Slice ownership decisions live in the audit doc's "Audit-as-Practice" section, not in chat. When a new package emerges (e.g., `@mlxts/sdk`, `@mlxts/evals`), slice ownership is decided in the audit and inherited by the next iteration.

## Stable section structure

Every audit doc has these sections, in order:

1. **Executive Read** — 2–3 paragraphs, severity-marked top concerns
2. **Mechanical Snapshot Summary** — single-paragraph headline; full numbers in companion metrics doc
3. **Repo Posture** — governance, gates, sub-agent workflow, doctrinal duplication
4. **Architectural Posture** — per-slice findings, drift since prior audit
5. **Product Surface Coherence** — CLI / API / SDK alignment, paradigm splits, AXI
6. **Direction** — proposed phase plans (not commitments)
7. **Live design questions** — current open architectural decisions
8. **Audit-as-Practice** — process meta, format evolution
9. **Specific Recommendations** — severity-ordered (🔴 → 🟡 → 🟢), concrete file paths
10. **Open Questions for the User** — decisions the audit cannot make
11. **Files Reviewed** — provenance

Format changes to this structure are noted in §8 of the next audit.

## What an audit is *not*

- Not code review for individual changes (use per-tranche reviews in `docs/reviews/`)
- Not a remediation PR (read-only diagnosis)
- Not a feature roadmap (direction proposals are user-reviewable, not committed)
- Not a substitute for per-tranche reviews (they cover what audits do not — feature-level evidence)
- Not a snapshot — every audit is comparative, anchored on the previous one

## Severity vocabulary

Findings and recommendations are severity-marked:

- 🔴 **Structural** — material to repo health; address out of phase order if needed
- 🟡 **Governance / hygiene** — address during the active phase
- 🟢 **Cheap hygiene** — bundle alongside structural fixes
- 🔵 **Direction** — phase-plan proposal, not committed

Severity is "how much pain does this cause if it stays as-is." Phase ordering is a separate question handled at decision time.

## Anchoring

Each new audit reads the prior audit's §9 (recommendations) and §1 (executive read) before starting. The chain forms a posture history:

- `2026-04-28-architectural-posture-audit.md` — first iteration, baseline
- `2026-05-02-architectural-posture-audit.md` — second iteration, drift report

A recommendation that persists across two iterations without resolution is escalated in the third. A recommendation that resolves between iterations validates the audit-as-practice mechanism.

## Invocation

Currently user-triggered. To run an audit, ask the lead agent for "a posture audit" or "the next architectural review." The agent fans out the six slices + metrics agent in parallel, then synthesizes.

Future options:

- Schedulable via `/loop` (every N days during active phase)
- Dedicated `audit-run` skill with the slice prompts encoded
- Hook on phase-exit gates so audits anchor phase transitions

## Cross-folder relationships

- [`docs/reviews/`](../reviews/) — per-tranche reviews. Audits reference these as evidence for `Add` vs `Prove` claims.
- [`MEMORY.md`](../../MEMORY.md) Tier 1 — durable repo-wide learnings; audits update Tier 1 only when something becomes a permanent fact.
- [`continuity.md`](../../continuity.md) — current-phase handoff state; audits flag bloat or doctrinal duplication here.
- [`AGENTS.md`](../../AGENTS.md) — repo doctrine; audits propose doctrine evolution but do not edit it directly.
- [`PLAN.md`](../../PLAN.md) — phase status; audits propose direction in §6 that may flow into PLAN.md if user-approved.
