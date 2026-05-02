# Architectural Posture Audits

This folder holds periodic holistic repo audits. A posture audit is not a
feature review and not a remediation PR. It is the repo stopping for a moment
to ask: "Is the system becoming more coherent, or are we just accumulating
working pieces?"

Treat this file as the reusable audit brief.

Keep the audit prompt outcome-first. Name the desired judgment, evidence
requirements, severity vocabulary, output shape, and stop rules. Do not grow a
process-heavy checklist unless every item materially changes audit quality.

## Audit Mission

Read the repo as a product, an architecture, and an agent-operated system.

The audit should identify structural drift, product-surface confusion, proof
gaps, governance decay, and missing risk lanes. It should be blunt enough to
change the next tranche, but concrete enough that the next tranche is not a
grab bag.

The most important rule: do not confuse "code exists" with "the capability is
owned, documented, validated, and usable end to end."

A proof is evidence, not completion. Audits should call out where the repo has
proved a narrow slice but has not finished the full capability: all intended
variants, product surfaces, artifacts, docs, validation, operator evidence, and
failure behavior.

## What To Produce

Each audit produces two artifacts:

1. `<date>-audit-metrics.md` - mechanical baseline and reproducible counts.
2. `<date>-architectural-posture-audit.md` - synthesized judgment,
   recommendations, direction, and open decisions.

If replacing a flawed same-day audit, move the older files into
`docs/audits/archive/` with `-superseded` in the filename and keep the canonical
date path for the fresh audit.

## Required Reading

Read in this order:

1. `AGENTS.md` and `MEMORY.md` Tier 1.
2. The previous audit and its metrics.
3. `PLAN.md`, `continuity.md`, and the relevant product docs.
4. The cited source files, examples, package docs, and review artifacts for the
   claims you plan to make.

For product-surface or CLI claims, also read `.agents/skills/axi/SKILL.md`.
If the local AXI skill is meant to mirror upstream, verify it against upstream
before using it as an authority. Distinguish "the upstream standard says this"
from "mlxts has chosen to adopt this for its product surface."

Use prior audits as evidence, not as truth. A prior claim can be wrong,
overweighted, stale, or too kind.

## Prompt And Tool Budget

Run audits from an outcome-first prompt: define the posture question, success
criteria, required evidence, risk lanes, and stopping condition. Avoid adding a
fixed shell-command itinerary unless a specific command is part of the evidence
contract.

Use a retrieval budget. After the prior audit, current roadmap, active continuity,
representative source paths, and relevant review artifacts are enough to
support the core findings, synthesize instead of continuing to read for color.
Read more only when a material claim is missing a path, owner, date, gate,
benchmark, source artifact, or counterexample check.

If an audit depends on live external documentation or an upstream product
contract, cite the current source and distinguish what transfers to mlxts from
what remains merely comparative context.

For very long audit sessions, compact deliberately: preserve completed reads,
accepted assumptions, file paths, commands/results, unresolved blockers, and
the next concrete synthesis goal. Do not compact into vague "reviewed docs"
summaries that erase provenance.

## Severity Vocabulary

Use P-severity labels instead of colored symbols.

- **P0 - Stop-ship**: the repo is making a false capability claim, has a likely
  correctness/safety issue, or has a release-blocking legal/security problem.
- **P1 - Structural / product**: material repo-health or product-coherence issue
  that should change phase order or become the next design/remediation tranche.
- **P2 - Governance / proof / hygiene**: important during the active phase, but
  not a reason to stop unrelated work immediately.
- **P3 - Cheap cleanup**: low-risk cleanup to bundle with nearby work.
- **P4 - Direction**: proposed phase plan or design question, not a commitment.

Severity means "how much pain remains if this stays as-is." It does not mean
"how easy is this to fix" or "how exciting is this work."

## Audit Structure

Every audit doc uses these sections, in order:

1. **Executive Read** - direct posture read and the few concerns that matter
   most.
2. **Intent Check** - what the human/product intent appears to be, and whether
   the repo is currently serving that intent.
3. **Drift Ledger** - prior findings marked resolved, persisting, migrated,
   reframed, or new.
4. **Mechanical Snapshot Summary** - headline numbers; detailed counts live in
   the metrics doc.
5. **Product Surface Coherence** - CLI, API, SDK, examples/workbooks, serving,
   and agent-native operation.
6. **Architectural Posture** - package/folder boundaries, structural pressure,
   cross-package and cross-example edges.
7. **Quality And Proof Bar** - Add vs Prove, real-checkpoint evidence, gates,
   and end-to-end validation.
8. **Risk Lanes** - security, dependency hygiene, native binary distribution,
   release readiness, licensing, and supply chain. This can be short, but it
   must not disappear.
9. **Direction** - proposed phase plan and sequencing. Separate "start design"
   from "start implementation."
10. **Specific Recommendations** - severity ordered, concrete paths, clear next
    owner, and intended destination.
11. **Open Questions For Nomi** - decisions the audit cannot make, or
    recommended answers that need Nomi's confirmation.
12. **Files Reviewed** - provenance.

Format changes to this structure are themselves audit findings.

## Drift Rules

For every P0, P1, and P2 item from the prior audit, classify it:

- **Resolved**: fixed end to end, with evidence. Do not mark resolved if the
  same pressure moved to a new path.
- **Persisting**: same issue remains in substantially the same place.
- **Migrated**: the original instance was fixed, but the same structural pattern
  reappeared elsewhere.
- **Reframed**: the previous wording was not quite right; name the better
  version.
- **New**: genuinely new pressure since the prior audit.

Be suspicious of "all red items resolved" statements. They are often true only
at the literal file-path level.

## Product-Surface Rules

Examples prove and teach. They do not own durable product contracts.

Flag any reusable behavior trapped in `examples/`, especially:

- artifact writers and verifiers
- prompt-conditioning composition that every family repeats
- flag readers, TOON formatters, structured errors, and report schemas
- model discovery, download/cache policy, and proof manifests
- serving or generation workflows that users are likely to run directly

For mlxts, product-surface coherence means:

- an agent can run the top-level CLI and discover what is possible
- generation, serving, training, model discovery, and verification are named
  consistently
- API/SDK/CLI surfaces share concepts without forcing one surface to wrap
  another awkwardly
- examples remain thin workbooks over package-owned capabilities

AI SDK, OpenAI APIs, Diffusers, mlx-lm, and other references are comparison
anchors, not authorities to cargo-cult. Say exactly which part transfers and
which part does not.

## Proof Rules

Use these terms consistently:

- **Add**: code path, skeleton, loader, parser, or finite synthetic proof exists.
- **Prove**: real checkpoint or realistic end-to-end workflow has run and has a
  review artifact with evidence.
- **Product-ready**: proof plus ergonomic product surface, docs, limits,
  structured errors, and operator evidence.
- **Capability-complete**: product-ready across the intended variants and
  surfaces, with no known missing core mode hidden behind a proof artifact.

Never write "supports X" when the truthful claim is "has parser/runtime pieces
for X" or "has a finite proof but no real checkpoint evidence."

When an audit says a capability is complete, it must name the close rule:
validated product surface, accepted unsupported modes, evidence location, and
where any remaining gaps are tracked.

## Process

The default audit process is:

1. Mechanical baseline pre-pass: gates status, package/file counts, CLI bins,
   AGENTS coverage, cross-package/cross-example edges, file-cap pressure.
2. Parallel read-only slice agents:
   - A: `core` / `nn` / `optimizers` / `quantize`
   - B: `train` / `data` / `tokenizers` / `lora` / `align`
   - C: `transformers`
   - D: `diffusion`
   - E: `serve` / `agent` / `protocols`
   - F: `examples` / `docs` / governance / AXI / product surface
3. Lead synthesis: challenge slice claims, spot-check representative evidence,
   write the audit and metrics artifacts.

When the context is tight, use fewer slice agents, but keep at least one
independent second opinion for non-trivial audits.

Use a retrieval budget. Read the required docs, then spot-check enough cited
source and review evidence to make the claim honestly. Continue reading when a
required fact, source, date, owner, or validation result is missing; stop when
the core posture read is supported and more reading would only add decoration.

## After The Audit

An audit does not become a second roadmap. Route accepted outcomes into the
right durable home:

- `PLAN.md`: phase order, mission goals, tranche sequencing, and high-level
  product/architecture direction.
- `continuity.md`: current active state only - what is in flight, what was just
  decided, what command or file continuing work should pick up, and what is
  blocked.
- `MEMORY.md`: durable sharp edges, repo-specific lessons, and cross-session
  rules that remain true beyond the current tranche.
- `AGENTS.md` or package-local `AGENTS.md`: operating doctrine that ongoing
  work must follow.
- `docs/reviews/`: evidence that a completed tranche was validated.
- Product and architecture docs: stable public contracts, design decisions, and
  user-facing product shape.

Do not copy an audit finding into every document. Move the truth to the place
where later work will naturally look for it.

For long-running sessions, do a documentation hygiene pass at every major pause
or completed tranche:

- update `continuity.md` with a compact state summary and delete stale details
- add or update review artifacts for evidence-heavy work
- promote durable learnings into `MEMORY.md`, but keep Tier 1 small
- update `PLAN.md` only when the mission, phase order, or accepted roadmap
  changed
- leave `docs/audits/` as the dated diagnosis, not a living task list

## Risk Lanes

Every holistic audit should at least scan for:

- security and sandbox boundaries
- dependency freshness and transitive risk
- native binary distribution and build reproducibility
- license compatibility for runtime deps, model checkpoints, and `.reference/`
  research inputs
- release-readiness gaps if the repo is public or near publish
- supply-chain assumptions around downloads, caches, and generated artifacts

These lanes may not be the main product concern in a given audit, but omitting
them entirely makes the audit less holistic.

## What An Audit Is Not

- Not a code review for individual changes. Use `docs/reviews/`.
- Not a remediation PR.
- Not a feature roadmap. Direction proposals are for Nomi to approve or change.
- Not a pass/fail scorecard.
- Not a place to reward busyness.

The audit earns its keep by making the next engineering decision clearer.
