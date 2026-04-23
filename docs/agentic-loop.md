# Agentic Engineering Loop

## Overview

This project is built using multiple AI coding agents in a structured, iterative loop. No single agent operates unchecked — every output is reviewed by a different agent or human before it ships.

The exact model or tool used for each role may change over time. The workflow matters more than the brand name attached to any individual step.

This document defines the process.

Runtime-sensitive code gets stricter handling than ordinary feature work. If a change touches production code in `packages/core/src/`, `packages/nn/src/`, `packages/optimizers/src/`, `packages/train/src/`, `packages/data/src/`, `packages/tokenizers/src/`, or the committed nanoGPT example in `examples/nanogpt/src/`, the change is not review-ready until it has:

- a line-by-line runtime audit
- an independent review by a different agent or human
- a review artifact under `docs/reviews/`
- the exact changed runtime-sensitive files listed in the artifact's `Files Reviewed` section
- any new memory/performance evidence that the change needs

## The Loop

```
                    ┌──────────────────────┐
                    │    1. SPEC / PLAN     │
                    │  (Human + Planner)    │
                    │                       │
                    │  Define what to build │
                    │  Design the approach  │
                    │  Write acceptance     │
                    │  criteria             │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    2. IMPLEMENT       │
                    │  (Implementation)     │
                    │                       │
                    │  Write code against   │
                    │  the spec             │
                    │  Write tests          │
                    │  Self-validate        │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    3. VALIDATE        │
                    │  (Automated)          │
                    │                       │
                    │  Type check first     │
                    │  Run coverage-backed  │
                    │  tests                │
                    │  Lint                 │
                    │  Build when relevant  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    4. REVIEW          │
                    │  (Different agent)    │
                    │                       │
                    │  Code review          │
                    │  Architecture check   │
                    │  Security review      │
                    │  API design review    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    5. ACCEPT / REJECT │
                    │  (Human)              │
                    │                       │
                    │  Final decision       │
                    │  Merge or iterate     │
                    └──────────┬───────────┘
                               │
                    ┌──────────┴───────────┐
                    │                       │
                    ▼                       ▼
              ┌──────────┐          ┌──────────────┐
              │  MERGE   │          │  ITERATE     │
              │          │          │  (back to 1  │
              │  Ship it │          │   or 2)      │
              └──────────┘          └──────────────┘
```

## Roles

### Human (Nomi) — Decision Maker

- Sets direction and priorities
- Defines acceptance criteria
- Resolves disagreements between agents
- Final approval on all merges
- The only one who can change the plan

### Planning / Architecture Agent

- Primary: planning, architecture, and design decisions
- Secondary: implementation of complex or novel code
- Review: code written by a different agent
- Debugging: diagnose issues that other agents cannot resolve
- Context keeper: maintains coherence across the project

### Implementation Agent

- Primary: bulk implementation, mechanical porting
- Strengths: parallel execution, large file generation, repetitive patterns
- Works best with: clear specs, defined interfaces, known patterns
- Example tasks: "Implement these 50 C wrapper functions", "Port this Python module to TypeScript"

### Independent Reviewer

- Primary: independent review, alternative approaches
- Strengths: a fresh perspective from the implementation author
- Review: architecture decisions, API design
- Research: edge cases, API behavior, platform details

## Rules

### The Four-Eyes Principle

Every piece of code or documentation that enters the project must be reviewed by at least one entity other than its author:

- Agent-written code → reviewed by a different agent or human
- Human-written code → reviewed by an agent
- No exceptions. No "it's just a small fix."

### Agent Boundaries

- **No agent modifies the plan** without human approval
- **No agent merges to main** — only human merges
- **No agent deletes or overwrites** another agent's work without review
- **Agents must document their reasoning** in commit messages and PR descriptions

### Validation Gate

- `bun run typecheck` must pass before work moves from implementation to review
- `bun run validate` is the standard gate: typecheck, lint, assertion checks, tensor-lifetime checks, runtime-review checks, and coverage-backed tests
- `bun run check:tensor-lifetimes` enforces a narrow AST-based static check against the anonymous-intermediate leak class in runtime-sensitive code
- `bun run check:runtime-review` enforces the review artifact requirement for runtime-sensitive diffs and verifies that the artifact names the changed runtime-sensitive files
- `bun run check:coverage` remains truthful about branch data: it enforces branch thresholds only when LCOV reports them, and otherwise says branch data was unavailable
- Pre-commit should call `bun run validate` directly rather than maintaining a weaker parallel checklist
- The canonical `@mlxts/*` package stack must stay at or above `95%` lines and `90%` functions
- `examples/nanogpt` remains tested as a committed example surface, but it is not part of the default hard package coverage gate
- Agents should prefer the smallest meaningful Bun test run while iterating locally, for example `bun test packages/core/src/fast.test.ts`, before moving back to `bun run validate`
- Tests, lint, and build remain required, but type errors are treated as design failures, not cosmetic issues
- Type assertions and `any` do not count as "fixing" a type problem unless they are isolated to a justified boundary such as FFI
- Acceptance runs for `gpt-tiny` and `gpt-small` are scripted separately from `validate`; they are part of phase sign-off, not pre-commit
- Long unattended training is validated through the supervised nanoGPT example flow (for example `cd examples/nanogpt && bun run manager ...`), not ad hoc one-shot scripts or root-level package scripts
- Runtime-sensitive changes must leave an explicit review record in `docs/reviews/`, including tensor-lifetime notes, the exact files reviewed, and remaining risks
- If a serious incident is fixed, the same change must also add a preventive rule, test, benchmark, or validation gate so the lesson becomes part of the repo
- Benchmarks and soak runs stay outside `validate`, but they are still required evidence before long unattended training is considered trustworthy again

### Spec-First Development

Before implementing, the spec must include:

1. **What** — Clear description of the deliverable
2. **Why** — Motivation and context
3. **Interface** — Public API signatures with types
4. **Tests** — At minimum, the test cases (can be written before implementation)
5. **Not included** — Explicit scope boundaries

### Runtime-Sensitive Change Protocol

Use this path for hot-path tensor code, long-run training control, memory behavior, checkpointing, or device/runtime orchestration:

1. Audit the changed file line by line for ownership, disposal, sync/eval points, and hidden operator costs
2. Get an independent reviewer who did not author the implementation
3. Record the review in `docs/reviews/`
4. Add or update the test/benchmark/gate that would have caught the issue earlier
5. Only then treat the change as ready for normal review

### Conflict Resolution

When agents disagree:

1. Each agent states its position with reasoning
2. Human decides
3. Decision is recorded in docs or code comments
4. Losing position is not relitigated unless new information emerges

## Workflow by Phase

### Phase 1 (Core Bindings)

```
Planning agent:       Design C wrapper API + TypeScript public API
                      ↓
Implementation agent: Implement the bindings and public API
                      ↓
Automated validation: typecheck first, then tests/lint/build
                      ↓
Independent review:   Review API design, memory safety, edge cases
                      ↓
Nomi:                 Accept or request changes
```

### Phase 4 (nanoGPT)

```
Planning agent:       Design model architecture + training loop
                      ↓
Implementation agent: Implement model, data pipeline, training script
                      ↓
Automated validation: typecheck first, then tests/lint/build
                      ↓
Independent review:   Review model correctness and suggest improvements
                      ↓
Nomi:                 Accept and evaluate generated text quality
```

## Anti-Patterns to Avoid

- **Rubber-stamp reviews**: Every review must include at least one substantive comment or explicit "no issues found with reasoning"
- **Scope creep during implementation**: If an agent discovers the spec is insufficient, stop and update the spec first
- **Agent echo chambers**: Don't use the same agent to write AND review
- **Skipping validation**: Never skip typecheck/test "just this once"
- **Treating smoke tests as full coverage**: Coverage should come from direct unit tests of exported behavior and failure modes, not incidental execution
- **Using casts to dodge type errors**: Boundary-only assertions can be valid; broad "make TypeScript shut up" casting is not
- **Over-planning**: Once a spec is approved, build it. Don't redesign in implementation.
- **Fixing incidents without changing the system**: If a crash, leak, or major performance regression is fixed without adding a preventive rule, test, or gate, the repo has not actually learned anything
- **Hiding runtime cost in unreadable expressions**: In tensor hot paths, anonymous disposable intermediates and hidden sync points are design smells, not cleverness
