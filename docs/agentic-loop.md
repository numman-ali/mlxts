# Agentic Engineering Loop

## Overview

This project is built using multiple AI coding agents in a structured, iterative loop. No single agent operates unchecked — every output is reviewed by a different agent or human before it ships.

This document defines the process.

## The Loop

```
                    ┌──────────────────────┐
                    │    1. SPEC / PLAN     │
                    │  (Human + Claude)     │
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
                    │  (Codex / Claude)     │
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
                    │  Type check           │
                    │  Run tests            │
                    │  Lint                 │
                    │  Build               │
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

### Claude — Architect / Reviewer

- Primary: planning, architecture, design decisions
- Secondary: implementation of complex/novel code
- Review: code from Codex or Gemini
- Debugging: diagnose issues that other agents can't resolve
- Context keeper: maintains coherence across the project

### Codex — Builder

- Primary: bulk implementation, mechanical porting
- Strengths: parallel execution, large file generation, repetitive patterns
- Works best with: clear specs, defined interfaces, known patterns
- Example tasks: "Implement these 50 C wrapper functions", "Port this Python module to TypeScript"

### Gemini — Second Opinion

- Primary: independent review, alternative approaches
- Strengths: different perspective from Claude, Google ecosystem knowledge
- Review: architecture decisions, API design
- Research: MLX internals, Metal/GPU details, Gemma model specifics

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

### Spec-First Development

Before implementing, the spec must include:

1. **What** — Clear description of the deliverable
2. **Why** — Motivation and context
3. **Interface** — Public API signatures with types
4. **Tests** — At minimum, the test cases (can be written before implementation)
5. **Not included** — Explicit scope boundaries

### Conflict Resolution

When agents disagree:

1. Each agent states its position with reasoning
2. Human decides
3. Decision is recorded in docs or code comments
4. Losing position is not relitigated unless new information emerges

## Workflow by Phase

### Phase 1 (mlx-ts Core)

```
Claude:  Design C wrapper API + TypeScript public API
         ↓
Codex:   Implement C wrapper functions (mechanical, parallelizable)
Claude:  Implement TypeScript FFI layer + public API
         ↓
Bun:     typecheck + test (automated)
         ↓
Gemini:  Review API design, memory safety, edge cases
         ↓
Nomi:    Accept or request changes
```

### Phase 4 (nanoGPT)

```
Claude:  Design model architecture + training loop
         ↓
Codex:   Implement model, data pipeline, training script
         ↓
Bun:     typecheck + test (automated)
Claude:  Train on Shakespeare, analyze loss curves
         ↓
Gemini:  Review model correctness, suggest improvements
         ↓
Nomi:    Accept, evaluate generated text quality
```

## Anti-Patterns to Avoid

- **Rubber-stamp reviews**: Every review must include at least one substantive comment or explicit "no issues found with reasoning"
- **Scope creep during implementation**: If an agent discovers the spec is insufficient, stop and update the spec first
- **Agent echo chambers**: Don't use the same agent to write AND review
- **Skipping validation**: Never skip typecheck/test "just this once"
- **Over-planning**: Once a spec is approved, build it. Don't redesign in implementation.

