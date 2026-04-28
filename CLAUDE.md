@AGENTS.md

## Project: mlxts

TypeScript-native ML stack for Apple Silicon. MLX bindings, neural networks,
training, pretrained model loading, generation, and serving.

Per-package AGENTS.md files auto-inject when files inside that package are
touched. Read the package's AGENTS.md before editing files inside it.

Always read first:
- AGENTS.md — repo doctrine
- MEMORY.md Tier 1 — durable cross-session sharp edges
- continuity.md — current-phase handoff state

Then as needed:
- PLAN.md for phase status and exit criteria
- docs/agentic-loop.md for the engineering workflow
- docs/design-reasoning.md for the reasoning behind structural choices
- docs/ecosystem-structure.md for the @mlxts/* package map
- docs/runtime-safety.md and docs/runtime-optimization-matrix.md for hot paths

Build, test, gate, and bench commands live in AGENTS.md § Build Commands.
