@AGENTS.md

## Project: nanogpt-ts → mlxts

TypeScript-native ML stack for Apple Silicon. MLX bindings, neural networks, training, and a working GPT.

Read PLAN.md for current phase and priorities.
Read AGENTS.md for architecture decisions and coding conventions.
Read docs/agentic-loop.md for the engineering workflow.
Read docs/ecosystem-structure.md for the @mlxts/* package map.

## Current Phase

Phase 1: Core Bindings — complete.
Phase 2: Autograd — complete.
Phase 3: Neural Network Layer — complete. Module, layers, losses, optimizers, nn.valueAndGrad.
Phase 4: nanoGPT — complete. Training, auto-stop, best-checkpoint, gradient-checkpointing, supervised runs.
Next: Phase 5 (Ecosystem Restructure — rename to mlxts, extract @mlxts/* packages).

## Quick Reference

- Runtime: Bun
- Test: `bun test`
- Build native: `cd packages/mlx-ts && bun run build:native`
- Typecheck: `bun run typecheck`
- Lint: `bun run lint`
- Full validation: `bun run validate` (typecheck + lint + test)

