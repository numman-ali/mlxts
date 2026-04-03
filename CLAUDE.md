@AGENTS.md

## Project: nanogpt-ts

TypeScript-native GPT implementation with MLX bindings for Apple Silicon.

Read PLAN.md for current phase and priorities.
Read AGENTS.md for architecture decisions and coding conventions.
Read docs/agentic-loop.md for the engineering workflow.

## Current Phase

Phase 1: Core Bindings — complete.
Phase 2: Autograd — complete.
Phase 3: Neural Network Layer — complete. Module, layers, losses, optimizers, nn.valueAndGrad.
Next: Phase 4 (nanoGPT).

## Quick Reference

- Runtime: Bun
- Test: `bun test`
- Build native: `cd packages/mlx-ts && bun run build:native`
- Typecheck: `bun run typecheck`
- Lint: `bun run lint`
- Full validation: `bun run validate` (typecheck + lint + test)

