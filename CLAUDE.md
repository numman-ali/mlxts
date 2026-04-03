@AGENTS.md

## Project: nanogpt-ts

TypeScript-native GPT implementation with MLX bindings for Apple Silicon.

Read PLAN.md for current phase and priorities.
Read AGENTS.md for architecture decisions and coding conventions.
Read docs/agentic-loop.md for the engineering workflow.

## Current Phase

Phase 1: Core Bindings — complete. FFI layer, MxArray, ops, random, eval all working.
Next: Phase 2 (Autograd) or Biome hardening.

## Quick Reference

- Runtime: Bun
- Test: `bun test`
- Build native: `cd packages/mlx-ts && bun run build:native`
- Typecheck: `bun run typecheck`
- Lint: `bun run lint`
- Full validation: `bun run validate` (typecheck + lint + test)

