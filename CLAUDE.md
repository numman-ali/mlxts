@AGENTS.md

## Project: mlxts

TypeScript-native ML stack for Apple Silicon. MLX bindings, neural networks, training, pretrained model loading and generation.

Read PLAN.md for current phase and priorities.
Read AGENTS.md for architecture decisions and coding conventions.
Read docs/agentic-loop.md for the engineering workflow.
Read docs/design-reasoning.md for why we make the design choices we do.
Read docs/ecosystem-structure.md for the @mlxts/* package map.

## Current Phase

Phase 1–4: Complete (core bindings, autograd, nn, nanoGPT).
Phase 5: Ecosystem Restructure — complete. Renamed to mlxts, extracted @mlxts/* packages.
Phase 6: Publish Core Packages — complete. Modern transformer primitives in @mlxts/nn.
Phase 6.5: Modern Transformer Primitives — complete.
Phase 7: Model Architectures — **in progress**. Dense text families (LLaMA, Mistral, Gemma) working. Expanding to Phi-3, Gemma 3, Mistral 3. MoE (Phase 7f) follows.

## Quick Reference

- Runtime: Bun
- Test: `bun test`
- Build native: `cd packages/core && bun run build:native`
- Typecheck: `bun run typecheck`
- Lint: `bun run lint`
- Full validation: `bun run validate`
- Build all packages: `bun run build`
- API docs: `bun run docs:api`
- Pack dry run: `bun run pack:dry-run`
