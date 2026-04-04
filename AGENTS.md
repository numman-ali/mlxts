# Agent Instructions

This document provides context and instructions for AI coding agents working on this project.

## Project Overview

nanogpt-ts is a TypeScript-native ML stack for Apple Silicon. It consists of:

- **mlx-ts**: FFI bindings from TypeScript/Bun to Apple's MLX C++ framework
- **nanogpt**: A GPT implementation built on mlx-ts

## Architecture Decisions

- **Runtime**: Bun (for FFI, speed, TypeScript-first)
- **Binding approach**: mlx-c (Apple's official C API, `ml-explore/mlx-c`) → Bun FFI → TypeScript API
- **Monorepo**: Bun workspaces
- **Build**: CMake for native code, Bun/TypeScript for everything else
- **Testing**: Bun's built-in test runner
- **No Python at runtime**: The nn layer and optimizers are rewritten in TypeScript, not wrapped

## Coding Conventions

See [docs/code-standards.md](./docs/code-standards.md) for the full code standards. Key points:

- TypeScript strict mode, no `any` types except at the FFI boundary
- Type assertions are boundary tools, not everyday design tools; avoid `as` and `!` unless a well-understood FFI edge requires them
- Use `using` (explicit resource management) for native array handles where appropriate
- Prefer functional style for tensor operations, class-based for nn modules (mirrors MLX's own design)
- All public APIs must have JSDoc with at least a one-line description
- Test files live next to source: `foo.ts` → `foo.test.ts`
- Use Bun's test runner: `bun test`
- Bun is the only JavaScript/TypeScript runtime in this repo. Do not add `node:*` imports or Node-only execution assumptions; prefer Bun-native APIs first, then Bun-compatible neutral imports only when needed.
- `bun run typecheck` is a required validation gate, not optional cleanup
- `bun run check:runtime-review` is required whenever runtime-sensitive production files change; the diff must include a review artifact under `docs/reviews/` and that artifact's `Files Reviewed` section must name the changed runtime-sensitive files
- `bun run check:tensor-lifetimes` is an AST-based static backstop for the anonymous-intermediate leak class; when a new tensor-producing primitive is added, update the canonical tracked-op list in `scripts/`
- `bun run check:coverage` is a required quality gate for the production packages: `mlx-ts` (`95%` lines, `90%` functions) and `nanogpt` (`90%` lines, `85%` functions). Branch thresholds are enforced only when LCOV provides branch counters; otherwise the script says branch data was unavailable
- Prefer direct unit coverage of exported behavior and dynamic failure paths over broad smoke-only tests
- The repo is forward-moving and canonical: do not add legacy compatibility code, fallback modes, or stale docs for APIs we no longer want to carry
- If a surface is no longer part of the intended product, delete it instead of preserving it behind flags or compatibility layers
- Runtime-sensitive changes must leave local tensor lifetimes visible in code. Do not hide disposable `MxArray` intermediates inside nested expressions.
- FFI result pointers must use per-call `OutSlot`-style ownership. Do not reintroduce shared reusable output buffers.
- Transform-returning helpers should be explicitly disposable when they hold native resources beyond a single call.
- If a serious runtime, memory, or performance incident is fixed, the same change must also add a preventive rule, test, benchmark, or validation gate.
- Long-running GPT training is an operational product surface, not an ad hoc script. Unattended or resumable runs must go through `packages/nanogpt/src/run/` and the `bun run run:nanogpt ...` manager flow.
- Non-trivial operator logic belongs under package source, not loose root scripts. Improve the canonical package-owned surface instead of adding a side path.
- Snapshot checkpoints and resume checkpoints are both canonical, but they serve different purposes: snapshots are lightweight model saves, resume checkpoints carry optimizer state for exact continuation.
- **Code must be self-documenting**: names, types, and structure carry meaning. Comments explain *why*, never *what*.
- **Human readability is a first-class concern**: every function should be immediately understandable to a TypeScript developer unfamiliar with this codebase.

## Documentation


| Document                                           | Purpose                                               |
| -------------------------------------------------- | ----------------------------------------------------- |
| [PLAN.md](./PLAN.md)                               | Phased build plan with deliverables and exit criteria |
| [docs/architecture.md](./docs/architecture.md)     | System architecture and layer responsibilities        |
| [docs/mlx-bindings.md](./docs/mlx-bindings.md)     | Technical guide to the MLX binding approach           |
| [docs/agentic-loop.md](./docs/agentic-loop.md)     | Multi-agent engineering workflow                      |
| [docs/code-standards.md](./docs/code-standards.md) | Code quality, naming, structure, testing standards    |
| [docs/runtime-safety.md](./docs/runtime-safety.md) | Runtime ownership, telemetry, and soak expectations   |
| [docs/product-surfaces.md](./docs/product-surfaces.md) | API, CLI, TUI, GUI design guidelines                  |
| [docs/setup.md](./docs/setup.md)                   | Development environment setup and build instructions  |


## Key Technical Context

### MLX's computation model

- **Lazy evaluation**: Operations build a computation graph. Nothing runs until `mx.eval()` is called.
- **Unified memory**: CPU and GPU share memory on Apple Silicon. No explicit transfers needed.
- **Functional autograd**: `mx.grad(fn)` returns a new function. This is JAX-style, not PyTorch-style tape recording.
- **Streams**: Operations are dispatched to streams (CPU or GPU). Default is GPU.

### FFI boundary concerns

- MLX arrays are C++ objects wrapped by mlx-c as opaque pointers (`struct { void* ctx; }`). We hold `ctx` as Bun's branded `Pointer` type in `MxArray._ctx`.
- We must prevent memory leaks by calling `mlx_array_free` when JS finishes with an array. Use FinalizationRegistry for automatic cleanup as a safety net, but prefer explicit disposal via `using` declarations.
- Native handle temporaries (vector arrays, device handles, RNG keys) must use `try/finally` for cleanup — structural safety, not hope-based cleanup.
- mlx-c provides pure C linkage — no custom C wrapper needed
- All mlx-c operations return `int` (0 = success) — check via `checkStatus()` which throws typed `MxError`
- Property getters (`mlx_array_shape`, `mlx_array_ndim`, etc.) return values directly — no output pointer pattern
- Bun's branded `Pointer` type stays at the FFI boundary; `unwrapPointer()` and `sizeToNumber()` in `src/core/ffi/` are the only places that narrow these types
- Autograd closures use `mlx_closure` + `JSCallback` — called synchronously on the main thread

### ABI integrity rules

- **ABI-first, types-second, ergonomics-third.** If the ABI model in `src/core/ffi/` is wrong, every higher-level type becomes fake confidence.
- **Require an ABI audit** of `src/core/ffi/symbols.ts` declarations whenever mlx-c is upgraded or Bun FFI semantics change.
- **Treat the primitive layer as a product surface**, not a staging area. If the primitives lie, everything above them compounds the lie.
- **No type escape hatches in core code.** Type assertions (`as`, `!`) are forbidden outside `src/core/ffi/`. If a type doesn't fit, the design needs improving — not a cast.
- **Prefer runtime checks that teach the type system something true** over casts that merely silence the compiler.

### What nanoGPT needs from mlx-ts (minimum viable surface)

1. Array creation: zeros, ones, full, arange, from typed arrays
2. Core ops: matmul, add, multiply, reshape, transpose, softmax, cross_entropy
3. Autograd: value_and_grad
4. Random: normal, uniform, key/split
5. nn: Module, Linear, Embedding, LayerNorm, GELU, Dropout
6. Optimizers: AdamW
7. eval() for forcing computation

## Build Commands

```bash
# Install dependencies
bun install

# Build native bindings
cd packages/mlx-ts && bun run build:native

# Run tests
bun test

# Run the coverage gate
bun run check:coverage

# Check runtime review artifacts for hot-path diffs
bun run check:runtime-review

# Check for suspicious nested tensor-producing calls
bun run check:tensor-lifetimes

# Memory and soak investigation
bun run bench:memory
bun run soak:gpt-tiny
bun run soak:gpt-small

# Long acceptance runs
bun run acceptance:gpt-tiny
bun run acceptance:gpt-small

# Canonical supervised long-run control
bun run run:nanogpt start --preset gpt-small --max-steps 5000
bun run run:nanogpt status --name <run-id>
bun run run:nanogpt stop --name <run-id>
bun run run:nanogpt resume --from <run-id> --max-steps 10000

# Type check
bun run typecheck

# Full validation
bun run validate
```

## Agentic Workflow

This project uses multiple AI agents in a structured loop. See [docs/agentic-loop.md](./docs/agentic-loop.md) for the full process. The key rules are: **no agent's output ships without review by a different agent or human**, and **work is not review-ready until typecheck and coverage gates pass**.

Runtime-sensitive changes add one more requirement: they need a review artifact under `docs/reviews/` that records the files reviewed, tensor-lifetime audit, memory/performance evidence, independent review, and remaining risks. The `Files Reviewed` section must list the exact changed runtime-sensitive files.

For long-running training, `bun run run:nanogpt ...` is the canonical operator surface. Do not add
alternate legacy scripts, one-off daemon paths, or undocumented checkpoint flows; if the supervised
run manager is not good enough, improve it directly and delete the stale path.
