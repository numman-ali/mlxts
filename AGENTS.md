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
- `bun run typecheck` is a required validation gate, not optional cleanup
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
- Bun's branded `Pointer` type stays at the FFI boundary; `unwrapPointer()` and `sizeToNumber()` in `ffi.ts` are the only places that narrow these types
- Autograd closures use `mlx_closure` + `JSCallback` — called synchronously on the main thread

### ABI integrity rules

- **ABI-first, types-second, ergonomics-third.** If the ABI model in `ffi.ts` is wrong, every higher-level type becomes fake confidence.
- **Require an ABI audit** of `ffi.ts` symbol declarations whenever mlx-c is upgraded or Bun FFI semantics change.
- **Treat the primitive layer as a product surface**, not a staging area. If the primitives lie, everything above them compounds the lie.
- **No type escape hatches in core code.** Type assertions (`as`, `!`) are forbidden outside `ffi.ts`. If a type doesn't fit, the design needs improving — not a cast.
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

# Type check
bun run typecheck
```

## Agentic Workflow

This project uses multiple AI agents in a structured loop. See [docs/agentic-loop.md](./docs/agentic-loop.md) for the full process. The key rules are: **no agent's output ships without review by a different agent or human**, and **work is not review-ready until typecheck passes**.
