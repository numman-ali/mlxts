# nanogpt-ts

A from-scratch implementation of GPT in TypeScript, built on native MLX bindings for Apple Silicon.

## What is this?

This project has three goals:

1. **mlx-ts** — First-class TypeScript bindings for Apple's [MLX](https://github.com/ml-explore/mlx) framework, enabling GPU-accelerated machine learning on Apple Silicon directly from TypeScript/Bun.
2. **nanogpt-ts** — A clean, educational GPT implementation in TypeScript, inspired by [Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT), built on top of mlx-ts.
3. **Education** — Every design decision is documented. The goal is to make transformers and LLM training accessible to the TypeScript/JavaScript developer community.

## Why TypeScript?

The ML ecosystem is locked into Python — not because Python is the best tool, but because of momentum. The actual compute happens in C++/CUDA/Metal. Python is glue.

TypeScript can be that glue too. With Bun's FFI and Apple's MLX, we can call the same Metal GPU kernels from TypeScript with near-zero overhead. The result: ML that feels native to the millions of developers who already think in TypeScript.

This is not a toy reimplementation. It's a real, GPU-accelerated training stack.

## Project Structure

```
nanogpt-ts/
├── packages/
│   ├── mlx-ts/          # MLX bindings for TypeScript (Bun FFI → MLX C++)
│   └── nanogpt/         # GPT implementation using mlx-ts
├── examples/            # Runnable demos and tutorials
├── docs/                # Architecture docs, guides, technical references
├── PLAN.md              # Phased build plan
└── AGENTS.md            # Agent coordination instructions
```

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- [Bun](https://bun.sh) runtime
- MLX (installed via `pip install mlx` or built from source)
- CMake (for building native bindings)

## Status

**Phase 0: Foundation** — Setting up project structure, documentation, and build plan.

See [PLAN.md](./PLAN.md) for the full roadmap.

## License

MIT
