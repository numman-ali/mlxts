# Agent Instructions

This document provides context and instructions for AI coding agents working on this project.

## Project Overview

mlxts is a TypeScript-native ML stack for Apple Silicon. The repo currently
centers on:

- `**@mlxts/core` / `@mlxts/nn` / `@mlxts/optimizers` / `@mlxts/train` / `@mlxts/data` / `@mlxts/tokenizers**`: the extracted reusable ML stack
- `**@mlxts/transformers**`: Pretrained model architectures — LLaMA, Mistral, Gemma families with KV cache, generation, and auto-dispatch
- `**Official Hugging Face JS packages**`: `@huggingface/hub` for snapshot download/cache and `@huggingface/jinja` for chat-template rendering inside the transformers loading surface
- `**packages/nanogpt**`: a temporary GPT validation fixture built on the extracted packages

## Architecture Decisions

- **Runtime**: Bun (for FFI, speed, TypeScript-first)
- **Binding approach**: mlx-c (Apple's official C API, `ml-explore/mlx-c`) → Bun FFI → TypeScript API
- **Monorepo**: Bun workspaces
- **Build**: CMake for native code, Bun/TypeScript for everything else
- **Testing**: Bun's built-in test runner
- **No Python at runtime**: The nn layer and optimizers are rewritten in TypeScript, not wrapped
- **Packages by generation paradigm, not modality**: `@mlxts/transformers` holds all autoregressive architectures (text, MoE, vision encoders, VLMs, encoder-decoders). `@mlxts/diffusion` holds all diffusion/flow generation (image, video, audio). There is no `@mlxts/vlm`, `@mlxts/audio`, or `@mlxts/multimodal` package. See [docs/design-reasoning.md § Generation Paradigms](./docs/design-reasoning.md#generation-paradigms).
- **CausalLM is the universal autoregressive contract**: MoE is a block-level swap inside the decoder, not a new model contract. Multimodal understanding composes encoders with CausalLM, not a replacement. Do not widen CausalLM for anticipated future consumers. See [docs/design-reasoning.md § Contract Boundaries](./docs/design-reasoning.md#contract-boundaries).
- **Weight tying via Embedding.asLinear()**: This is a functional projection (`matmul(x, transpose(weight))`), not a `Linear` module. When `tieWordEmbeddings` is true, call `embedTokens.asLinear(hidden)` in forward — never create shared parameter aliases on the module tree.
- **Shard-iterator-first weight loading**: Use one-tensor-at-a-time safetensor iteration as the default loading strategy, not whole-shard eager materialization.
- **MLX-C first, JS fallback last**: When an operation is needed on the GPU, always check if mlx-c exposes it (`packages/core/native/build/_deps/mlx-c-src/mlx/c/ops.h`) before writing a JS workaround. Binding a missing mlx-c op properly is always better than approximating in JS. Fall back to JS only for genuinely host-side work (small lookups, user-provided callbacks). If mlx-c doesn't expose a needed op but MLX Python has it, consider a custom C binding.
- **Compile before native helper for repeated pure subgraphs**: If a hot path repeatedly rebuilds the same pure tensor->tensor motif, try `compile({ shapeless: true })` before introducing custom native bindings. Native helpers are for hot mutable state or primitives compile cannot express cleanly.
- **When parity still lags, deepen one semantic stage instead of scattering micro-optimizations**: After compile-first and obvious algorithmic fixes are exhausted, choose the next optimization seam around one hot semantic stage (`decode attention`, `cache update + visible fetch`, `optimizer step`), not around isolated plumbing ops. Package together the hot adjacent intermediates that are private to that stage and would otherwise bounce across Bun/FFI unnecessarily. Keep the model/training surface semantic, benchmark against paired reference numbers, and keep only seams that shrink the paired gap repeatably.
- **Performance is an observable, not a review opinion**: Generation hot-path changes require before/after benchmark numbers in the review artifact, not just "I considered performance." See [docs/runtime-safety.md § Generation Performance](./docs/runtime-safety.md#generation-performance). Key invariants: one eval per token in steady state, sampling stays on GPU, prefill is chunked, GPU never idles between tokens via async eval pipelining.
- **Runtime strategy is not model identity**: Keep checkpoint truth separate from runtime/backend choices. Do not create duplicate model configs for managed vs native cache, eager vs compiled helpers, or future KV representation variants.
- **Heavy MLX commands are exclusive**: Benchmarks, soak runs, acceptance runs, memory benches, and long-running training/proof commands must hold the shared runtime lock. Never run them in parallel on one machine.
- **Keep semantic names semantic**: Public helpers and model call sites should read in terms of math and model behavior. Compile and keyed transform reuse are runtime strategies under those names, not the dominant vocabulary of product code.
- **Keep repo ergonomics consistent at the pattern level**: Similar families and package surfaces should use similar semantic/control-flow patterns so understanding transfers across the repo. Consistency does not mean identical file counts or helper layers everywhere; it means the same kind of thing should read the same kind of way.

## Coding Conventions

See [docs/code-standards.md](./docs/code-standards.md) for the full code standards. Key points:

- TypeScript strict mode, no `any` types except at the FFI boundary
- Type assertions are boundary tools, not everyday design tools; avoid `as` and `!` unless a well-understood FFI edge requires them
- Use `using` (explicit resource management) for native array handles where appropriate
- Prefer functional style for tensor operations, class-based for nn modules (mirrors MLX's own design)
- All public APIs must have JSDoc with at least a one-line description
- Test files live next to source: `foo.ts` → `foo.test.ts`
- Use Bun's test runner: `bun test`
- Bun is the only JavaScript/TypeScript runtime in this repo. Do not add `node:`* imports or Node-only execution assumptions; prefer Bun-native APIs first, then Bun-compatible neutral imports only when needed.
- `bun run typecheck` is a required validation gate, not optional cleanup
- `bun run check:runtime-review` is required whenever runtime-sensitive production files change; the diff must include a review artifact under `docs/reviews/` and that artifact's `Files Reviewed` section must name the changed runtime-sensitive files
- `bun run check:tensor-lifetimes` is an AST-based static backstop for the anonymous-intermediate leak class; when a new tensor-producing primitive is added, update the canonical tracked-op list in `scripts/`
- `bun run check:coverage` is a required quality gate across the canonical package stack and the temporary `packages/nanogpt/` validation fixture (`95%` lines, `90%` functions, with branch thresholds enforced only when LCOV reports branch counters)
- Prefer direct unit coverage of exported behavior and dynamic failure paths over broad smoke-only tests
- The repo is forward-moving and canonical: do not add legacy compatibility code, fallback modes, or stale docs for APIs we no longer want to carry
- If a surface is no longer part of the intended product, delete it instead of preserving it behind flags or compatibility layers
- Runtime-sensitive changes must leave local tensor lifetimes visible in code. Do not hide disposable `MxArray` intermediates inside nested expressions.
- FFI result pointers must use per-call `OutSlot`-style ownership. Do not reintroduce shared reusable output buffers.
- Transform-returning helpers should be explicitly disposable when they hold native resources beyond a single call.
- If a serious runtime, memory, or performance incident is fixed, the same change must also add a preventive rule, test, benchmark, or validation gate.
- `packages/nanogpt/` is currently a temporary validation fixture during the package extraction, not the long-term canonical product surface. Prefer improving the reusable `@mlxts/*` packages over deepening the temporary app.
- Non-trivial operator logic belongs under canonical package-owned surfaces, not loose root scripts. Avoid creating new permanent product contracts on top of temporary migration code.
- Snapshot checkpoints and resume checkpoints are both canonical, but they serve different purposes: snapshots are lightweight model saves, resume checkpoints carry optimizer state for exact continuation.
- **Code must be self-documenting**: names, types, and structure carry meaning. Comments explain *why*, never *what*.
- **Human readability is a first-class concern**: every function should be immediately understandable to a TypeScript developer unfamiliar with this codebase.

## Reference Repositories

The `.reference/` folder contains local clones of upstream repositories for research and cross-referencing:


| Repo                      | Purpose                                                                                                                        |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `.reference/mlx-lm`       | Apple's MLX language model library — primary reference for model architectures, weight loading, generation, LoRA, quantization |
| `.reference/transformers` | HuggingFace Transformers — reference for model configs, tokenizer formats, weight naming conventions                           |
| `.reference/trl`          | HuggingFace TRL — reference for SFT, DPO, alignment training patterns                                                          |
| `.reference/mlx-examples` | Apple's MLX examples — reference for LLaVA, Stable Diffusion, Whisper implementations                                          |
| `.reference/diffusers`    | HuggingFace Diffusers — planned Phase 10 reference for diffusion pipeline structure and checkpoint conventions; clone before Phase 10 research begins |
| `.reference/rapid-mlx`    | Rapid-MLX — reference for inference speed (MTP, speculative decode, DeltaNet snapshots, prompt caching, tool parsing, jump-forward decoding) |
| `.reference/vllm-mlx`     | vLLM-MLX — foundational reference for paged KV cache, continuous batching, multimodal engine composition on MLX                 |
| `.reference/omlx`         | oMLX — reference for production serving (tiered SSD KV cache, multi-model memory management, oQ mixed-precision quantization)   |


**These should be kept up to date.** When investigating a new model family, architecture pattern, or training technique, check `.reference/` first. Pull latest before starting research for a new phase.

**Usage**: Read and study these for design decisions, architecture patterns, and correctness validation. Do not copy code — our implementations are TypeScript-native and designed from first principles. These repos inform *what* to build and validate *correctness*, not dictate *how* to build it.

## Documentation


| Document                                               | Purpose                                                                            |
| ------------------------------------------------------ | ---------------------------------------------------------------------------------- |
| [PLAN.md](./PLAN.md)                                   | Phased build plan with deliverables and exit criteria                              |
| [docs/design-reasoning.md](./docs/design-reasoning.md) | Why we make the design choices we do — composition, visibility, abstraction timing |
| [docs/architecture.md](./docs/architecture.md)         | System architecture and layer responsibilities                                     |
| [docs/mlx-bindings.md](./docs/mlx-bindings.md)         | Technical guide to the MLX binding approach                                        |
| [docs/agentic-loop.md](./docs/agentic-loop.md)         | Multi-agent engineering workflow                                                   |
| [docs/code-standards.md](./docs/code-standards.md)     | Code quality, naming, structure, testing standards                                 |
| [docs/runtime-safety.md](./docs/runtime-safety.md)     | Runtime ownership, telemetry, and soak expectations                                |
| [docs/product-surfaces.md](./docs/product-surfaces.md) | API, CLI, TUI, GUI design guidelines                                               |
| [docs/setup.md](./docs/setup.md)                       | Development environment setup and build instructions                               |


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

### nn.Module parameter scanning

`Module.parameters()` scans own enumerable properties:

- `MxArray` → leaf parameter (included in the tree)
- `Module` → recurse into that module's parameters
- `Module[]` → scanned with string indices ("0", "1"...) matching HuggingFace weight naming (`model.layers.0.self_attn.q_proj.weight`)
- Everything else (numbers, strings, functions, `#private` fields) → silently skipped

**Implication for @mlxts/transformers**: Every structural component (attention, MLP, norm, block, model) MUST extend `nn.Module`. Config scalars and non-parameter state must use `#` private fields to avoid being scanned.

### Tensor ownership in weight loading

When loading pretrained weights via `iterateSafetensorWeights`:

- **Assigned** tensors: owned by the model after `assignWeightPath()`. Freed when model is disposed.
- **Skipped** tensors (sanitize returns null): freed immediately after the skip decision.
- **Error path** tensors: freed in the catch block. Never leaked.

A post-load audit compares assigned paths against the model's parameter tree. Missing weights throw `MissingWeightsError`. Unexpected weights either warn or throw depending on strict mode.

## Build Commands

```bash
# Install dependencies
bun install

# Build native bindings
bun run build:native

# Build package dist outputs
bun run build

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

# Local release-readiness checks
bun run docs:api
bun run pack:dry-run
bun run release:check
```

## Agentic Workflow

This project uses multiple AI agents in a structured loop. See [docs/agentic-loop.md](./docs/agentic-loop.md) for the full process. The key rules are: **no agent's output ships without review by a different agent or human**, and **work is not review-ready until typecheck and coverage gates pass**.

### Workflow principles

- **Done fully, never simply.** Every phase should be thoroughly researched and planned before implementation begins. Do not rush features into earlier phases to "get them done sooner." If something is scoped for a later phase, that is the right decision — better to do it fully later than partially now.
- **Research spike before new territory.** Phases involving genuinely new architectural ground (new generation paradigms, new model families with novel architecture patterns) require a dedicated research spike before implementation begins. See Phase 0.5 as the model: investigate official sources, validate assumptions, then plan.
- **Phases fan out, not chain.** Phases 8 (fine-tuning), 9 (serving), and 10 (multimodal + diffusion) all depend on Phase 7 (model architectures) but not on each other. Do not treat the phase numbering as a strict sequential dependency.
- **Contracts describe behavior, not internals.** When a new model variant appears, ask "does the existing contract cover its external behavior?" before proposing a new interface. MoE is a block-level swap, not a new CausalLM. VLMs compose encoders with CausalLM, not a replacement.
- **Reference parity audit before benchmarking.** When implementing a new model family in `@mlxts/transformers`, audit each hot-path function against its mlx-lm equivalent *before* running benchmarks. Count: (a) MLX ops per decode token, (b) intermediate tensor allocations per decode token, (c) mask values passed to SDPA during single-token decode (`null` vs boolean tensor), (d) cache update strategy (O(1) write vs O(n) concatenation). If any of these differ significantly from mlx-lm, investigate and resolve before benchmarking. The benchmark tells you *that* something is slow; the parity audit tells you *why* before you ship it. See [docs/runtime-safety.md § Forward pass performance invariants](./docs/runtime-safety.md#forward-pass-performance-invariants) for the specific invariants to check.
- **Use the runtime optimization matrix.** For repeated hot-path work across families or packages, consult and update [docs/runtime-optimization-matrix.md](./docs/runtime-optimization-matrix.md). It is the canonical map for which surfaces should try compile first, which require deeper core work, and which justify native helpers only after compile has been exhausted.
- **Keep runtime strategy out of semantic names.** Public and call-site names should describe the math or model behavior (`swiglu`, `crossEntropy`, `repetition penalty`), not the execution strategy. Compile, shape-keyed reuse, and native-assist choices stay internal unless a dedicated runtime surface is truly needed.
- **Run optimization work as a revertable research loop.** Start from a clean baseline, try one bounded idea at a time, measure it against the paired reference metric, remove losing experiments immediately, and document what was tried so later compactions and rollouts stay grounded in evidence. We are allowed to be ambitious in the backend layer, but only the wins earn permanence.
- **Winner candidates need long-context confirmation, not just short-run luck.** Short probes are useful for isolating mechanisms, but they do not prove a keeper. Before a performance experiment earns permanence, recheck it with boundary-sensitive or longer-context parity runs where cache growth, visible-window behavior, and long decode loops can actually show themselves. Judge the winner on the paired gap under those conditions, not on a single tiny benchmark.

Runtime-sensitive changes add one more requirement: they need a review artifact under `docs/reviews/` that records the files reviewed, tensor-lifetime audit, memory/performance evidence, independent review, and remaining risks. The `Files Reviewed` section must list the exact changed runtime-sensitive files.

For now, `packages/nanogpt/` remains a validation fixture rather than the long-term operator
surface. Do not create new permanent product contracts on top of it; the package ecosystem is the
primary deliverable, and polished examples move to a dedicated examples repo later.
