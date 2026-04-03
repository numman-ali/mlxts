# Build Plan

## Vision

Build a complete, GPU-accelerated ML training stack in TypeScript for Apple Silicon — from MLX bindings through to a working GPT that trains on a MacBook.

**Non-goal (for now)**: Performance parity with Python MLX. The priority is correctness, clarity, and education.

## Design Philosophy

This is a **product**, not a project. The difference matters.

Most open source ML repositories are built by researchers for researchers. Code ships fast, documentation is an afterthought, APIs break between releases, features land without migration guides. The codebase reflects the author's mental model, not the reader's learning path.

We reject that. Our principles:

1. **The first-time user is the primary audience.** Every API, every file, every error message is designed for someone encountering it fresh. If a developer can't understand what a module does within 30 seconds of opening it, we've failed.

2. **Original thinking from first principles.** We study the official MLX source code and nothing else. No third-party wrappers, no community bindings, no derivative projects. Their compromises are not our compromises. Our architecture emerges from understanding the problem, not from copying someone else's solution.

3. **If it's not documented, it doesn't exist.** Every public API has JSDoc. Every architectural decision has a rationale. Every phase has exit criteria.

4. **Stability is a feature.** We don't ship half-built APIs. Each phase delivers something complete and tested before the next begins.

5. **Developer experience is not optional.** TypeScript types, clear error messages, predictable behavior, logical naming. The library should feel inevitable — like it couldn't have been designed any other way.

6. **Every surface is a product surface.** API, CLI, TUI, GUI — each has its own users, principles, and quality bar. Even when we're only building the API, we design it knowing a CLI will be built on top, a TUI on top of that, and a GUI on top of that. Good foundations make every layer above them better. See [docs/product-surfaces.md](./docs/product-surfaces.md) for the full guidelines.

---

## Phase 0: Foundation

**Status**: Complete

**Goal**: Project structure, documentation, and a reviewed plan before any code is written.

**Deliverables**:

- Monorepo directory structure
- README.md — project overview and motivation
- CLAUDE.md / AGENTS.md — agent instructions
- PLAN.md — this document
- docs/architecture.md — system architecture and design decisions
- docs/agentic-loop.md — engineering workflow
- docs/mlx-bindings.md — technical guide to the binding approach
- Bun workspace configuration (package.json, tsconfig)
- Review by at least two agents before proceeding

**Exit criteria**: Plan is reviewed and approved. All agents have consistent context.

---

## Phase 0.5: Research Spike — Validate Assumptions

**Status**: Complete

**Goal**: Verify that our plan aligns with the current state of MLX and Bun before writing implementation code. The plan was built from training-data knowledge — this phase grounds it in reality.

**Principle**: Research only the **official source** — Apple's MLX repository and Bun's documentation. We do not look at third-party implementations, community wrappers, or derivative projects. Our architecture should emerge from first principles and the official API, not from inheriting another project's compromises. We are building a product, not a fork.

**What this phase covers**:

### 0.5a. MLX official API investigation
- Clone `ml-explore/mlx` — the **only** source of truth
- Inspect `mlx-c/` — MLX ships an official C API. If it's mature enough, it **eliminates the need for our custom C wrapper**, which would be a major simplification.
- Catalog the available C functions and compare against our binding plan in `docs/mlx-bindings.md`
- Check how autograd (`grad`, `value_and_grad`) is exposed at the C level
- Identify any gaps (functions we need that aren't in the C API)
- Read MLX's latest release notes for breaking changes or new capabilities

**Findings**: mlx-c is a **separate repo** (`ml-explore/mlx-c`, v0.6.0, tracking MLX v0.31.1) with 580+ C functions. Autograd is fully exposed via the `mlx_closure` + `mlx_value_and_grad` pipeline. Memory is manual new/free with opaque pointers (`struct { void* ctx; }`). All primitives nanoGPT needs are present. Bonus: fused SDPA, RoPE, RMS norm, layer norm, and safetensors I/O. **Decision: use mlx-c directly — no custom C wrapper needed.**

### 0.5b. Bun FFI validation
- Verify Bun 1.3.x FFI callback support (`JSCallback`) — required for autograd
- Test a minimal FFI proof-of-concept: load a .dylib, call a function, get a result
- Confirm pointer handling, memory semantics, and cleanup patterns

**Findings**: Bun 1.3.4 FFI works well. JSCallback supports closures, iteration, and multi-callback patterns. ~8ns/call for basic FFI, ~35ns for callbacks — negligible vs ML compute. Pointers are JS `number` (not BigInt). `threadsafe: true` crashes on 1.3.4, but MLX calls closures synchronously during graph construction, so `threadsafe: false` suffices. FinalizationRegistry works. **Assessment: Bun FFI is sufficient for all binding needs.**

### 0.5c. Update documentation
- Revise `docs/mlx-bindings.md` with findings
- Update `docs/architecture.md` if the C API changes our layer diagram
- Flag any changes that affect the Phase 1-4 plan

**What this phase explicitly excludes**:
- No reviewing third-party MLX bindings (Node, Swift, Rust, or otherwise)
- No adopting patterns from other JS/TS ML libraries
- No community wrappers or derivative projects
- Our API design comes from our own product thinking, informed by the official MLX API and TypeScript best practices — nothing else

**Deliverables**:
- Updated `docs/mlx-bindings.md` grounded in official MLX source code
- A working Bun FFI proof-of-concept (even if trivial)
- Go/no-go decision on custom C wrapper vs. official `mlx-c` API

**Exit criteria**: All binding-related claims in docs are verified against official MLX source. Architecture is updated if needed.

---

## Phase 1: mlx-ts Core Bindings

**Goal**: TypeScript can create arrays, run operations, and evaluate results on the GPU.

**What this phase covers**:

### 1a. Build infrastructure

- CMakeLists.txt to build mlx-c from source (fetches MLX automatically via FetchContent)
- Bun build script that compiles mlx-c and produces libmlxc.dylib
- CI-like validation script (typecheck + test)

### 1b. Bun FFI bindings (`src/core/ffi.ts`)

- Load libmlxc.dylib via Bun's `dlopen`
- Map mlx-c functions to TypeScript with correct types
- Pointer management for array handles
- FinalizationRegistry for automatic cleanup

### 1d. TypeScript API (`src/core/`)

- `array.ts` — mx.array class wrapping FFI handles
- `ops.ts` — elementwise, reductions, linear algebra
- `dtype.ts` — float32, float16, bfloat16, int32, bool
- `device.ts` — cpu/gpu device selection
- `random.ts` — mx.random.normal, uniform, key, split
- `transforms.ts` — eval, compile

### 1e. Tests

- Array creation and property access
- Arithmetic operations (CPU and GPU)
- Matmul correctness against known values
- Memory: no leaks over 10k array allocations
- GPU execution: verify ops run on Metal

**Deliverables**:

- `packages/mlx-ts` with working core bindings
- Can run: `const a = mx.ones([3, 3]); const b = mx.matmul(a, a); mx.eval(b); console.log(b.toList())`
- All tests pass

**Exit criteria** — Phase 1 is complete when all of the following are true:
1. `bun run typecheck` passes with zero errors
2. `bun test` passes all tests (75+ across 6 test files)
3. `bun run lint` (Biome) passes clean
4. FFI symbol declarations in `ffi.ts` are verified against mlx-c v0.6.0 headers
5. No type assertions (`as`, `!`) exist outside the FFI boundary layer (`ffi.ts`)
6. All native handle temporaries use `try/finally` for cleanup
7. Explicit-dtype array creation and all scalar dtype paths are covered by tests
8. Smoke test works: `mx.ones([3,3])` → `mx.matmul(a,a)` → `mx.eval(b)` → `b.toList()` returns `[[3,3,3],[3,3,3],[3,3,3]]`

---

## Phase 2: Autograd

**Goal**: `mx.grad()` and `mx.valueAndGrad()` work from TypeScript.

**Why this is its own phase**: Autograd is the hardest part of the binding. MLX's grad traces through the C++ computation graph, but the loss function is defined in TypeScript. The FFI boundary must handle callbacks correctly.

**What this phase covers**:

### 2a. Research and design

- Study MLX's C++ transform implementation
- Design the callback mechanism (TS function → C++ trace → TS)
- Document the approach in docs/mlx-bindings.md

### 2b. Implementation

- C wrapper for grad, value_and_grad, jvp, vjp
- FFI callback support in Bun
- TypeScript `mx.grad(fn)` and `mx.valueAndGrad(fn)` APIs

### 2c. Tests

- Gradient of simple functions (x^2, linear, polynomial)
- Gradient through matmul
- Gradient through softmax + cross_entropy
- Numerical gradient checking (finite differences vs autograd)

**Deliverables**:

- Working autograd from TypeScript
- Can compute gradients of arbitrary TypeScript functions over mx.arrays

**Exit criteria**: `mx.valueAndGrad(fn)(x)` returns correct loss and gradients, verified by numerical gradient checking.

---

## Phase 3: Neural Network Layer

**Goal**: A PyTorch-like nn.Module system in TypeScript, built on mlx-ts core.

**What this phase covers**:

### 3a. Module system

- Base `Module` class with parameter registration, tree traversal
- `parameters()`, `update()`, `loadWeights()`, `saveWeights()`
- Nested module support

### 3b. Layers

- `Linear` — fully connected layer
- `Embedding` — token/position embeddings
- `LayerNorm`, `RMSNorm` — normalization
- `Dropout` — training regularization
- `Conv1d`, `Conv2d` — convolutional layers (lower priority)

### 3c. Activations

- `ReLU`, `GELU`, `SiLU`, `Sigmoid`, `Tanh`, `Softmax`

### 3d. Losses

- `crossEntropy` — the main one for language modeling
- `mse` — mean squared error

### 3e. Optimizers

- `SGD` with momentum and weight decay
- `Adam`, `AdamW`
- Learning rate schedules: cosine annealing, linear warmup

### 3f. Tests

- Each layer: forward pass shape and value correctness
- Each optimizer: parameter update matches hand calculation
- End-to-end: train a 2-layer MLP on XOR (converges to 0 loss)

**Deliverables**:

- `packages/mlx-ts/src/nn/` and `packages/mlx-ts/src/optimizers/`
- Can define, train, and evaluate a simple neural network in TypeScript

**Exit criteria**: A small MLP trains to convergence on a toy problem.

---

## Phase 4: nanoGPT

**Goal**: A working GPT that trains on Shakespeare and generates text.

**What this phase covers**:

### 4a. Tokenizer

- Character-level tokenizer (simplest, for initial training)
- BPE tokenizer (for real training — port tiktoken or implement from scratch)

### 4b. Data pipeline

- Download and preprocess Shakespeare (or other small corpus)
- Batched data loading with sequence windowing
- Train/val split

### 4c. Model architecture

```
Token Embedding + Position Embedding
         ↓
    N × Transformer Block:
        ├── LayerNorm → Multi-Head Causal Self-Attention → Residual
        └── LayerNorm → MLP (Linear → GELU → Linear) → Residual
         ↓
    LayerNorm → Linear (to vocab logits)
```

- Configurable: n_layers, n_heads, n_embd, block_size, vocab_size, dropout
- Weight tying (embedding weights = output projection weights)
- Causal attention mask

### 4d. Training

- Training loop with gradient accumulation
- Loss logging, learning rate scheduling (warmup + cosine decay)
- Periodic validation loss evaluation
- Checkpoint saving/loading (safetensors or JSON)

### 4e. Generation

- Autoregressive text generation
- Temperature and top-k/top-p sampling
- Interactive generation mode

### 4f. Configurations

- `gpt2-tiny`: ~1M params (for fast iteration, trains in minutes)
- `gpt2-small`: ~124M params (the real nanoGPT target)

**Deliverables**:

- `packages/nanogpt/` — complete GPT implementation
- Trains on Shakespeare, generates coherent text
- Example scripts in `examples/`

**Exit criteria**: GPT-2 tiny trains to <1.5 loss on Shakespeare in under 30 minutes on M4 Max. Generated text is recognizably English and vaguely Shakespearean.

---

## Phase 5: Polish and Publish

**Goal**: Make this usable and educational for others.

**What this phase covers**:

- npm packages: `mlx-ts`, `nanogpt-ts`
- API documentation (TypeDoc)
- Educational walkthrough: "Building GPT from scratch in TypeScript"
- Example: fine-tune a small model
- Benchmarks: mlx-ts vs Python MLX for common operations
- CI: GitHub Actions (macOS Apple Silicon runners)

---

## Phase 6: mlx-vlm-ts (Future)

**Goal**: Vision-language model support using mlx-ts.

**Scope TBD** — depends on what emerges from Phases 1-5.

Potential directions:

- Port mlx-vlm to TypeScript
- Support Gemma 4 multimodal models
- Image preprocessing pipeline in TypeScript

---

## Phase 7: Repository Separation

**Goal**: Split the monorepo into independent repositories, each with its own identity, releases, and community.

**Why**: During development, a monorepo is the right call — tight coupling, atomic changes, one CI pipeline. But the packages we're building serve fundamentally different audiences at different layers:

| Layer | Package | Audience | Lifecycle |
|-------|---------|----------|-----------|
| **Primitive** | `mlx-ts` | Any TS developer doing ML on Apple Silicon | Slow, stable, foundational |
| **Framework** | `nanogpt-ts` | Educators, learners, GPT experimenters | Moderate, educational |
| **Application** | `mlx-vlm-ts` | VLM researchers and builders | Fast-moving, experimental |

A diffusion model developer should be able to depend on `mlx-ts` without knowing nanoGPT exists. A learner should be able to study nanoGPT without wading through FFI binding code. Separate repos make this possible.

**What this phase covers**:

### 7a. Stabilize public APIs
- Freeze mlx-ts public API surface (semver 1.0)
- Document breaking change policy
- Ensure nanogpt depends on published mlx-ts, not workspace link

### 7b. Extract repositories
- `mlx-ts` → own GitHub repo with:
  - Independent CI/CD (build, test, publish to npm)
  - Dedicated README, contributing guide, issue templates
  - npm package `mlx-ts`
- `nanogpt-ts` → own GitHub repo with:
  - Depends on published `mlx-ts` from npm
  - Dedicated README with educational focus
  - Example scripts and walkthroughs
- Future: `mlx-vlm-ts` → own repo when ready

### 7c. Cross-repo development workflow
- Contribution guide: how to develop mlx-ts and nanogpt together (npm link / workspace override)
- CI: nanogpt tests run against mlx-ts main branch (catch breakage early)
- Release coordination: mlx-ts releases trigger nanogpt compatibility checks

### 7d. Archive monorepo
- This monorepo becomes a historical reference or redirects to individual repos
- Alternatively: keep as an umbrella with git submodules (lower maintenance)

**Exit criteria**: Each package lives in its own repo, publishes independently to npm, and has its own CI. Cross-repo development workflow is documented and tested.

**When to trigger this phase**: When mlx-ts API has been stable for at least 2 weeks and nanogpt trains successfully using only the published npm package.

---

## Agent Assignments

See [docs/agentic-loop.md](./docs/agentic-loop.md) for the full engineering workflow.

| Role | Primary Responsibility | Strengths |
| ---- | ---------------------- | --------- |
| Planning / Architecture Agent | Architecture, planning, review, debugging | Deep reasoning, context management |
| Implementation Agent | Bulk implementation, mechanical porting | Fast parallel execution, large codegen |
| Independent Reviewer | Alternative review, research, validation | Fresh perspective from implementation author |
| Human (Nomi) | Decision-making, direction, acceptance | Domain authority, final approval |

The exact model or tool used for each role may change over time. The workflow matters more than the brand name.


## Principles

1. **Correctness over speed** — Get it right, then make it fast
2. **Document as we go** — Every design decision is recorded
3. **Test everything** — No code ships without tests
4. **Agent review** — No agent's output merges without review by a different agent
5. **Incremental delivery** — Each phase produces something that works
6. **Education first** — Code clarity trumps cleverness

