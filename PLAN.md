# Build Plan

## Vision

Build **mlxts** (`@mlxts/*`) — a complete, GPU-accelerated ML ecosystem in TypeScript for Apple Silicon, powered by MLX.

TypeScript-native MLX stack for training, fine-tuning, serving, and evaluating ML models on Apple Silicon. Designed for human readability, agentic development, and modular extensibility.

**Competitive context:** Transformers.js v4 provides inference on WebGPU at ~60 tok/s on M4. Our differentiator is training, fine-tuning, and native MLX performance — not just inference.

**Prior art:** @frost-beta/mlx provides Node.js MLX bindings. mlxts is Bun-native, training-capable, and aims for a complete ecosystem.

**Non-goal:** Performance parity with Python at every layer. The priority is correctness, clarity, and developer experience. Performance follows from correct abstractions — MLX and Metal do the heavy lifting.

### Planning Documents

This file is the roadmap. Detailed designs live in separate docs:

| Document | Purpose |
|----------|---------|
| [docs/ecosystem-structure.md](./docs/ecosystem-structure.md) | Complete package map, repo layout, migration table |
| [docs/future-backends.md](./docs/future-backends.md) | Multi-backend vision (WebGPU, CUDA) — not part of current plan |
| [docs/python-equivalence-map.md](./docs/python-equivalence-map.md) | Python ML ecosystem → mlxts mapping |
| [docs/gates-and-milestones.md](./docs/gates-and-milestones.md) | Exit criteria for every phase |
| [docs/inference-optimizations.md](./docs/inference-optimizations.md) | Inference optimization catalog — techniques, papers, reference implementations |
| [docs/serving-runtime-strategy.md](./docs/serving-runtime-strategy.md) | Serving/runtime strategy boundaries and future backend flag posture |
| [docs/architecture.md](./docs/architecture.md) | System architecture and layer responsibilities |
| [docs/code-standards.md](./docs/code-standards.md) | Code quality, naming, structure, testing standards |
| [docs/agentic-loop.md](./docs/agentic-loop.md) | Multi-agent engineering workflow |
| [`.agents/skills/axi/SKILL.md`](./.agents/skills/axi/SKILL.md) | Agent-facing CLI output contract |

## Roadmap to Phase 10 Completion

The current roadmap runs through Phase 10 as the active product horizon. Phase
numbers still describe dependency order, but they are not a single serial queue:
Phases 8, 9, and 10 fan out after the Phase 7 architecture base. Work advances
by narrow tranches with review artifacts and gates, not by bundling multiple
product areas into one commit.

### Product-area order

1. **Phase 7 closeout: model architecture truth.** Keep dense decoder families,
   chat-template behavior, MoE text families, tokenizer parity, and generation
   performance evidence coherent before widening the product surface.
2. **Phase 8: fine-tuning and alignment.** Harden LoRA, QLoRA, SFT, DPO,
   dataset preparation, official-checkpoint proofs, report verification, and
   future training CLI surfaces without turning `@mlxts/train` into a black-box
   framework.
3. **Phase 9: serving and quantized inference.** Complete quantization,
   cache-backend evolution, continuous scheduling, tool/structured protocol
   support, dynamic model pools, embeddings, and serving regression ladders.
4. **Phase 9.5: AXI and agent-operated CLI surfaces.** Every agent-facing CLI
   becomes predictable for shell-driving agents: compact TOON-shaped stdout,
   structured stdout errors, clear exit codes, no prompts in non-TTY paths, and
   diagnostics/progress off the consumable stdout channel. Package-owned
   binaries migrate before an umbrella `@mlxts/cli` claims the surface.
5. **Phase 10 research spike.** Before each new modality or generation
   paradigm, audit canonical references first: Hugging Face Transformers,
   Diffusers, MLX examples, `mlx-lm`, and the relevant MLX runtime sources.
6. **Phase 10a: multimodal understanding.** Expand `@mlxts/transformers` with
   vision/audio encoders, VLM wrappers, encoder-decoder families, prepared
   prompt contracts, and serving routes that preserve model-native media
   semantics.
7. **Phase 10b: diffusion and flow generation.** Create `@mlxts/diffusion` for
   image, video, and audio generation with package-owned configs, schedulers,
   VAE/backbone loading, conditioning, sampling, and examples.
8. **Phase 10 completion fence.** A Phase 10 claim requires real checkpoint
   proofs, examples/workbooks, package docs, AXI-shaped finite commands where a
   CLI exists, runtime review artifacts for hot paths, and full validation.

## Design Philosophy

This is a **product**, not a project. The difference matters.

Most open source ML repositories are built by researchers for researchers. Code ships fast, documentation is an afterthought, APIs break between releases, features land without migration guides. The codebase reflects the author's mental model, not the reader's learning path.

We reject that. Our principles:

1. **The first-time user is the primary audience.** Every API, every file, every error message is designed for someone encountering it fresh. If a developer can't understand what a module does within 30 seconds of opening it, we've failed.

2. **Original thinking from first principles.** Official MLX and mlx-c sources are the primary truth. Comparative research from other high-signal repos is allowed when it improves correctness, semantics, or performance understanding, but we do not cargo-cult someone else's API or operational compromises.

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

**Status**: Complete

**Goal**: TypeScript can create arrays, run operations, and evaluate results on the GPU.

**What this phase covers**:

### 1a. Build infrastructure

- CMakeLists.txt to build mlx-c from source (fetches MLX automatically via FetchContent)
- Bun build script that compiles mlx-c and produces libmlxc.dylib
- CI-like validation script (typecheck + test)

### 1b. Bun FFI bindings (`src/core/ffi/`)

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
4. `bun run check:coverage` passes for the production packages:
   - `mlx-ts` at `95%` lines and `90%` functions
   - `nanogpt` at `90%` lines and `85%` functions
5. FFI symbol declarations in `src/core/ffi/symbols.ts` are verified against mlx-c v0.6.0 headers
6. No type assertions (`as`, `!`) exist outside the FFI boundary package (`src/core/ffi/`)
7. All native handle temporaries use `try/finally` for cleanup
8. Explicit-dtype array creation and all scalar dtype paths are covered by direct unit tests
9. Smoke test works: `mx.ones([3,3])` → `mx.matmul(a,a)` → `mx.eval(b)` → `b.toList()` returns `[[3,3,3],[3,3,3],[3,3,3]]`

---

## Phase 2: Autograd

**Status**: Complete

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

**Status**: Complete

**Goal**: A PyTorch-like nn.Module system in TypeScript, built on mlx-ts core.

**What this phase covers**:

### 3a. Module system

- Base `Module` class with property scanning (no registration), `#` private fields
- `parameters()`, `trainableParameters()`, `update()` (recursive partial merge)
- `freeze()`/`unfreeze()`, `train()`/`eval()` with recursive propagation
- `nn.valueAndGrad` bridge: flatten params → Phase 2 autograd → unflatten grads
- Tree utilities: `treeFlatten`, `treeUnflatten`, `treeMap`, `treeLeaves`

### 3b. Layers

- `Linear` — fully connected layer with optional bias
- `Embedding` — token/position embeddings with `asLinear()` for weight tying
- `LayerNorm` — layer normalization (composed from core ops)
- `Dropout` — training regularization with eval-mode bypass

### 3c. Activations

- `gelu`, `relu`, `silu` — free functions (not Module subclasses)

### 3d. Losses

- `crossEntropy` — classification loss with integer target validation
- `mse` — mean squared error

### 3e. Optimizers

- `SGD` with momentum and weight decay
- `AdamW` — Adam with decoupled weight decay
- `Adam` — zero-weight-decay wrapper around AdamW
- Failure-safe update with path-keyed gradient lookup

### 3f. Core ops enhancements

- `array()` accepts `number` → creates scalar (0-dim) MxArray
- Binary ops accept `MxArray | number` operands (scalar coercion)
- `takeAxis` op for Embedding (via `mlx_take_axis` FFI symbol)

### 3g. Tests

- 219 tests across 19 files
- Each layer: forward pass shape and value correctness, gradient flow
- Each optimizer: parameter update matches hand calculation
- Constructor validation (positive dimensions, valid dropout p)
- Integer dtype validation for Embedding and crossEntropy
- End-to-end: 2-class MLP on XOR converges with crossEntropy (deterministic)

**Deliverables**:

- `packages/mlx-ts/src/nn/` and `packages/mlx-ts/src/optimizers/`
- Can define, train, and evaluate a neural network in TypeScript

**Exit criteria** — Phase 3 is complete when all of the following are true:
1. `bun run validate` passes (typecheck + lint + assertions + coverage)
2. MLP trains to convergence on XOR with crossEntropy (loss < 0.05, 500 steps)
3. Predictions match XOR truth table (argmax of 2-class logits)
4. 97.91% line coverage, 95.89% function coverage

**Explicitly deferred to later phases**:
- `loadWeights()`, `saveWeights()` → Phase 5 (serialization)
- `RMSNorm` → Phase 4 (when modern architectures need it)
- `Conv1d`, `Conv2d` → when vision/audio models are targeted
- Learning rate schedules → Phase 4 (cosine annealing)
- Module[] (layer list) support → Phase 4 (transformer block arrays)
- Sigmoid, Tanh, Softmax as nn.Module → already available as core op functions

---

## Phase 4: nanoGPT

**Status**: Complete.

**Goal**: A working GPT that trains on Shakespeare and generates text.

**What this phase covers**:

### 4a. Tokenizer

- Character-level tokenizer
- Canonical serialized vocab captured in checkpoints

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
- Fused fast attention when MLX exposes the right primitive
- Optional gradient checkpointing at the transformer-block level for memory-sensitive runs

### 4d. Training

- Training loop with gradient accumulation
- Loss logging, learning rate scheduling (warmup + cosine decay)
- Periodic validation loss evaluation
- Canonical checkpoint saving/loading (`manifest.json` + `tensors.bin`)
- Supervised long-run management with explicit status, stop, cancel, and resume flows

### 4e. Generation

- Autoregressive text generation
- Temperature and greedy / categorical sampling
- Interactive CLI generation mode

### 4f. Configurations

- `gpt-tiny`: ~10.8M params with char-level vocab (the fast acceptance model)
- `gpt-small`: GPT-2-small-sized architecture variant with runtime vocab from the tokenizer

**Deliverables**:

- `examples/nanogpt/` — complete GPT example implementation
- Trains on Shakespeare, generates coherent text
- Canonical CLI with `train` / `generate`
- Memory benchmark and supervised soak surfaces before loss-targeted acceptance
- Acceptance scripts: `cd examples/nanogpt && bun run acceptance:gpt-tiny` and `cd examples/nanogpt && bun run acceptance:gpt-small`

**Exit criteria**:
- `bun run validate` passes
- `gpt-tiny` trains to <1.8 validation loss on Shakespeare in an explicit acceptance run
- `gpt-small` also has an explicit loss-targeted acceptance run
- Long unattended runs use the supervised `cd examples/nanogpt && bun run manager ...` surface rather than one-off scripts
- Long-run acceptance follows a soak ladder (`50 → 250 → 1000 → 5000`) rather than jumping straight to overnight runs
- Runtime-sensitive diffs leave a review artifact under `docs/reviews/` and pass `bun run check:runtime-review`
- Generated text is recognizably English and vaguely Shakespearean

**Post-Phase-4 posture**: Phase 4 completion triggers the ecosystem restructure. The proven code from `mlx-ts` and `nanogpt` becomes the foundation for the `@mlxts/*` package family. nanoGPT now lives conceptually as `examples/nanogpt/`: a committed in-repo example and regression surface rather than a package.

---

## Phase 5: Ecosystem Restructure

**Status**: Complete.

**Goal**: Adopt the `mlxts` package identity inside the monorepo, extract the canonical `@mlxts/*` packages, and make the repo truthful about the package-first transition. `examples/nanogpt/` is part of that truthful contract: a committed example surface, not a package deliverable.

See [docs/ecosystem-structure.md](./docs/ecosystem-structure.md) for the complete package map and migration table.

**What this phase covers**:

### 5a. Identity and repo contract

- Root workspace adopts the `mlxts` identity
- Docs, validation gates, and runtime-review rules reflect the new package layout
- External repo rename and npm publishing can trail the internal package extraction

### 5b. Extract canonical packages

Extract these packages from the existing codebase (see migration table in ecosystem-structure.md):

| Package | Source | What moves |
|---------|--------|-----------|
| `@mlxts/core` | Legacy MLX monolith | MxArray, FFI, ops, transforms, fast fused ops, device, memory, random, I/O, dtype, tree utils, shape utils, error types |
| `@mlxts/nn` | Legacy nn layer | Module, Linear, Embedding, LayerNorm, Dropout, activations, losses |
| `@mlxts/optimizers` | Legacy optimizer layer | Adam, AdamW, SGD, optimizer base, LR schedules |
| `@mlxts/train` | Former generic pieces of `examples/nanogpt/src/train.ts` and `checkpoint.ts` | Training loop, checkpoint, gradient utilities, typed checkpoint metadata, reusable step orchestration |
| `@mlxts/data` | Former `examples/nanogpt/src/data.ts` | Text data loading, batching |
| `@mlxts/tokenizers` | Former `examples/nanogpt/src/tokenizer.ts` | Character tokenizer (BPE comes in Phase 7) |

This extraction is now real, not hypothetical:

- `@mlxts/core`, `@mlxts/nn`, `@mlxts/optimizers`, `@mlxts/train`, `@mlxts/data`, and `@mlxts/tokenizers` exist as workspace packages
- `packages/core` owns the native MLX build and the canonical FFI/runtime surface
- `examples/nanogpt` now consumes the extracted packages directly and is the thin GPT-specific example/regression surface

### 5c. Package-first validation posture

- `examples/nanogpt` is a committed in-repo example, not a publish target
- `@mlxts/train`, `@mlxts/data`, and `@mlxts/tokenizers` are the sole implementations of generic training/data/tokenizer behavior; `examples/nanogpt` only wraps them for GPT-specific flows
- `bun run check:file-lines` enforces the 500-line cap for active production source, including the committed example surface
- Coverage, runtime-review, type-assertion, and tensor-lifetime gates recognize the extracted package layout
- `docs/reviews/phase-5-restructure.md` records the runtime-sensitive package extraction work
- Phase 5 success is defined by clean package surfaces and truthful repo docs, not by forcing a premature in-repo example rewrite

### 5d. Validation

- `bun run validate` passes across entire monorepo
- Each extracted package typechecks and tests independently
- `bun run check:file-lines` passes
- Runtime review stays in place for the extracted package surfaces
- Transitional shims are explicitly documented, while `examples/nanogpt` is documented as an intentional example surface

### 5e. What comes next

- Keep hardening the extracted packages and their docs while preserving a thin, teachable `examples/nanogpt` surface
- Keep the nanoGPT example useful as a regression path without letting it accrete package-owned abstractions
- Consider a broader dedicated examples repo later if the example portfolio outgrows the monorepo, but `examples/nanogpt` is now an intentional in-repo surface

**Exit criteria**:
- See [gates-and-milestones.md](./docs/gates-and-milestones.md#phase-5-ecosystem-restructure)
- Canonical `@mlxts/*` packages exist and pass their package-local tests
- `bun run validate` and `bun run check:file-lines` pass
- Runtime review artifacts and top-level docs accurately describe the package-first state
- `packages/mlx-ts` is clearly marked as transitional, while `examples/nanogpt` is clearly documented as an example surface rather than a package

---

## Phase 6: Publish Core Packages

**Status**: Complete.

**Goal**: Make the public packages fully repo-ready for first npm publish. TypeDoc, CI, and package ergonomics land in-repo even if the actual publish step is still manual.

**What this phase covers**:

### 6a. npm publishing

- Prepare `@mlxts/core`, `@mlxts/nn`, `@mlxts/optimizers`, `@mlxts/train`, `@mlxts/data`, and `@mlxts/tokenizers` for publish with correct manifests, dist output, and dry-run packaging
- Semver versioning with changesets
- `package.json` `exports` field for clean import paths
- Build step for type declarations

### 6b. Documentation

- TypeDoc API docs generate cleanly in-repo and are ready to host
- Quick-start guide: "Hello Tensor in 10 lines"
- README with clear value prop, install instructions, examples

### 6c. Educational content (can trail npm publish)

- Educational walkthrough: "Building GPT from scratch in TypeScript"
- Benchmarks: mlxts vs Python MLX for core ops (matmul, softmax, attention, training step)
- Published results with methodology

### 6d. CI

- GitHub Actions on Apple Silicon runners (M-series)
- Full `bun run validate` on every push
- Publish workflow triggered on tagged releases

### 6e. Community

- CONTRIBUTING.md
- Issue templates
- Discussion board or Discord

**Exit criteria**: See [gates-and-milestones.md](./docs/gates-and-milestones.md#phase-6-publish-core-packages).

---

## Phase 6.5: Modern Transformer Primitives

**Goal**: The nn and ops layers have everything modern architectures need.

**Status**: Complete. `@mlxts/core` and `@mlxts/nn` now expose RMSNorm, RoPE,
Grouped-Query Attention, SwiGLU, `sort` / `topk` / `repeat` / `tile`, exact
float16/bfloat16 safetensors I/O, and low-level MLX quantize/dequantize
bindings.

**What this phase covers**:

### 6.5a. Fast ops from mlx-c

- Bind `mlx_fast_rms_norm`, `mlx_fast_rope` from mlx-c
- Implement `RMSNorm` and `RoPE` (Rotary Position Embeddings) nn modules

### 6.5b. Modern attention and activations

- Grouped-Query Attention (GQA) module
- SwiGLU activation

### 6.5c. Additional ops

- `sort`, `topk`, `repeat`/`tile`

### 6.5d. Dtype and I/O

- float16/bfloat16 safetensors I/O support

### 6.5e. Quantization bindings

- Bind `mlx_quantize`/`mlx_dequantize` from mlx-c

**Exit criteria**:
- RMSNorm + RoPE + GQA modules tested
- float16 I/O works
- quantize/dequantize works

**Note**: Must complete before Phase 7 begins.

---

## Phase 7: Model Architectures

**Goal**: Load and run pretrained LLaMA, Mistral, Gemma, Phi. The `@mlxts/transformers` package.

See [docs/python-equivalence-map.md](./docs/python-equivalence-map.md) for the full Python → mlxts mapping.

**What this phase covers**:

### 7a. Hub integration (official `@huggingface/hub` + transformers pretrained loading)

- Official Hugging Face JS client for model snapshot resolution and cache reuse
- Local pretrained-source resolution inside `@mlxts/transformers`
- safetensors reading stays native via `@mlxts/core`
- `config.json`, tokenizer sidecars, `chat_template.jinja`, and processor config inspection
- Structured loader progress events for future CLI/TUI work
- HF checkpoint name mapping stays local to the transformer family implementations

### 7b. Tokenizers (`@mlxts/tokenizers` expansion)

- ByteLevel BPE from `tokenizer.json` (covers LLaMA/Mistral/GPT-2). Full 5-stage HF tokenizer pipeline is future work.
- Batch encode/decode with offset tracking
- Special-token-aware prompt encoding from tokenizer sidecars and `added_tokens`, so control markers emitted by chat templates encode as their intended token IDs instead of ordinary text fragments
- Conformance fixtures against Hugging Face tokenizers for every supported chat-capable family: canonical rendered prompts must encode to the same token IDs and decode losslessly with `skipSpecialTokens: false`

### 7c. Model architectures (`@mlxts/transformers`)

- Config-driven architecture dispatch: `model_type` in config.json → model class
- KV cache for efficient autoregressive generation
- Generation utilities: temperature, top-k, top-p, min-p, repetition penalty
- Model families (each ~200-400 lines):

LLaMA first, done right, then expand:

| Family | Priority | Why |
|--------|----------|-----|
| LLaMA | Highest | Most popular open model family — get this right first |
| Mistral | High | Shares LLaMA architecture, efficient, widely deployed |
| Phi | Medium | Small, fast, great for local |
| Gemma | Medium | Google's open models |
| Qwen | Medium | Strong multilingual |
| GPT-2 | Already done | From nanoGPT |

**Scope: dense text models only.** MoE variants (Mixtral, DeepSeek) are deferred
to Phase 7f. The architecture accommodates this — the decoder block's MLP slot
is a swappable `Module` property, and
`FamilyRegistration.sanitizeWeight()` handles per-family weight name
translation including expert weight stacking. The `CausalLM` contract does not
change for MoE because MoE is a block-internal optimization, not a different
model contract. See [design-reasoning.md § Contract Boundaries](./docs/design-reasoning.md#contract-boundaries)
for the rationale.

- `AutoModel.fromPretrained(modelId)` auto-dispatch
- `AutoTokenizer.fromPretrained(modelId)`

### 7d. Interaction profiles and prompt compilation

Before Phase 9 serving, chat behavior must be a first-class contract inside
`@mlxts/transformers`, not ad hoc logic in examples or future HTTP handlers.

- `InteractionProfile`-style surface in `@mlxts/transformers` for checkpoint-aware prompting behavior
- Owns chat-template selection, special-token handling, checkpoint generation defaults, and full EOS / end-of-turn stop sets
- Prompt compiler that turns a normalized request into:
  - rendered prompt text
  - prompt token IDs
  - resolved generation defaults
  - resolved stop-token behavior
- Hugging Face parity coverage for every supported chat-capable family:
  - rendered prompt text matches `apply_chat_template(..., tokenize=False)`
  - encoded prompt token IDs match Hugging Face tokenizer output
  - single-user, system+user, and multi-turn assistant-history prompts are all covered
- The interactive `examples/chat/` surface is the first real consumer of this path and should remain thin over the shared compiler

**Why now:** This is the contract that future serving depends on. If prompt
rendering, special tokens, and stop behavior diverge by endpoint or by example,
the server will return incorrect model behavior even when the model math is
correct.

### 7e. Examples

- `examples/chat/` — interactive chat with a local supported decoder model
- Broader future example surfaces can adopt `@mlxts/transformers` where it improves their design, but `examples/nanogpt` remains an in-repo example rather than a package concern

**Exit criteria**: See [gates-and-milestones.md](./docs/gates-and-milestones.md#phase-7-model-architectures).

### 7f. MoE text architectures (follows Phase 7 dense completion)

MoE (Mixture of Experts) models use the same `CausalLM` contract as dense models. The difference is entirely inside the decoder block: instead of a single dense MLP, an MoE block routes tokens through a subset of expert MLPs via a learned router.

**What changes:**

- `SwitchLinear` primitive in `@mlxts/nn` — batched expert dispatch via `mx.gatherMm`, holds all experts in a single weight of shape `(numExperts, outDims, inDims)`
- MoE MLP block variant in `families/` — router + top-K expert selection + weighted combination
- Expert weight stacking in `sanitizeWeight()` — HuggingFace stores per-expert tensors individually, MLX stacks them for efficient batched matmul
- New family registrations: Mixtral, potentially DeepSeek-v2, OLMoE

**What stays the same:**

- `CausalLM` interface — `forward(inputIds, options?) → logits`
- `TransformerCache` — KV cache is per-attention-layer; MoE only affects the FFN sublayer
- Generation pipeline — `generateStep()`, `generateTokens()`, `generateText()`
- Weight loading pipeline — same `iterateSafetensorWeights` → `sanitizeWeight` → `assignWeightPath` flow
- LoRA and quantization — work the same way; LoRA targets `SwitchLinear` via type dispatch, quantization uses `SwitchLinear.toQuantized()`

**Current proof state:** Gemma 4 A4B MoE loads and generates from the cached
`mlx-community/gemma-4-26b-a4b-it-4bit` checkpoint, including mixed-quant expert
loading, continuous serving, and coherent chat-template generation. Qwen A3B
split-quantized MoE also loads from the cached
`unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit` checkpoint: direct `128x128` decode
reported `generation_tps=89.954`, `evals_per_token=1.00`, and flat active
memory (`20.816 GB` start/end); serve streamed `128x32@1` through
`continuous:eligible` with `mean_post_ttft_completion_tps=79.300` and
`active_delta=0.004 GB`. Mixtral remains future family registration work.

**Exit criteria**: Real MoE checkpoints load through the unchanged `CausalLM`
contract, generate coherent text, and pass family-appropriate forward/decode
parity evidence against upstream MLX references before publishable parity
claims. Gemma 4 A4B and Qwen A3B are proven loading/decode/serve targets;
Mixtral remains a separate target-family proof.

### 7g. Performance observability (follows Phase 7 performance optimization)

Generation performance must be measurable, comparable, and regression-protected. This phase builds the infrastructure that keeps performance visible going forward — it does not include the performance fixes themselves (those are part of completing 7c).

**What this phase covers:**

1. **Synthetic throughput benchmark** (`packages/transformers/scripts/benchmark-generation.ts`) — synthetic-prompt generation benchmark over real cached transformer checkpoints (no tokenization, no network in the benchmark itself). Measures prefill tok/s, decode tok/s, peak memory, and eval-count-per-token. Runs warmup + N trials and reports per-trial numbers plus averages. This is the low-level throughput canary.

2. **Parity benchmark** (`packages/transformers/scripts/benchmark-generation-parity.ts`) — MLX-LM-comparison benchmark over the same real cached checkpoints and token counts. Includes the reference-style decode work we care about for shipping claims and records the paired MLX-LM reference numbers alongside the mlxts baselines.

3. **Benchmark commands** (`bun run bench:generation`, `bun run bench:generation:parity`) — run the benchmarks, compare results against recorded baselines in `benchmarks/baselines.json`, and warn (do not fail) on >2x regression. Reports numbers to stdout.

4. **Metal trace integration** — `--metal-trace` flag on both benchmark surfaces that wraps execution in `startMetalCapture()` / `stopMetalCapture()` (already bound in `@mlxts/core`) for Instruments analysis. Zero overhead when not used.

5. **Performance section in runtime review** — the review artifact for a hot-path diff must include the benchmark numbers that justify the change. The canonical evidence now includes both synthetic throughput and parity measurements when the change affects generation behavior.

6. **Baseline file** (`benchmarks/baselines.json`) — recorded tok/s for the canonical real-model benchmark targets (for example Llama 3.2 1B, Gemma 3 1B, Phi-4 mini) for both synthetic and parity modes, including the eval-count canary and paired MLX-LM reference numbers for parity targets. Updated explicitly when intentional performance-affecting changes land.

**Design principle:** Performance is an observable, not a review opinion. Don't ask "did you think about performance?" — ask "what do the numbers say?" See [runtime-safety.md § Generation Performance](./docs/runtime-safety.md#generation-performance).

**Profiling tools available (no code changes to hot path required):**
- Metal System Trace via Instruments.app + `startMetalCapture()` / `stopMetalCapture()`
- DTrace probes on mlx-c dylib calls (e.g., trace every `mlx_eval` with timing)
- MLX memory telemetry: `getActiveMemoryBytes()`, `getPeakMemoryBytes()`, `getCacheMemoryBytes()` — already bound in core

**Exit criteria**: `bun run bench:generation` and `bun run bench:generation:parity` both run and report numbers. Baselines are recorded. A diff that makes decode 2x slower is caught by the benchmark comparison. The review gate requires performance numbers for hot-path diffs.

**Near-term sequencing note:** Once the dense Phase 7 base is stable, the next
implementation priority is not "whatever phase number comes next." The priority
order is:

1. Official-checkpoint quantization proofs and long-context evidence
2. MoE text architectures
3. Minimal serving on the shared request / prompt-compiler path
4. AXI-shaped agent-operated CLI surfaces for the package-owned binaries that
   already exist
5. Phase 10 multimodal and diffusion model families
6. Deeper training orchestration ergonomics

Training remains a first-class product surface throughout. The deferral is
about orchestration ergonomics, not about deprioritizing fine-tuning or
alignment correctness.

---

## Phase 8: Fine-Tuning

**Goal**: LoRA fine-tuning, DPO alignment, dataset loading. A TS developer can fine-tune a model on their own data.

**What this phase covers**:

This phase establishes the canonical fine-tuning packages, real-data proof
surfaces, and regression expectations now. More opinionated training
orchestration ergonomics — policy-driven checkpointing, evaluation hooks,
artifact sinks, and higher-level composition helpers — are intentionally a
follow-on after official-model quantization proofs, long-context evidence, MoE,
and minimal serving are in place.

### 8a. LoRA (`@mlxts/lora`)

- Low-rank adapter injection for any `Linear` layer
- `applyLoRA(model, config)` — wrap target layers
- `mergeLoRA(model)` — merge adapters back into weights (zero inference overhead)
- `LoRAConfig` — target layers, rank, alpha, dropout
- QLoRA (quantized base model + fp16 adapters) — requires `@mlxts/quantize`

### 8b. Alignment (`@mlxts/align`)

- SFT trainer (supervised fine-tuning)
- DPO trainer (Direct Preference Optimization — simpler than PPO, better results)
- Preference pair data formatting
- Chat template support for instruction tuning
- Raw-chat preparation helpers that normalize, length-cap, and account for
  skipped supervision/preference rows before recipe loops run

### 8c. Data expansion (`@mlxts/data`)

- HuggingFace Datasets format loading
- Conversation/chat formatting
- Instruction tuning data collation

### 8d. Examples

- `examples/lora-finetune/` — fine-tune LLaMA on custom data, merge, generate

### 8e. Training orchestration and proof gates (deferred follow-on)

- Keep `@mlxts/train` explicit and model-agnostic. Future composition work
  should add small package-owned primitives such as train hooks, checkpoint
  policies, evaluation policies, artifact sinks, and `AsyncIterable`-style
  train events.
- As a first ergonomics step, move reusable dataset-level SFT/DPO evaluation
  and fixed-step recipe helpers into `@mlxts/align` so examples stay thin and
  package-owned, without turning `@mlxts/train` into a black-box trainer
  framework.
- Do **not** turn `@mlxts/train` into a black-box pipeline framework and do not
  add reactive framework dependencies such as RxJS or Effect to the core
  training layer.
- Recipe-specific orchestration belongs above the core loop: in `@mlxts/align`,
  example surfaces, and later CLI/application layers.
- The canonical training proof uses official checkpoints plus pinned real-data
  subsets:
  - `meta-llama/Llama-3.2-1B-Instruct` as the training anchor
  - `HuggingFaceH4/ultrachat_200k` subsets for LoRA / QLoRA / SFT
  - `HuggingFaceH4/ultrafeedback_binarized` subsets for DPO
- `@mlxts/align` recipe runners report per-step training loss traces alongside
  average loss; Phase 8 proof verifiers require trace length and mean
  consistency without requiring monotonic loss.
- This proof path should eventually become a CI-gated regression. Until then,
  keep `bun run examples/train-proof/index.ts` as the canonical Apple Silicon
  proof surface.
  If a model-architecture, quantization, tokenizer, or trainer change breaks
  LoRA / QLoRA / SFT / DPO on the canonical proof, the build should fail once
  we promote the gate.

**Exit criteria**: See [gates-and-milestones.md](./docs/gates-and-milestones.md#phase-8-fine-tuning).

---

## Phase 9: Inference and Serving

**Goal**: Production-quality inference server. Quantized inference. Any OpenAI-compatible client can connect. Architecturally flexible enough that new optimization techniques (from papers or upstream projects) slot in without rewiring the stack.

**Current status**: Phase 9 now contains both landed serving tranches and future production tranches. Treat the existing `@mlxts/serve` endpoint, streaming, admission, cancellation, multi-model loading, metrics, benchmark/regression harnesses, cache-generic continuous scheduler, and single-request prompt-prefix cache as real surfaces to preserve. The scheduler now covers eligible LLaMA-like, Qwen 3.6 text, and Gemma 3/4 layer-pattern requests for buffered and streaming generation, including model-native sampled defaults. Continuous routes share a model-level reservation controller for prompt tokens, completion tokens, aggregate total tokens, and estimated memory pressure. Prompt-prefix cache retention is operator-bounded and defaults to four retained prompt-boundary snapshots per served model so small divergent repeated-turn agent sessions stay warm. The real Qwen/Gemma serving matrix includes `@4` and `@8` streaming continuous guardrails with per-request TTFT, scheduler-queued-time, scheduler memory evidence, SSE lifecycle, and max-generation-batch evidence, plus mixed long-prefill/short-arrival fairness guardrails that assert short-request fairness and long-request prefill cadence separately from aggregate throughput. Keep paged/batch-native prefix cache, richer scheduler policy, multimodal serving, embeddings, broader Anthropic/Responses compatibility, and advanced optimization hooks as future work.

**Research basis**: Deep analysis of three MLX inference servers — Rapid-MLX (speed-focused), vLLM-MLX (foundational batching/paging), oMLX (production serving/memory management). These share lineage (vLLM-MLX → forks) but diverged into complementary specializations. Key findings are documented in [docs/inference-optimizations.md](./docs/inference-optimizations.md). Reference repos at `.reference/rapid-mlx`, `.reference/vllm-mlx`, `.reference/omlx`.

**Design principle — strategy-agnostic boundaries**: These reference projects study research papers and implement them as needed — MTP from one paper, speculative decoding from another, sparse prefill from a third. Our architecture must be flexible enough to swap strategies at each boundary: different cache backends, different decoding strategies, different scheduling algorithms. The interfaces must be stable even as the implementations behind them evolve. No technique should be hardwired in a way that prevents replacing it with a better one.

**Cross-phase runtime audit**: now that `.reference/mlx` and `.reference/mlx-c` are part of the local reference set, run a systematic capability audit against `packages/core/src/ffi/symbols.ts`, the runtime optimization matrix, and current native helper seams. The goal is to identify (a) MLX/MLX-C primitives we should bind directly, (b) places where compile should be retried before native work, and (c) the narrower cases that genuinely justify custom C or custom Metal helpers.

**What this phase covers**:

### 9a. Quantization (`@mlxts/quantize`)

- 4-bit and 8-bit quantization via MLX native `mx.quantize`/`mx.dequantize`
- GGUF tensor dequantization (15+ quant formats, moved from Phase 7)
- GGUF export (create GGUF files from mlxts models)
- Calibration dataset support for quantization quality
- Quantized inference at full speed
- Mixed-precision budget planner — per-layer bit allocation based on sensitivity measurement (informed by oMLX's oQ approach: temporarily quantize-dequantize each block against calibration data, rank by MSE, greedy allocate extra bits to sensitive layers). MoE experts stay at base bits, attention value projections and down projections get protection floors.
- Streaming quantizer — process one safetensor shard at a time, never load the full model. Essential for large models on constrained memory.

### 9b. KV cache architecture

KV cache is the critical infrastructure for both single-user generation and multi-user serving. The design must support progressive enhancement from simple to sophisticated.

**Core cache interface**: Design with an `isTrimmable()` discriminator from the start. Standard KV cache layers are trimmable (can roll back to any prefix length). Future hybrid RNN+attention architectures (Qwen3.5 DeltaNet, Mamba hybrids) have non-trimmable recurrent layers. The interface must accommodate both without widening the contract later.

**Progression**:

1. **Simple KV cache** (already in `@mlxts/transformers`) — one contiguous cache per sequence. Sufficient for single-user generation.
2. **Paged KV cache** — 64-token blocks with reference counting, doubly-linked free list (O(1) alloc/free), Copy-on-Write when shared blocks diverge. Chain hashing (block N's SHA-256 includes blocks 0..N-1) for O(1) prefix dedup without explicit trie traversal. This is the foundation for concurrent serving. (Reference: vLLM-MLX `paged_cache.py`)
3. **Prompt cache with LCP matching** — first landed as a small single-request message/chat prompt-prefix cache using family-owned snapshots, cache-hit telemetry, and OpenAI-compatible cache read/write accounting. It now has block-hash candidate narrowing, token-block metadata, and operator-bounded multi-entry retention for divergent repeated-turn agents while cache tensors still restore through family-owned snapshot/fork semantics. The next cache pass should widen this into memory-aware paged or tensor-block deduplicated reuse, batch-native integration, byte-budgeted eviction, and optional 4/8-bit quantization for stored entries. (Reference: Rapid-MLX `memory_cache.py`)
4. **SSD-persistent cache** (future) — serialize KV blocks to safetensors on disk, survive server restarts. Write-back RAM hot cache with LRU eviction to SSD. Background writer thread with pure-JS safetensors serializer (no MLX calls, thread-safe). ~2ms read per 10MB block on NVMe. (Reference: oMLX `paged_ssd_cache.py`)

**Cache backend must be pluggable** — the generation loop and scheduler interact with the cache through a stable interface. Swapping from simple → paged → SSD-backed should not require changing model code or the serving API.

**Cross-family requirement** — prefix cache is not complete until it is proven
across the major cache shapes we serve:

- LLaMA-like full KV: every layer retains growing keys/values and is the
  baseline for trim/copy/LCP semantics.
- Gemma 3/4 layer-pattern caches: sliding layers retain only their window while
  full layers retain the long prefix, so cache accounting must separate logical
  context length from retained tensor length.
- Qwen 3.5/3.6 hybrid caches: full-attention layers retain KV while
  linear-attention layers retain recurrent/conv state. Exact-continuation reuse
  and arbitrary LCP reuse are different capabilities, and non-trimmable state
  must stay visible in the contract.

The acceptance proof should include repeated chat-turn benchmarks with cache
read/write token reporting, lower second-turn TTFT, reduced suffix prefill
events, and no regression to continuous batching fairness for uncached requests.

The implementation seam should be family-owned prefix snapshots rather than a
serving-owned dump of cache tensors. `@mlxts/transformers` owns snapshot/fork
correctness for each cache family; `@mlxts/serve` owns longest-prefix matching,
admission, accounting, eviction, metrics, and OpenAI-compatible cache usage
reporting.

### 9c. Generation engine architecture

**Dual-strategy engine**: A single loaded model can serve through two strategies — a serial engine (maximum single-user throughput, supports speculative decoding) and a batched engine (concurrent users, continuous batching). Switch based on active request count: when requests >= threshold, use batched; switch back to serial only when idle. Saves memory by sharing one model instance. (Reference: Rapid-MLX HybridEngine)

**Chunked prefill with decode interleaving**: Break large prefills into configurable chunks. The current serving default is `512` tokens to favor heterogeneous request fairness; future operator strategy may expose larger throughput-oriented chunks such as 2048-8192 tokens. Between chunks, run one decode step for all active requests. This prevents decode starvation during long-context prefills. The chunk boundary is prefix-aware: if a request has a cached prefix, the first chunk aligns to that boundary for optimal cache capture. (Reference: all three servers implement this)

**Output streaming**: Per-request output collector with non-blocking put and smart merge when producer outpaces consumer. Maps directly to TypeScript async iterators. Pre-computed SSE envelope — template-compile the JSON structure once at request start, substitute only content per token. 20-30% CPU reduction in streaming overhead. (Reference: Rapid-MLX)

### 9d. Multi-model serving and memory management

**Engine pool**: Manage multiple loaded models with LRU eviction, model pinning (priority models never evicted), per-model TTL (idle timeout). Estimate model memory from safetensors file sizes + 25% KV headroom before loading. Landed source-backed lazy loading with idle TTL, pins, local model-root discovery, and an explicit `modelPressurePolicy` where `reject` preserves active requests and `shed_non_pinned` evicts idle models before aborting bounded active non-pinned request scopes one at a time; operators can tune the release wait with `modelPressureReleaseTimeoutMs` / `--model-pressure-release-timeout-ms`. (Reference: oMLX EnginePool)

**Dual-layer memory management**: Bookkeeping-based estimates for fast pre-load decisions (will this model fit?) + real `mx.get_active_memory()` polling as safety net. Do NOT use `mx.set_memory_limit()` — it causes alloc/free churn during model loading that triggers swap. When only one model is loaded and memory is critical: abort active requests but keep model loaded (frees KV cache so short-context requests can still be served). (Reference: oMLX ProcessMemoryEnforcer)

**Model discovery**: Auto-detect model type from `config.json` fields (`architectures`, `model_type`, presence of `vision_config`), not from model name strings. Two-level directory scan (org/model convention). Size estimation from safetensors file totals.

### 9e. Tool calling and structured output

**Parser registry**: Pluggable tool call parsers registered by model family. Auto-detection from model path via ordered `(regex, config)` list, first-match. Cascading "auto" parser tries multiple formats on unknown models. Auto-recovery for quantized model degradation (4-bit models emit broken tool calls as text after multiple rounds — detect and convert back to structured format). (Reference: Rapid-MLX, 17 parser formats)

**Reasoning separation**: Streaming state machine for `<think>` tag parsing with correction-on-finalize. Handles three scenarios: both tags present, implicit think (only closing tag), no tags. Separate `reasoning_content` field in SSE chunks. (Reference: Rapid-MLX reasoning parsers)

**Model auto-configuration**: Zero-flag model setup. Detect model family → apply optimal tool parser, reasoning parser, prefill chunk size, sampling defaults. Users run `mlxts serve <model>` and get the best configuration automatically. (Reference: Rapid-MLX `model_auto_config.py`)

### 9f. Serving (`@mlxts/serve`)

- Landed shared internal request model and prompt compiler inherited from Phase 7 interaction profiles; endpoint handlers are protocol adapters, not model-specific prompt logic
- Landed OpenAI-compatible API slices: `/v1/chat/completions`, `/v1/completions`, `/v1/responses` text support, `/v1/models`, `/health`, and `/info`
- Landed bounded Anthropic Messages-compatible text and base64 image-block support at `/v1/messages`, including Anthropic SSE event framing and reasoning/thinking separation
- Future OpenAI `/v1/embeddings` support maps into the same internal request model
- Future Anthropic documents/audio and broader content-block support map into the same internal request model; local image file IDs are an explicit root-scoped image transport policy, not a general files API
- `Bun.serve()` — no Express, no Node HTTP
- Server-sent events for token streaming, cancellation, long-context heartbeats, and cooperative prefill progress
- Landed cache-generic continuous batching for eligible LLaMA-like, Qwen 3.6 text, and Gemma 3/4 layer-pattern requests, including streaming and model-native sampled defaults
- Landed continuous scheduler budget admission with separate prompt, completion, aggregate total, and estimated memory caps
- Landed higher-concurrency Qwen/Gemma streaming guardrails for `@4` and `@8` continuous routes, including per-request TTFT and scheduler queue budgets
- Landed mixed long-prefill/short-arrival guardrails for Qwen/Gemma continuous streaming, including request-shape-specific short-request TTFT, scheduler queue, and stream-cadence budgets
- Future scheduler tranches cover stronger fairness policy beyond current queue/TTFT evidence, per-row decode state evidence, and higher-concurrency sampled proof
- Future cache tranches deepen prefix cache into batch-native/paged reuse,
  rotating/max-KV policy, quantized KV, and TurboQuant-style attention backends
- Future model loading/unloading without restart via engine pool
- Future per-model settings (sampling params, TTL, aliases) persisted to JSON
- Landed disconnect guard — monitor client disconnection and cancel generation

### 9g. Package-owned CLI expansion

The current CLI shape is package-owned binaries such as `mlxts-serve`,
`mlxts-agent`, example-local manager commands, and benchmark/report scripts.
Keep that shape until an umbrella `@mlxts/cli` has a stronger reason to exist
than centralizing names.

- `mlxts-serve` owns model serving, model-root discovery, startup validation,
  status/introspection, and endpoint-oriented benchmark entrypoints.
- `mlxts-agent` is an experimental local tool-loop harness over served models.
  Its finite paths stay AXI-shaped for testing, while PI-agent integration is
  the preferred future product-agent path once the core surfaces are stable.
- `examples/nanogpt` owns its manager, status, stop/resume, acceptance, and
  soak commands until those workflows become reusable package APIs.
- Training proof, Qwen image, future VLM, Whisper, text-to-image, quantization,
  and evaluation commands grow beside their backing packages/examples first.
- The future umbrella commands (`mlxts serve`, `mlxts quantize`,
  `mlxts download`, `mlxts train`, `mlxts eval`) wrap coherent package-owned
  surfaces instead of hiding inconsistent CLI contracts.

### 9h. Future optimization hooks

The architecture must accommodate these techniques without requiring them at launch. Each is documented in [docs/inference-optimizations.md](./docs/inference-optimizations.md) with source references and implementation requirements.

- **Multi-Token Prediction (MTP)**: 1.4x decode speedup for models with MTP heads. Requires model architecture support (`return_hidden`, `mtp_forward`).
- **Speculative decoding**: 1.5-2.3x decode via draft-model or prompt-lookup (n-gram index, no draft model). Requires O(1) cache trim.
- **Jump-forward decoding**: Logits bias toward structural tokens during tool calls. State machine tracks position in markup patterns. 2-5x faster structured output.
- **DeltaNet state snapshots**: Deep-copy non-trimmable RNN layers at prefix boundaries for hybrid architectures. ~0.1ms restore.
- **KV cache quantization**: Quantize stored KV entries to 4/8-bit, dequantize on fetch. Or run attention directly on quantized states via custom Metal kernel (TurboQuant approach).
- **SpecPrefill**: Attention-based sparse prefill — draft model scores token importance, target prefills only top-K%. Reduces TTFT on long prompts.
- **Cloud routing**: Pre-generation middleware. When uncached tokens exceed threshold, forward to external API.

**Exit criteria**: See [gates-and-milestones.md](./docs/gates-and-milestones.md#phase-9-inference-and-serving).

---

## Phase 9.5: Product-Agent Experience and AXI Hardening

**Goal**: Make every agent-operated CLI surface predictable, token-efficient,
and safe to drive through shell tools before Phase 10 broadens the product
surface.

Agent-driven operation is a product requirement, not a wrapper convenience.
Every CLI that agents drive through a shell follows the repo-local AXI contract
in [`.agents/skills/axi/SKILL.md`](./.agents/skills/axi/SKILL.md).

**What this phase covers**:

- Finite commands emit compact TOON-shaped stdout by default. JSON is an
  explicit compatibility/export mode when a command already promises it.
- Structured errors use stdout with actionable hints and stable exit codes:
  `0` for success/no-op, `1` for runtime errors, `2` for usage errors.
- Progress, diagnostics, debug logs, and long-running status lines stay off the
  consumable stdout channel.
- Non-TTY paths never prompt. Missing required values fail before calling any
  dependency.
- Long-running servers, REPLs, training managers, and benchmark harnesses expose
  structured status/report surfaces rather than pretending to be one-shot data
  commands.
- Package-owned binaries migrate first: `mlxts-serve`, experimental `mlxts-agent`,
  `examples/nanogpt` manager commands, training proof commands, Qwen image
  workbooks, benchmark/report commands, and future diffusion/multimodal
  inspection commands.
- `mlxts-serve discover --model-root <directory>` and
  `mlxts-serve status --base-url <url>` are landed finite AXI-shaped serve
  commands; broader serve startup logs remain a follow-up migration tranche.

**Exit criteria**: See
[gates-and-milestones.md](./docs/gates-and-milestones.md#phase-95-product-agent-experience-and-axi-hardening).

---

## Phase 10: Generative Media and Multimodal Understanding

**Goal**: On-device generative AI across modalities — image, video, and audio generation via diffusion models; multimodal understanding via transformer encoders and VLM composition.

**Design principle**: Packages are organized by **generation paradigm**, not by input/output modality. See [design-reasoning.md § Generation Paradigms](./docs/design-reasoning.md#generation-paradigms). There is no `@mlxts/vlm`, `@mlxts/audio`, or `@mlxts/multimodal` package — vision/audio encoders are transformer architectures (→ `@mlxts/transformers`), media generation uses diffusion/flow (→ `@mlxts/diffusion`).

**Research basis:** Phase 10 should be grounded in MLX-native reference work from
`.reference/mlx-examples` plus diffusion pipeline and checkpoint-structure
reference work from Hugging Face Diffusers. Refresh and audit
`.reference/diffusers` before each new Phase 10 family or modality tranche.

**Current status:** The first Qwen image-conditioned path is already real:
`@mlxts/transformers` owns Qwen media prompt preparation, `@mlxts/serve` owns
protocol media transport and scheduling, and `examples/qwen3_5-image` is the
direct workbook. `@mlxts/diffusion` now owns Stable Diffusion / SDXL package
surfaces through Diffusers snapshot source resolution, local manifest
inspection, scheduler/config loading, VAE/UNet construction and loading,
sampling, pipeline loading, and an AXI-shaped example proof command. Diffusers
Euler metadata now maps to sigma-space scheduler behavior for SD/SDXL leading
or trailing timestep spacing, `steps_offset`, and final sigma policy. It also
owns the first FLUX.1 package path: FlowMatch Euler scheduling, FLUX
transformer config/backbone/weights, FLUX VAE config/loading/decoding, latent
packing, sampling, and an AXI-shaped `examples/flux` proof command. The SDXL
and FLUX proof commands accept a local directory or Hugging Face model id and
resolve it to a concrete local snapshot before generation; remote resolution
selects component-local Diffusers weights and supports an explicit filename
variant such as `fp16` so proof runs do not download root monolith exports or
duplicate weight variants. The official SDXL base fp16 checkpoint has passed a
bounded real image proof through this path, loading the Hub snapshot, encoding
prompt conditioning, running denoising, and writing a BMP artifact. Official
`black-forest-labs/FLUX.1-schnell` has also passed a bounded real image proof
through `examples/flux`, including Hub snapshot resolution, FLUX transformer
and no-quant VAE loading, CLIP/T5 prompt conditioning, two denoise steps, and
BMP output. Base Qwen-Image runtime support has landed: snapshot recognition,
Qwen-Image transformer and 3D causal VAE config parsing/loading, FlowMatch
`shift_terminal` scheduling, latent packing, true-CFG denoising, Diffusers-style
Qwen2.5-VL prompt conditioning in `examples/qwen-image`, and an AXI-shaped
finite proof command. Official `Qwen/Qwen-Image-2512` has passed a bounded real
checkpoint proof through that path. Base Z-Image snapshot recognition and
config parsing has
also landed for current Diffusers `ZImagePipeline` snapshots, including
`ZImageTransformer2DModel` geometry, RoPE axes, padding constants, standard
AutoencoderKL metadata, and the Qwen3 text-encoder manifest boundary. Base
Z-Image tensor execution now has a package-owned foundation for dense
Diffusers snapshots: latent patching, Z RoPE, single-stream transformer blocks,
FlowMatch denoising over prepared Qwen caption embeddings, VAE decode layout,
transformer weight mapping/loading, and an AXI-shaped finite proof command in
`examples/z-image`. Official `Tongyi-MAI/Z-Image-Turbo` has passed a bounded
real checkpoint proof through that path. Phase 10 image proof commands now
write machine-checkable artifact evidence in JSON output, and
`examples/image-proof/verify-report.ts` verifies saved reports against the BMP
bytes on disk without rerunning generation. The first Whisper audio foundation
has landed in `@mlxts/transformers`: config parsing, feature-extractor config
parsing, Slaney mel filter creation, and channel-last log-mel feature
preparation over MLX-backed `@mlxts/core` primitives (`hanning`, `rfft`,
`asStrided`, `log10`). The first executable Whisper encoder-decoder foundation
now owns audio encoder blocks, text decoder blocks, cross-attention, tied
decoder-embedding logits projection, and Hugging Face safetensor loading.
`examples/whisper` now provides a finite AXI-shaped greedy transcription proof
for 16 kHz WAV inputs over the package-owned Whisper prompt/decode helpers.
Cached decoder state, timestamp segmentation, language detection, resampling,
and long-form audio chunking remain follow-up work.
FLUX.2 Klein snapshot/config recognition now exists as a separate
`@mlxts/diffusion` family contract over current Diffusers
`Flux2KleinPipeline`, `Flux2Transformer2DModel`, and `AutoencoderKLFlux2`
metadata. The first prepared-embedding sampling foundation has also landed:
NCHW 2x2 latent patching, 4-axis image/text ids, empirical FlowMatch dynamic
shift, external classifier-free guidance, distilled-guidance suppression, and
the VAE batch-norm inverse decode boundary. FLUX.2 transformer execution,
transformer/VAE weight loading, Qwen3 prompt conditioning, and a finite
AXI-shaped `examples/flux2` proof command have landed. Bounded real checkpoint
evidence has also passed through the official `black-forest-labs/FLUX.2-klein-4B`
checkpoint. Stable Diffusion 3 / 3.5 snapshot/config recognition now exists for
Diffusers `StableDiffusion3Pipeline` layouts, including
`SD3Transformer2DModel`, FlowMatch Euler, AutoencoderKL, three text
encoder/tokenizer components, and SD3.5 dual-attention metadata. The prepared
runtime foundation now covers NHWC latent patch embedding, fixed SD3 2D sincos
position crops, MMDiT joint attention, SD3.5 RMS q/k norm plus dual-attention
blocks, FlowMatch denoising over prepared conditioning tensors, and the VAE
shift/scale decode boundary. Transformer and VAE weight mapping/loading now
exist for inspected Diffusers snapshots, including base SD3 and SD3.5-style
generated safetensor proofs. The example-owned SD3 prompt-conditioning bridge
now composes the two CLIP projection encoders with the T5 encoder while keeping
encoder ownership outside `@mlxts/diffusion`, and the finite AXI proof command
now runs that bridge through FlowMatch denoising plus BMP artifact evidence.
Authenticated gated checkpoint proof remains a separate tranche. Reference-image
/ KV variants, broader VLM families, audio encoder/decoder families, and
additional diffusion/flow families remain Phase 10 work. LTX-Video and LTX-2
now have a package-owned Diffusers manifest entry point: current
`LTXPipeline`, `LTXConditionPipeline`, `LTXLatentUpsamplePipeline`, and
`LTX2Pipeline` snapshots parse into video/audio component roles and typed
component configs without importing transformer encoders or claiming runtime generation. Package-owned LTX latent
geometry now covers Diffusers-compatible video BCFHW packing and LTX-2 audio
BCLM packing, and package-owned RoPE geometry now covers classic LTX video
coordinate scaling plus LTX-2 video/audio patch-boundary coordinates. Classic
LTX-Video packed-latent denoising now covers prepared prompt embeddings and
attention masks, raw FlowMatch timesteps, unpatched video-length dynamic shift,
and negative-first batched CFG. Classic LTX-Video transformer execution now
covers packed video tokens, cached classic RoPE, AdaLayerNormSingle timestep
modulation, PixArt caption projection, RMS-normalized self/cross attention, and
Diffusers transformer weight mapping/loading. Classic LTX-Video VAE decode now
covers decoder-only `AutoencoderKLLTXVideo` execution, Diffusers latent
denormalization, decoder safetensor loading, and BFHWC `0..1` video tensors.
Classic LTX sidecar latent upsampling now covers `LTXLatentUpsamplerModel`
config parsing, safetensor loading, normalized latent upsampling, and packed
latent repacking. `examples/ltx-video` now provides the finite classic LTX
text-to-video proof command with T5 conditioning, denoising, VAE decode, and a
BMP preview-sheet artifact. LTX-2 denoising and LTX-2 latent upsampling remain
future tranches.

**What this phase covers**:

### 10a. Multimodal understanding (`@mlxts/transformers` expansion)

Vision encoders, audio encoders, and VLM wrappers are transformer architectures. They extend `@mlxts/transformers`, not a separate package.

- Vision encoder families: CLIP, SigLIP, ViT
- VLM wrapper families: initial Qwen 3.5 / Qwen 3.6 multimodal wrapper, then LLaVA, PaliGemma, Gemma 3/4, newer Mistral conditional-generation models
- Encoder-decoder families: Whisper (speech → text), T5, BART
- Audio preprocessing utilities co-located with Whisper family
- `ForwardOptions` gains optional `inputEmbeddings` and `positionIds` fields (landed for the first multimodal tranche)
- Explicit prepared-prompt generation (`generatePreparedTokens()`) and Qwen image-preparation helpers land before the broader family rollout

The `CausalLM` contract does not change. VLMs compose a vision encoder with a text decoder — the vision encoder preprocesses images into the text model's embedding space, then the text decoder generates autoregressively as normal.

### 10b. Diffusion/flow generation (`@mlxts/diffusion`)

All diffusion and flow-based generation across modalities: image, video, and audio. The package mirrors `@mlxts/transformers` in structure: explicit family registry, config-driven model construction, and official Hugging Face JS-backed snapshot loading.

- Backbone architectures: UNet2D, DiT (Diffusion Transformers), 3D variants for video
- VAE: image VAE, video VAE (3D causal), audio VAE
- Schedulers: DDPM, DDIM, DPM-Solver, Euler, Flow Matching
- Conditioning: cross-attention from text/image embeddings (produced by encoders from `@mlxts/transformers`)
- Sampling: classifier-free guidance, negative prompts
- Target families (informed by mlxr proving workloads): Stable Diffusion/SDXL, FLUX.1, Z-Image-Turbo, Qwen-Image/Qwen-Image-2512, FLUX.2 Klein, Stable Diffusion 3 / 3.5, LTX-Video, LTX-2
- Fine-tuning: `@mlxts/lora` and `@mlxts/train` work on diffusion models — LoRA targets attention layers in UNet/DiT the same way it targets attention in text decoders. DreamBooth and textual inversion are diffusion-specific techniques that live in this package.

**Image-generation support ladder:**

1. **Stable Diffusion / SDXL baseline**: this remains first because it proves
   the reusable package surface end to end: VAE, UNet2D, scheduler, CLIP
   conditioning, Diffusers local-or-Hub snapshot loading, sampling, and an
   AXI-shaped proof command. Diffusers Euler checkpoint metadata now maps to
   sigma-space scheduler behavior, and official SDXL base fp16 has passed a
   bounded real checkpoint image proof.
2. **FLUX.1 family**: this is the first modern flow-matching target after the
   Stable Diffusion baseline because it moves `@mlxts/diffusion` from UNet2D
   pipelines into DiT/flow-style backbones. The local `FLUX.1-schnell` proof
   path is implemented and now accepts Hub model ids through the package-owned
   Diffusers snapshot resolver, with its timestep-distilled constraints kept
   explicit: short prompt sequence length, guidance disabled, and few-step
   sampling. Official `black-forest-labs/FLUX.1-schnell` has passed a bounded
   real checkpoint image proof through that path. Gated or non-commercial
   variants require explicit operator and license handling before they are
   advertised.
3. **Z-Image-Turbo**: this is the first speed-first modern image target after
   FLUX.1 because it keeps the next runtime tranche focused: Diffusers exposes
   base `ZImagePipeline` snapshots as FlowMatch Euler plus a 6B
   `ZImageTransformer2DModel`, standard `AutoencoderKL`, and Qwen chat-template
   prompt encoding. The reference-audited snapshot/config skeleton has landed,
   and the base package runtime now owns patching, RoPE, single-stream
   denoising, VAE decode layout, and dense transformer weight mapping. The
   finite proof command is implemented in `examples/z-image`, and official
   dense checkpoint evidence has passed as a bounded 256px capability proof.
4. **Qwen-Image family**: this is the Qwen text-to-image generation track, not
   the already-landed Qwen 3.5 / Qwen 3.6 image-understanding route.
   `Qwen/Qwen-Image-2512` is the primary proved forward runtime target;
   `Qwen/Qwen-Image` remains the base compatibility fixture. Diffusers exposes
   it as `QwenImagePipeline` over FlowMatch Euler,
   `QwenImageTransformer2DModel`, `AutoencoderKLQwenImage`, and a Qwen2.5-VL
   text encoder. Its VAE is a 3D causal Qwen/Wan-derived autoencoder, so the
   landed implementation path is separate from the Stable Diffusion, FLUX, and
   Z-Image VAE paths. Runtime tensor execution, the AXI-shaped proof command,
   and official `Qwen/Qwen-Image-2512` bounded checkpoint evidence have landed.
5. **FLUX.2 Klein 4B**: this is a later separate family, not a FLUX.1 variant.
   It uses Diffusers `Flux2KleinPipeline`, `Flux2Transformer2DModel`,
   `AutoencoderKLFlux2`, and Qwen3 text encoding, so it should land only after
   the FLUX.1 and Z/Qwen flow-transformer seams are clean enough to avoid a
   parallel package shape. Snapshot recognition and component config parsing
   have landed, along with a prepared-embedding sampling foundation over
   package-owned FLUX.2 latent ids, FlowMatch denoising, external CFG, and VAE
   batch-norm decode semantics. Transformer execution, transformer/VAE weight
   loading, Qwen3 prompt conditioning, and a finite proof command in
   `examples/flux2` have landed. Official `black-forest-labs/FLUX.2-klein-4B`
   has passed a bounded real checkpoint image proof through that path.
   Image/reference conditioning, KV cache behavior, and larger
   quality/performance characterization remain separate tranches.
6. **Stable Diffusion 3 / 3.5 and distilled variants**: these become follow-on
   targets when their MMDiT/flow components can reuse the FLUX/Z-Image/Qwen
   infrastructure without creating a parallel package shape. SD3 has its own
   product cost because it combines `SD3Transformer2DModel`, FlowMatch Euler,
   AutoencoderKL, and three text encoders including T5-XXL. Snapshot recognition
   and component config parsing have landed for Diffusers
   `StableDiffusion3Pipeline` snapshots, including SD3.5 dual-attention fields.
   Runtime tensor execution has landed for prepared conditioning tensors,
   including SD3.5 q/k norm and dual-attention blocks. Transformer and VAE
   weight mapping/loading have landed for generated local safetensor snapshots.
   `examples/stable-diffusion-3` now owns the CLIP/T5 prompt-conditioning
   bridge with Diffusers hidden-state selection, pooled projection embeddings,
   T5 prompt embedding padding, and classifier-free guidance prompt rules.
   The finite proof command now resolves snapshots, loads the package-owned
   FlowMatch scheduler, transformer, and VAE, runs the prompt bridge, and writes
   BMP artifact evidence. The official `stabilityai/stable-diffusion-3.5-medium`
   proof is blocked on gated Hub access for the configured token; rerun the same
   proof command when access is granted or a local SD3/SD3.5 Diffusers snapshot
   is supplied.
7. **LTX-Video / LTX-2**: these open the video and audio-video diffusion track.
   Current Diffusers LTX-Video snapshots expose `LTXPipeline` or
   `LTXConditionPipeline` over FlowMatch Euler, T5 text metadata,
   `LTXVideoTransformer3DModel`, and `AutoencoderKLLTXVideo`; classic LTX
   sidecar upscalers expose `LTXLatentUpsamplePipeline` with
   `LTXLatentUpsamplerModel`. Current LTX-2 snapshots expose `LTX2Pipeline` over Gemma3 text metadata,
   `LTX2VideoTransformer3DModel`, `AutoencoderKLLTX2Video`,
   `AutoencoderKLLTX2Audio`, `LTX2TextConnectors`, and `LTX2Vocoder`. Snapshot
   recognition and component config parsing have landed as the entry point.
   Video latent shape/packing and LTX-2 audio latent shape/packing now match
   Diffusers token order. Classic LTX video RoPE coordinates and LTX-2
   video/audio patch-boundary RoPE coordinates now match the current Diffusers
   geometry. Classic LTX-Video prepared-tensor denoising now matches the
   current Diffusers loop shape for raw timesteps, video-length dynamic shift,
   semantic attention masks, VAE-derived RoPE interpolation scale, and batched
   CFG. Classic LTX-Video transformer execution now matches the packed-token
   Diffusers block shape for self/cross attention, AdaLayerNormSingle
   modulation, caption projection, and transformer weight mapping. Classic
   LTX-Video VAE decode now matches the Diffusers decoder-side boundary for
   packed-latent unpacking, channelwise denormalization, Conv3d kernel layout,
   decoder safetensor loading, and BFHWC `0..1` output tensors. Classic LTX
   sidecar latent upsampling now loads and runs over normalized BCFHW latents
   with packed-token unpack/repack helpers. The classic LTX finite proof command
   now writes a BMP preview-sheet artifact from decoded video. LTX-2 denoising
   and LTX-2 latent upsampling remain separate tranches.

### 10c. Examples

- `examples/whisper/` — transcribe audio on device
- `examples/text-to-image/` — generate images from text prompts
- `examples/qwen3_5-image/` — first dedicated Qwen image-conditioned generation example
- `examples/vlm-chat/` — broader chat-with-images surface after the initial wrapper tranche

### 10d. Completion fence

Phase 10 is complete only when multimodal understanding and diffusion/flow
generation are both represented by package-owned APIs, examples, and product
proofs:

- At least one VLM path describes and answers questions about local images
  through `@mlxts/transformers`, `@mlxts/serve`, and an example/workbook.
- At least one audio or encoder-decoder path proves local preprocessing,
  model execution, and text output without widening `CausalLM`.
- At least one diffusion/flow pipeline generates an image through
  `@mlxts/diffusion`, with scheduler, VAE/backbone, conditioning, and sampling
  owned by the package.
- Image-generation proof artifacts are machine-checkable from saved JSON and
  BMP files without rerunning model generation.
- Media transport, file-store, remote-fetch, preprocessing, cache, and serving
  ownership boundaries are documented before each capability is advertised.
- Finite Phase 10 CLI/proof commands are AXI-shaped from day one.
- Runtime-sensitive paths have review artifacts, focused real-checkpoint
  evidence, and the same validation posture as text serving.

**Exit criteria**: See [gates-and-milestones.md](./docs/gates-and-milestones.md#phase-10-completion-fence).

---

## Phase 11: Future — Multi-Backend (if warranted)

If demand warrants, the package structure supports adding secondary backends (WebGPU for browser, CUDA via libtorch shim). This is not part of the current plan. See [docs/future-backends.md](./docs/future-backends.md) for the vision and technical analysis.

---

## Phase 12: Evaluation and Benchmarks

**Goal**: Standardized model evaluation. Reproducible benchmarks. Credibility.

**What this phase covers**:

### 12a. Eval harness (`@mlxts/eval`)

- 6 core benchmark tasks: MMLU, HellaSwag, ARC, WinoGrande, TruthfulQA, GSM8K
- `LM` interface: `loglikelihood`, `generate`, `loglikelihood_rolling`
- JSON result output for comparison
- `mlxts eval --model Llama-3.2-1B --tasks mmlu,hellaswag`

### 12b. Cross-validation

- Results match Python lm-eval-harness within 1% for same model
- Published methodology

**Exit criteria**: See [gates-and-milestones.md](./docs/gates-and-milestones.md#phase-12-evaluation-and-benchmarks).

---

## North Star: "Implement Any Paper"

The ecosystem is complete when a TypeScript developer can:

1. **Implement a custom model architecture** using `@mlxts/nn` building blocks
2. **Train it** using `@mlxts/train`
3. **Fine-tune a pretrained model** using `@mlxts/lora`
4. **Serve it** using `@mlxts/serve`
5. **Evaluate it** using `@mlxts/eval`
6. **Add custom Metal kernels** via FFI and compose them with existing ops
7. **Do all of the above without leaving TypeScript**
8. **Read the code and understand how it works**

See [gates-and-milestones.md](./docs/gates-and-milestones.md#ultimate-milestone-implement-any-paper) for how we test this.

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

---

## Principles

1. **Correctness over speed** — Get it right, then make it fast
2. **Document as we go** — Every design decision is recorded
3. **Test everything** — No code ships without tests
4. **Agent review** — No agent's output merges without review by a different agent
5. **Incremental delivery** — Each phase produces something that works
6. **Education first** — Code clarity trumps cleverness
7. **MLX-native everywhere** — No abstraction layers between your code and the GPU
8. **Modular by default** — Every package earns its place by having a real consumer
9. **Interoperable** — Load HF models, serve OpenAI-compatible APIs, read community formats
10. **One person, many agents** — Decision quality and architectural coherence over headcount
11. **Known risks documented** — Bun FFI bugs and mlx-c pre-1.0 instability are tracked, not ignored
