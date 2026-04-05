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
| [docs/architecture.md](./docs/architecture.md) | System architecture and layer responsibilities |
| [docs/code-standards.md](./docs/code-standards.md) | Code quality, naming, structure, testing standards |
| [docs/agentic-loop.md](./docs/agentic-loop.md) | Multi-agent engineering workflow |

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

- `packages/nanogpt/` — complete GPT implementation
- Trains on Shakespeare, generates coherent text
- Canonical CLI with `train` / `generate`
- Memory benchmark and supervised soak surfaces before loss-targeted acceptance
- Acceptance scripts: `bun run acceptance:gpt-tiny` and `bun run acceptance:gpt-small`

**Exit criteria**:
- `bun run validate` passes
- `gpt-tiny` trains to <1.8 validation loss on Shakespeare in an explicit acceptance run
- `gpt-small` also has an explicit loss-targeted acceptance run
- Long unattended runs use the supervised `bun run run:nanogpt ...` surface rather than one-off scripts
- Long-run acceptance follows a soak ladder (`50 → 250 → 1000 → 5000`) rather than jumping straight to overnight runs
- Runtime-sensitive diffs leave a review artifact under `docs/reviews/` and pass `bun run check:runtime-review`
- Generated text is recognizably English and vaguely Shakespearean

**Post-Phase-4 posture**: Phase 4 completion triggers the ecosystem restructure. The proven code from `mlx-ts` and `nanogpt` becomes the foundation for the `@mlxts/*` package family. nanoGPT remains a temporary in-repo validation fixture during extraction; richer examples move to a dedicated examples repo later.

---

## Phase 5: Ecosystem Restructure

**Goal**: Adopt the `mlxts` package identity inside the monorepo, extract the canonical `@mlxts/*` packages, and make the repo truthful about the package-first transition. Full example rewrites are intentionally deferred until the ecosystem is more complete.

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
| `@mlxts/train` | Former generic pieces of `nanogpt/src/train.ts` and `checkpoint.ts` | Training loop, checkpoint, gradient utilities, typed checkpoint metadata, reusable step orchestration |
| `@mlxts/data` | Former `nanogpt/src/data.ts` | Text data loading, batching |
| `@mlxts/tokenizers` | Former `nanogpt/src/tokenizer.ts` | Character tokenizer (BPE comes in Phase 7) |

This extraction is now real, not hypothetical:

- `@mlxts/core`, `@mlxts/nn`, `@mlxts/optimizers`, `@mlxts/train`, `@mlxts/data`, and `@mlxts/tokenizers` exist as workspace packages
- `packages/core` owns the native MLX build and the canonical FFI/runtime surface
- `packages/nanogpt` now consumes the extracted packages directly and is being reduced to a thin GPT-specific validation fixture

### 5c. Package-first validation posture

- `packages/nanogpt` remains a temporary private validation fixture while the reusable packages stabilize
- `@mlxts/train`, `@mlxts/data`, and `@mlxts/tokenizers` are the sole implementations of generic training/data/tokenizer behavior; `packages/nanogpt` only wraps them for GPT-specific flows
- `bun run check:file-lines` enforces the 500-line cap for active production source, including the temporary fixture
- Coverage, runtime-review, type-assertion, and tensor-lifetime gates recognize the extracted package layout
- `docs/reviews/phase-5-restructure.md` records the runtime-sensitive package extraction work
- Phase 5 success is defined by clean package surfaces and truthful repo docs, not by forcing a premature in-repo example rewrite

### 5d. Validation

- `bun run validate` passes across entire monorepo
- Each extracted package typechecks and tests independently
- `bun run check:file-lines` passes
- Runtime review stays in place for the extracted package surfaces
- The temporary shim and fixture are explicitly documented as transitional surfaces

### 5e. What comes next

- Keep hardening the extracted packages and their docs until the temporary shim and fixture can be deleted cleanly
- Continue using `packages/nanogpt` as a validation harness in the meantime
- Design the dedicated examples repo later, including a ground-up nanoGPT rewrite once the ecosystem surface is broader and more stable

**Exit criteria**:
- See [gates-and-milestones.md](./docs/gates-and-milestones.md#phase-5-ecosystem-restructure)
- Canonical `@mlxts/*` packages exist and pass their package-local tests
- `bun run validate` and `bun run check:file-lines` pass
- Runtime review artifacts and top-level docs accurately describe the package-first state
- `packages/mlx-ts` and `packages/nanogpt` are clearly marked as temporary transitional surfaces rather than long-term end states

---

## Phase 6: Publish Core Packages

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

### 7a. Hub integration (`@mlxts/hub`)

- HuggingFace Hub REST client (download models, datasets)
- safetensors reader/writer (native via mlx-c or pure TS)
- GGUF header/metadata parsing only. Tensor dequantization (15+ quant formats) deferred to Phase 9.
- `config.json` and `tokenizer.json` parsing
- Local model cache with integrity checking
- HF → mlxts weight name mapping

### 7b. Tokenizers (`@mlxts/tokenizers` expansion)

- ByteLevel BPE from `tokenizer.json` (covers LLaMA/Mistral/GPT-2). Full 5-stage HF tokenizer pipeline is future work.
- Batch encode/decode with offset tracking

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

- `AutoModel.fromPretrained(modelId)` auto-dispatch
- `AutoTokenizer.fromPretrained(modelId)`

### 7d. Examples

- `examples/llama-chat/` — interactive chat with a local LLaMA model
- The later dedicated examples repo can adopt `@mlxts/transformers` where it improves the rewritten examples

**Exit criteria**: See [gates-and-milestones.md](./docs/gates-and-milestones.md#phase-7-model-architectures).

---

## Phase 8: Fine-Tuning

**Goal**: LoRA fine-tuning, DPO alignment, dataset loading. A TS developer can fine-tune a model on their own data.

**What this phase covers**:

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

### 8c. Data expansion (`@mlxts/data`)

- HuggingFace Datasets format loading
- Conversation/chat formatting
- Instruction tuning data collation

### 8d. Examples

- `examples/lora-finetune/` — fine-tune LLaMA on custom data, merge, generate

**Exit criteria**: See [gates-and-milestones.md](./docs/gates-and-milestones.md#phase-8-fine-tuning).

---

## Phase 9: Inference and Serving

**Goal**: Production-quality inference server. Quantized inference. Any OpenAI-compatible client can connect.

**What this phase covers**:

### 9a. Quantization (`@mlxts/quantize`)

- 4-bit and 8-bit quantization via MLX native `mx.quantize`/`mx.dequantize`
- GGUF tensor dequantization (15+ quant formats, moved from Phase 7)
- GGUF export (create GGUF files from mlxts models)
- Calibration dataset support for quantization quality
- Quantized inference at full speed

### 9b. KV cache and efficient generation

- KV cache for autoregressive generation (prerequisite for serving)
- PagedAttention-style cache management is a major engineering effort. Start with simple KV cache, optimize later.

### 9c. Serving (`@mlxts/serve`)

- OpenAI-compatible API: `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`
- `Bun.serve()` — no Express, no Node HTTP
- Server-sent events for token streaming
- KV cache management for concurrent requests
- Continuous batching (serve multiple requests)
- Model loading/unloading without restart

### 9d. CLI expansion (`@mlxts/cli`)

- `mlxts serve --model Llama-3.2-1B --quantize 4bit`
- `mlxts convert --source hf --model meta-llama/Llama-3.2-1B`
- `mlxts quantize --model ./my-model --bits 4`
- `mlxts download --model meta-llama/Llama-3.2-1B`

**Exit criteria**: See [gates-and-milestones.md](./docs/gates-and-milestones.md#phase-9-inference-and-serving).

---

## Phase 10: Diffusion and Multi-Modal

**Goal**: Image generation, speech recognition, vision-language models.

**What this phase covers**:

### 10a. Whisper (`@mlxts/audio`) — speech recognition

- Whisper speech recognition (smallest scope — one model family)
- Audio I/O (mel spectrograms, resampling)
- `examples/whisper/` — transcribe audio

### 10b. Diffusion (`@mlxts/diffusion`)

- Requires UNet, VAE, noise schedulers
- Noise schedulers: DDPM, DDIM, DPM-Solver, Euler
- UNet2D architecture
- VAE (Variational Autoencoder)
- `StableDiffusionPipeline` — text to image
- ControlNet support
- `examples/stable-diffusion/` — generate images from text

### 10c. Vision-Language (VLM)

- Requires vision encoder + LLM integration
- VLM support via `@mlxts/transformers` (LLaVA, PaliGemma)
- Image preprocessing pipeline

**Exit criteria**: See [gates-and-milestones.md](./docs/gates-and-milestones.md#phase-10-diffusion-and-multi-modal).

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
