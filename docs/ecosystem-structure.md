# mlxts Ecosystem Structure

## Identity

| Field | Value |
|-------|-------|
| **Name** | mlxts |
| **npm scope** | `@mlxts/*` |
| **Repo** | `mlxts` (monorepo) |
| **Runtime** | Bun (only) |
| **Primary target** | Apple Silicon via MLX |
| **Prior art** | @frost-beta/mlx (Node.js MLX bindings), Transformers.js v4 (WebGPU inference) |

## Design Principles

1. **One scope, many packages.** Every package is `@mlxts/<name>`. Users install only what they need.
2. **Layers, not monoliths.** Each package has a single concern. Dependencies flow downward.
3. **MLX-native throughout.** All packages use MxArray directly. No abstraction layers between your code and Metal.
4. **Examples are separate surfaces.** Committed end-to-end examples live under `examples/` in this monorepo today. Keep them thin, package-powered, and non-publishable; split them into a dedicated examples repo only if the portfolio grows enough to justify it.
5. **Extensible via FFI.** Custom Metal kernels, C/Rust extensions, and new ops can be added and composed with existing infrastructure.

## Current Implementation State

The package-first Phase 5 extraction is already underway in the repo today.

- `@mlxts/core`, `@mlxts/nn`, `@mlxts/optimizers`, `@mlxts/train`,
  `@mlxts/data`, and `@mlxts/tokenizers` all exist as workspace packages
- `packages/core` owns the native MLX build and the canonical FFI/runtime layer
- `examples/nanogpt` now consumes the extracted packages directly and serves as
  the committed in-repo GPT example and regression surface while the reusable
  package ecosystem settles

---

## Package Map

### Layer 0: Core

#### `@mlxts/core`

The foundation of the mlxts ecosystem. Contains the MxArray tensor type, all ops, transforms, and the MLX FFI bindings. There is no separate `@mlxts/mlx` backend package — the MLX bindings live directly in `@mlxts/core` because MLX is the only backend.

| Concern | What it provides |
|---------|-----------------|
| MxArray | The tensor type — wraps mlx-c array handles with disposal, shape, dtype |
| DType system | `DType` union, `INTEGER_DTYPES`, `isIntegerDType`, `DTYPE_TO_BYTES` |
| FFI bindings | `dlopen` of `libmlxc.dylib`, symbol declarations, pointer management |
| Ops | All tensor operations: arithmetic, comparison, linalg, reduction, shape |
| Transforms | `valueAndGrad`, `grad`, `compile`, `checkpoint` via mlx-c |
| Fast fused ops | `scaledDotProductAttention`, `layerNorm`, `rmsNorm`, `rope` |
| Random | `normal`, `uniform`, `key`, `split` |
| Device/Stream | Device and stream management |
| Memory | MLX memory management, Metal device control |
| I/O | `loadSafetensors`, `saveSafetensors` via mlx-c |
| Tree utilities | `treeFlatten`, `treeUnflatten`, `treeMap`, `treeLeaves`, `ParameterTree` |
| Shape utilities | `formatShape`, `broadcastShapes` |
| Error types | `MxError`, `mlxtsError` base class |

**Dependencies:** None. This is a leaf package.

**Source origin:** Extracted from the former single-package MLX monolith and now lives in `packages/core/`.

---

### Layer 1: Neural Network Framework

#### `@mlxts/nn`

Neural network layers, Module system, activations, losses.

| Concern | What it provides |
|---------|-----------------|
| Module base | `Module` class with property scanning, freeze/unfreeze, train/eval |
| Linear layers | `Linear` |
| Convolution layers | `Conv1d`, `Conv2d` |
| Normalization | `LayerNorm`, `RMSNorm`, `GroupNorm` |
| Embeddings | `Embedding` with `asLinear()` for weight tying |
| Regularization | `Dropout` |
| Activations | `gelu`, `relu`, `silu`, `swiglu` (free functions) |
| Losses | `crossEntropy`, `mse` |
| Attention | `GroupedQueryAttention` |
| Module autograd | `nn.valueAndGrad` — the module-aware gradient transform |

**Dependencies:** `@mlxts/core` — nn modules import MxArray and ops directly.

**Source origin:** Extracted from the former monolithic nn layer. GPT-specific
causal attention remains in `examples/nanogpt` for now, while the reusable
transformer primitives now live here.

#### `@mlxts/optimizers`

Gradient-based optimizers and learning rate schedules.

| Concern | What it provides |
|---------|-----------------|
| Optimizers | `SGD`, `Adam`, `AdamW`, `Adafactor` (future), `Lion` (future) |
| LR schedules | `cosineAnnealing`, `warmupCosine`, `linearWarmup`, `constant` |
| Optimizer base | `Optimizer` abstract class with state management and disposal |

**Dependencies:** `@mlxts/core`, `@mlxts/nn` — `Optimizer.update()` accepts `Module` for parameter extraction.

**Source origin:** Extracted from the former monolithic optimizer layer. Generic learning-rate schedule helpers now live in `@mlxts/train`.

---

### Layer 2: Training Infrastructure

#### `@mlxts/train`

Model-agnostic training loop, checkpointing, gradient utilities.

| Concern | What it provides |
|---------|-----------------|
| Training loop | `trainLoop()` with explicit config validation and callbacks |
| Checkpointing | `saveCheckpoint`, `loadCheckpoint`, `applyCheckpoint`, optimizer restore helpers |
| Gradient utils | `accumulateGradients`, `clipGradientTree`, `scaleGradientTree`, norm/eval/free helpers |
| Schedules | `warmupCosineSchedule`, `getLearningRate`, schedule validation |
| Config | `TrainLoopConfig`, `TrainLoopOptions`, learning-rate config types |
| Composition follow-ons (planned) | Explicit hooks, checkpoint/eval policies, artifact sinks, and train-event streams — not a reactive framework |

**Dependencies:** `@mlxts/core`, `@mlxts/nn`, `@mlxts/optimizers`

**Source origin:** Extracted from the model-agnostic parts of the former `examples/nanogpt/src/train.ts` and `examples/nanogpt/src/checkpoint.ts`. The package now also owns the canonical metadata-driven checkpoint format and reusable step-orchestration helpers used by the committed nanoGPT example.

`@mlxts/train` is intentionally not a black-box pipeline framework. Recipe-level
orchestration lives above it, in `@mlxts/align`, examples, and later CLI/app
surfaces. Future composition work should stay explicit and package-owned rather
than introducing RxJS/Effect-style reactive dependencies into the core training
layer.

#### `@mlxts/data`

Dataset loading, batching, and preprocessing.

| Concern | What it provides |
|---------|-----------------|
| Text data | `loadText`, `prepareData`, `getBatch` |
| Dataset base | `Dataset` interface, `IterableDataset` for streaming |
| Data collation | Padding, batching, attention mask generation |
| Preprocessing | Text normalization, train/val splitting |

**Dependencies:** `@mlxts/core`

**Source origin:** Extracted from the former `examples/nanogpt/src/data.ts` and now lives in `packages/data/src/text.ts`. `examples/nanogpt` imports these helpers directly rather than carrying a second copy.

---

### Layer 3: Tokenization and Pretrained Loading

#### `@mlxts/tokenizers`

Fast tokenization with support for HuggingFace tokenizer formats.

| Concern | What it provides |
|---------|-----------------|
| BPE | Byte-pair encoding from `tokenizer.json` |
| Char tokenizer | Simple character-level tokenizer |
| SentencePiece | SentencePiece `.model` loading in pure TypeScript |
| Tekken | `tekken.json` loading for modern Mistral tokenizers |
| Encoding | Batch encode/decode, offset tracking, special tokens |

**Dependencies:** none

**Source origin:** Extracted from the former `examples/nanogpt/src/tokenizer.ts`. The package now owns the char tokenizer plus pretrained `tokenizer.json`, SentencePiece, and Tekken loading for the Phase 7 decoder families.

#### External: `@huggingface/hub` and `@huggingface/jinja`

Official Hugging Face JavaScript packages used by the pretrained loading surface.

| Concern | What it provides |
|---------|-----------------|
| Hub client | Download and cache model snapshots from Hugging Face |
| Snapshot layout | Canonical Hub cache structure shared with the wider ecosystem |
| Chat templates | Jinja rendering for `chat_template` / `chat_template.jinja` |

`mlxts` now uses these packages directly from `@mlxts/transformers` instead of carrying a separate repo-owned hub package.

---

### Layer 4: Model Architectures

#### `@mlxts/transformers`

All transformer-based model architectures — the mlxts equivalent of HuggingFace Transformers. Covers text decoders, MoE variants, vision encoders, VLM wrappers, and encoder-decoder models. Packages are organized by **generation paradigm**, not input/output modality — see [design-reasoning.md § Generation Paradigms](../docs/design-reasoning.md#generation-paradigms).

| Concern | What it provides |
|---------|-----------------|
| Text decoder families | LLaMA, Mistral, Mistral 3, Gemma, Gemma 3, Gemma 4 text, Phi 3/4-mini |
| MoE families (Phase 7f) | Mixtral, DeepSeek — block-level MoE swap, same CausalLM contract |
| Vision encoder families (Phase 10) | CLIP, SigLIP, ViT — for VLM composition and diffusion conditioning |
| VLM wrapper families (Phase 10) | Initial Qwen 3.5 / Qwen 3.6 multimodal wrapper plus future LLaVA, PaliGemma, Gemma 3/4 — compose vision encoder + text decoder |
| Encoder-decoder families (Phase 10) | Whisper (speech → text), T5, BART |
| Auto dispatch | `AutoModel.fromPretrained(modelId)` — config-driven architecture selection |
| Generation | `generateText()` / `generateTokens()` / `generateStep()` with KV cache and sampling |
| Weight loading | Load from safetensors via `@mlxts/core` with internal pretrained snapshot inspection |

**Dependencies:** `@mlxts/core`, `@mlxts/lora`, `@mlxts/nn`, `@mlxts/quantize`, `@mlxts/tokenizers`, `@huggingface/hub`, `@huggingface/jinja`

**Architecture pattern:** An explicit registry maps `model_type` to family
parsers and model constructors. Shared LLaMA-like structure lives under
`families/llama-like/`, while per-family config and weight-name mapping stays
under `families/<family>/`. The `CausalLM` contract is the right boundary for
all autoregressive models — MoE is a block-level swap, multimodal understanding
is a composition layer. See [design-reasoning.md § Contract Boundaries](../docs/design-reasoning.md#contract-boundaries).
CLIP text encoders live under `families/clip/` as explicit encoder surfaces for
conditioning and multimodal composition; they are not registered as CausalLMs.

#### `@mlxts/diffusion` (Phase 10)

All diffusion and flow-based generation across modalities: image, video, and audio. This is the generative media package — the counterpart to `@mlxts/transformers` for the diffusion/flow generation paradigm.

| Concern | What it provides |
|---------|-----------------|
| Backbone architectures | UNet2D, DiT (Diffusion Transformers), 3D variants for video |
| VAE | Image VAE, video VAE (3D causal), audio VAE |
| Schedulers | DDPM, DDIM, DPM-Solver, Euler, Flow Matching |
| Conditioning | Cross-attention from text/image embeddings (from `@mlxts/transformers` encoders) |
| Sampling | Classifier-free guidance, negative prompts |
| Fine-tuning support | DreamBooth, textual inversion (LoRA via `@mlxts/lora` works on any Linear) |

**Dependencies:** `@mlxts/core`, `@mlxts/nn`

**Current state:** Scheduler infrastructure, local scheduler-config loading,
Diffusers `model_index.json` snapshot inspection, and Stable Diffusion VAE/UNet
config translation exist. Stable Diffusion VAE and conditional UNet module
construction exists, with VAE and UNet safetensor loading in place. Stable
Diffusion pipeline assembly over supplied conditioning tensors owns NHWC latent
shape, DDIM/Euler denoising, classifier-free guidance, VAE unscale, and
postprocessing. Local snapshots now load into a disposable Stable Diffusion
runtime bundle with VAE, UNet, scheduler, parsed manifest/configs, and thin
sampling methods. FLUX.1 owns FlowMatch scheduling, transformer
config/backbone/weights, VAE loading/decoding, latent packing, and sampling.
Base Qwen-Image and Z-Image snapshots are recognized and parsed at the
package-owned metadata boundary while their runtime tensor execution remains
follow-on work. `examples/stable-diffusion` and `examples/flux` own
application-layer prompt-conditioning composition and finite BMP image proof
commands over package surfaces. Hugging Face Hub-backed Diffusers root loading,
real checkpoint evidence, and broader image output formats remain follow-on
Phase 10 tranches.

**Image-generation support order:** Stable Diffusion / SDXL is the baseline
pipeline family. FLUX.1 is the next flow-matching target. Qwen-Image is the
Qwen text-to-image generation family and stays separate from Qwen 3.5 / Qwen
3.6 image understanding in `@mlxts/transformers`. Z-Image-Turbo is the
speed-first local target after Qwen-Image and now has a reference-backed
snapshot/config seam. Stable Diffusion 3 / 3.5 and distilled or turbo variants
follow only when they reuse the base family infrastructure or come with a
documented architecture delta.

**Architecture pattern:** Mirrors `@mlxts/transformers` — explicit family
registry, config-driven model construction, and official Hugging Face JS-backed snapshot loading.
Conditioning embeddings are produced by text/image encoders from
`@mlxts/transformers` and passed as tensors — no direct dependency between the
two architecture packages. Pipeline orchestration (composing both) is
application-layer code.

---

### Layer 5: Fine-Tuning

#### `@mlxts/lora`

Parameter-efficient fine-tuning.

| Concern | What it provides |
|---------|-----------------|
| LoRA | Low-rank adaptation for any `Linear` layer |
| QLoRA | Quantized LoRA (requires `@mlxts/quantize`) |
| Adapter injection | `applyLoRA(model, config)` — wraps target layers |
| Merge | `mergeLoRA(model)` — merge adapters back into weights for zero-overhead inference |
| Config | `LoRAConfig` — target layers, rank, alpha, dropout |

**Dependencies:** `@mlxts/core`, `@mlxts/nn`

#### `@mlxts/align`

Alignment and RLHF.

| Concern | What it provides |
|---------|-----------------|
| SFT | Supervised fine-tuning trainer |
| DPO | Direct Preference Optimization |
| Reward modeling | Reward model training |
| Data | Preference pair formatting, chat template support, raw-chat normalization, dataset-level SFT/DPO evaluation helpers |
| Recipe loops | Small fixed-step SFT/DPO runners that keep examples and future CLIs thin without hiding control flow |
| Proof surface | Canonical real-data proof via `examples/train-proof/` on pinned dataset subsets |

**Dependencies:** `@mlxts/core`, `@mlxts/data`, `@mlxts/lora`, `@mlxts/nn`, `@mlxts/tokenizers`, `@mlxts/train`, `@mlxts/transformers`

The short real-data training proof is the regression path for this layer. Today
it is a canonical runnable surface; long term, architecture or quantization
changes that break LoRA, QLoRA, SFT, or DPO on the canonical proof should fail
CI rather than relying on manual spot checks.

---

### Layer 6: Inference and Serving

#### `@mlxts/protocols`

Small shared protocol helpers used by serving and agent clients. This package is
intentionally zero-dependency: it owns wire-format text normalization that must
not drift between packages, such as model-native reasoning tag separation.

| Concern | What it provides |
|---------|-----------------|
| Reasoning tags | Shared splitting/streaming cleanup for Qwen `<think>`, Anthropic-style `<antThinking>`, and Gemma thought-channel markers |
| Wire helpers | Tiny protocol utilities that do not depend on model execution, HTTP servers, or MLX tensors |

**Dependencies:** none

#### `@mlxts/serve`

Production inference server with OpenAI-compatible and Anthropic-compatible API
slices.

| Concern | What it provides |
|---------|-----------------|
| API server | `/v1/chat/completions`, `/v1/completions`, text-only `/v1/responses`, text-only `/v1/messages`, `/v1/models`, `/health`, `/info` |
| Admission | prompt/generated/total-token limits, memory preflight, cancellation, and streaming lifecycle |
| Batching | admission micro-batching plus cache-generic continuous scheduling for eligible LLaMA-like, Qwen 3.6 text, and Gemma 3/4 layer-pattern requests |
| Streaming | Server-sent events for completions, chat, narrow Responses, bounded Anthropic Messages, reasoning separation, and stream keepalive |
| Model serving | single-model and multi-model load-at-start serving; dynamic load/unload and engine-pool eviction remain future work |
| Benchmarking | endpoint `bench:serve` and Qwen/Gemma regression matrices with route, scheduler, stream, and memory evidence |
| Protocol adapters | OpenAI chat/completions, narrow text Responses, and bounded Anthropic Messages over one shared request model; embeddings and broader content/tool semantics remain future adapters |

**Dependencies:** `@mlxts/core`, `@mlxts/protocols`, `@mlxts/transformers`, `@mlxts/tokenizers`

**Runtime:** `Bun.serve()` — no Express, no Node HTTP.

#### `@mlxts/agent`

Local tool-using agent loops over served models. This is a debugging and product
surface for exercising model/tool behavior without putting orchestration inside
the serving package.

| Concern | What it provides |
|---------|-----------------|
| Tool loop | model → tool call → observation → continue orchestration |
| Local tools | Read-only filesystem tools for safe local debugging |
| Chat client | OpenAI chat-completions client with streaming, reasoning, and tool-call support |

**Dependencies:** `@mlxts/protocols`

#### `@mlxts/quantize`

Model quantization and compressed inference.

| Concern | What it provides |
|---------|-----------------|
| Quantize | 4-bit, 8-bit quantization (via MLX native `mx.quantize`) |
| Dequantize | Runtime dequantization for inference |
| GGUF quantization | Create GGUF files from mlxts models |
| Calibration | Calibration dataset for quantization quality |

**Dependencies:** `@mlxts/core`, `@mlxts/nn`

---

### Layer 7: Evaluation

#### `@mlxts/eval`

**Future package.** Evaluation is a planned layer, not a current workspace
package. Current benchmark and regression surfaces live in `@mlxts/transformers`,
`@mlxts/serve`, and example-owned proof flows.

Model evaluation and benchmarking.

| Concern | What it provides |
|---------|-----------------|
| Benchmark tasks | MMLU, HellaSwag, ARC, WinoGrande, TruthfulQA, GSM8K |
| LM interface | `loglikelihood`, `generate`, `loglikelihood_rolling` |
| Metrics | Accuracy, perplexity, F1, exact match |
| Reporting | JSON results, comparison tables |

**Dependencies:** `@mlxts/core`, `@mlxts/transformers`, `@mlxts/tokenizers`, `@mlxts/data`

---

### Layer 8: Developer Tools

#### `@mlxts/cli`

**Future umbrella package.** The current CLI shape is package-owned binaries
such as `mlxts-serve` and `mlxts-agent`. Keep that shape until an umbrella CLI
has a stronger reason to exist than centralizing names. Agent-facing finite
commands follow the AXI contract before they are wrapped by any umbrella
command.

Command-line tools for the mlxts ecosystem.

| Concern | What it provides |
|---------|-----------------|
| Package-owned commands | Current binaries and example-local managers stay beside their backing packages |
| AXI finite output | Compact structured stdout, structured errors, stable exit codes, and no non-TTY prompts |
| `mlxts train` | Future wrapper over package-owned training/proof commands |
| `mlxts generate` | Future wrapper over text and media generation workbooks |
| `mlxts serve` | Future wrapper over `mlxts-serve` once serving CLI contracts are coherent |
| `mlxts convert` | Future model conversion command |
| `mlxts quantize` | Future wrapper over `@mlxts/quantize` |
| `mlxts eval` | Future wrapper over `@mlxts/eval` |
| `mlxts download` | Future Hugging Face snapshot helper |

**Dependencies:** All packages as needed per subcommand.

---

## Examples Strategy

`examples/nanogpt` is intentionally committed in-repo as the canonical GPT
example and operator surface while the package ecosystem continues to harden.
`examples/qwen3_5-image` is the first dedicated multimodal example and stays
thin by pushing model, prompt, and patchification logic down into
`@mlxts/transformers`.
Additional examples can live under `examples/` too; a separate examples repo is
only worth it if the portfolio becomes large enough to need one.

```
examples/
  nanogpt/                    # Committed in-repo example and regression surface
    src/
      run/                    # Supervised run manager, soak, and acceptance infra
      bench/                  # Memory and throughput checks
  qwen3_5-image/              # Qwen 3.5 / Qwen 3.6 image-conditioned example
```

These examples are intentionally thin: they validate the reusable packages,
host operator or smoke workflows, and avoid becoming second package-owned
product surfaces.

---

## Repository Layout

```
mlxts/                                # Monorepo root
  package.json                        # Bun workspaces, root scripts
  PLAN.md                             # Ecosystem roadmap (references docs/)
  AGENTS.md                           # Agent instructions
  CLAUDE.md                           # Project entry point for agents
  biome.json                          # Linting/formatting config
  tsconfig.json                       # Root TypeScript config

  packages/
    core/                             # @mlxts/core — ALL MLX bindings + types + ops
      package.json
      native/                         # Vendored mlx-c source + CMake
      scripts/
        build-native.ts               # Build libmlxc.dylib
      src/
        index.ts                      # Public API barrel
        array.ts                      # MxArray tensor type
        dtype.ts                      # DType union, utilities
        tree.ts                       # Parameter tree utilities
        error.ts                      # Base error types
        shape.ts                      # Shape utilities
        ffi/                          # FFI bindings
          lib.ts                      # dlopen
          symbols.ts                  # Symbol declarations
          index.ts                    # Helpers
          pointer.ts                  # Pointer narrowing utilities
          closure-bridge.ts           # Autograd callback bridge
        ops/                          # Op implementations
          arithmetic.ts
          comparison.ts
          linalg.ts
          reduction.ts
          shape.ts
        transforms.ts                 # Autograd, compile, checkpoint
        fast.ts                       # Fused ops (SDPA, layerNorm, etc.)
        device.ts                     # Device/stream management
        memory.ts                     # Memory management
        metal.ts                      # Metal-specific APIs
        random.ts                     # RNG
        io.ts                         # safetensors native I/O

    nn/                               # @mlxts/nn
      package.json
      src/
        index.ts
        module.ts                     # Module base class
        linear.ts                     # Linear layer
        embedding.ts                  # Embedding + asLinear
        layer-norm.ts                 # LayerNorm
        dropout.ts                    # Dropout
        activations.ts                # gelu, relu, silu
        losses.ts                     # crossEntropy, mse
        attention.ts                  # MultiHeadAttention
        value-and-grad.ts             # nn.valueAndGrad bridge

    optimizers/                       # @mlxts/optimizers
      package.json
      src/
        index.ts
        optimizer.ts                  # Base class
        adam.ts                       # Adam, AdamW
        sgd.ts                        # SGD
        schedules.ts                  # LR schedules

    train/                            # @mlxts/train
      package.json
      src/
        index.ts
        loop.ts                       # Training loop orchestration
        schedule.ts                   # Learning-rate schedules
        gradients.ts                  # Gradient accumulation, clipping, scaling
        checkpoint.ts                 # Public checkpoint surface
        checkpoint-*.ts               # Manifest / serialization / I/O helpers

    tokenizers/                       # @mlxts/tokenizers
      package.json
      src/
        index.ts
        bpe.ts                        # BPE from tokenizer.json
        char.ts                       # Character-level tokenizer
        sentencepiece.ts              # SentencePiece loading
        types.ts                      # Tokenizer interface

    data/                             # @mlxts/data
      package.json
      src/
        index.ts
        text.ts                       # Text data loading/batching
        # Dataset abstractions can arrive later if a second consumer needs them

  examples/
    nanogpt/                          # Committed in-repo example and regression surface
      package.json
      src/
        run/
        bench/

  scripts/                            # Validation gates (repo-level)
    check-coverage.ts
    check-type-assertions.ts
    check-file-lines.ts
    check-runtime-review.ts
    check-visible-tensor-lifetimes.ts

  docs/                               # Architecture and planning docs
    ecosystem-structure.md            # This document
    future-backends.md                # Multi-backend design
    python-equivalence-map.md         # Python ecosystem mapping
    gates-and-milestones.md           # Exit criteria for every phase
    architecture.md                   # System architecture
    code-standards.md                 # Coding conventions
    agentic-loop.md                   # Agent workflow
    mlx-bindings.md                   # MLX binding approach
    runtime-safety.md                 # Runtime safety practices
    product-surfaces.md               # API/CLI/TUI/GUI design
    setup.md                          # Development setup
    reviews/                          # Runtime review artifacts
```

---

## Dependency Graph

```
@mlxts/core -> none
@mlxts/protocols -> none
@mlxts/tokenizers -> none
@mlxts/nn -> @mlxts/core
@mlxts/data -> @mlxts/core
@mlxts/diffusion -> @mlxts/core, @mlxts/nn
@mlxts/optimizers -> @mlxts/core, @mlxts/nn
@mlxts/train -> @mlxts/core, @mlxts/nn, @mlxts/optimizers
@mlxts/lora -> @mlxts/core, @mlxts/nn
@mlxts/quantize -> @mlxts/core, @mlxts/nn
@mlxts/transformers -> @mlxts/core, @mlxts/lora, @mlxts/nn, @mlxts/quantize, @mlxts/tokenizers
@mlxts/align -> @mlxts/core, @mlxts/data, @mlxts/lora, @mlxts/nn, @mlxts/tokenizers, @mlxts/train, @mlxts/transformers
@mlxts/serve -> @mlxts/core, @mlxts/protocols, @mlxts/tokenizers, @mlxts/transformers
@mlxts/agent -> @mlxts/protocols
```

---

## Migration Snapshot

The rows below describe the current package extraction state. Future example
work is deferred unless a row says otherwise.

| Current location | Destination | Package |
|-----------------|-------------|---------|
| Legacy core monolith: `dtype.ts` | `packages/core/src/dtype.ts` | `@mlxts/core` |
| Legacy core monolith: `error.ts` | `packages/core/src/error.ts` | `@mlxts/core` |
| Legacy core monolith: `array.ts` | `packages/core/src/array.ts` | `@mlxts/core` |
| Legacy FFI layer: `lib.ts` | `packages/core/src/ffi/lib.ts` | `@mlxts/core` |
| Legacy FFI layer: `symbols.ts` | `packages/core/src/ffi/symbols.ts` | `@mlxts/core` |
| Legacy FFI layer: `index.ts` | `packages/core/src/ffi/index.ts` | `@mlxts/core` |
| Legacy FFI layer: `pointer.ts` | `packages/core/src/ffi/pointer.ts` | `@mlxts/core` |
| Legacy FFI layer: `closure-bridge.ts` | `packages/core/src/ffi/closure-bridge.ts` | `@mlxts/core` |
| Legacy ops layer | `packages/core/src/ops/` | `@mlxts/core` |
| Legacy transforms layer | `packages/core/src/transforms.ts` | `@mlxts/core` |
| Legacy fused-ops layer | `packages/core/src/fast.ts` | `@mlxts/core` |
| Legacy device layer | `packages/core/src/device.ts` | `@mlxts/core` |
| Legacy memory layer | `packages/core/src/memory.ts` | `@mlxts/core` |
| Legacy Metal tooling | `packages/core/src/metal.ts` | `@mlxts/core` |
| Legacy random layer | `packages/core/src/random.ts` | `@mlxts/core` |
| Legacy I/O layer | `packages/core/src/io.ts` | `@mlxts/core` |
| Legacy tree utilities | `packages/core/src/tree.ts` | `@mlxts/core` |
| Legacy shape-format utility | `packages/core/src/format-shape.ts` | `@mlxts/core` |
| Legacy nn layer | `packages/nn/src/` | `@mlxts/nn` |
| Legacy module checkpoint helper | `packages/nn/src/checkpoint.ts` | `@mlxts/nn` |
| Legacy optimizer layer | `packages/optimizers/src/` | `@mlxts/optimizers` |
| `examples/nanogpt/src/train.ts` | `packages/train/src/loop.ts` + `packages/train/src/schedule.ts` + `packages/train/src/gradients.ts` + `packages/train/src/step.ts` | `@mlxts/train` |
| `examples/nanogpt/src/checkpoint.ts` | `packages/train/src/checkpoint.ts` + `packages/train/src/checkpoint-*.ts` | `@mlxts/train` |
| `examples/nanogpt/src/safetensors.ts` | `packages/transformers/src/pretrained/weights.ts` + `@mlxts/core` safetensor readers | `@mlxts/transformers` |
| Former `examples/nanogpt/src/data.ts` | `packages/data/src/text.ts` | `@mlxts/data` |
| Former `examples/nanogpt/src/tokenizer.ts` | `packages/tokenizers/src/char.ts` | `@mlxts/tokenizers` |
| `examples/nanogpt/src/model/` | `examples/nanogpt/src/model/` for now; later dedicated examples repo | Committed in-repo example |
| `examples/nanogpt/src/config.ts` | `examples/nanogpt/src/config.ts` for now; later dedicated examples repo | Committed in-repo example |
| `examples/nanogpt/src/cli.ts` | `examples/nanogpt/src/cli.ts` for now; later dedicated examples repo | Committed in-repo example |
| `examples/nanogpt/src/generate.ts` | `examples/nanogpt/src/generate.ts` for now; later dedicated examples repo | Committed in-repo example |
| `examples/nanogpt/src/run/` | `examples/nanogpt/src/run/` for now; later dedicated examples repo | Committed in-repo example |
| `examples/nanogpt/src/bench/` | `examples/nanogpt/src/bench/` for now; later dedicated examples repo | Committed in-repo example |
| All `*.test.ts` files | Move with their source file | (same as source) |

---

## Package Sizing Principles

- **If it has no consumer yet, don't create it.** Packages are extracted when a second consumer needs them, not when we can imagine one.
- **`@mlxts/core` contains everything MLX-related:** FFI, array, ops, transforms, device, memory, random, I/O, fast fused ops. This is one package because MLX is one coherent system — splitting it would create artificial boundaries.
- **Phase 5 creates:** `core`, `nn`, `optimizers`, `train`, `data`, `tokenizers`. These are already extracted, with `examples/nanogpt` acting as the committed example and regression harness while the package surfaces settle.
- **Phase 7 creates:** `transformers`. Pretrained loading uses official Hugging Face JS packages plus repo-owned helpers under `packages/transformers/src/pretrained/`.
- **Phase 8 surfaces now exist:** `lora`, `align`, and their proof/example surfaces are in-repo while the real-world evidence and CI gates continue to harden.
- **Phase 9 surfaces now exist:** `quantize`, `serve`, and `agent` are in-repo; serving, cache, scheduler, protocol, and model-pool work continue through package-owned surfaces.
- **Phase 9.5 hardens:** agent-operated CLI contracts. Package-owned binaries adopt AXI-shaped finite output before any umbrella CLI centralizes names.
- **Phase 10 creates:** `diffusion`. Scheduler infrastructure, Stable
  Diffusion / SDXL package surfaces, and an example proof command are in place;
  real checkpoint image evidence and additional diffusion/flow families remain
  Phase 10 work. Vision/audio encoders extend `transformers`, not a separate
  package. Generative media (image/video/audio) uses diffusion/flow →
  `@mlxts/diffusion`.
- **Phase 12 creates:** `eval`. This appears when benchmark evaluation lands.
- **`cli` grows incrementally** — subcommands arrive as their backing packages ship, after finite command contracts are already AXI-shaped at the package-owned boundary.

---

## Convenience Meta-Package

Consider publishing an unscoped `mlxts` package that re-exports `@mlxts/core` + `@mlxts/nn` + `@mlxts/optimizers` for beginners who don't want to manage multiple imports.

```ts
// With the meta-package:
import { mx, nn, optim } from "mlxts";

// Without it (fine-grained):
import * as mx from "@mlxts/core";
import * as nn from "@mlxts/nn";
import * as optim from "@mlxts/optimizers";
```

This is a convenience layer only — it adds no code, just re-exports. All real logic lives in the scoped packages.
