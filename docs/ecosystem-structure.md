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
4. **Examples are products.** Every example must actually run, train, and produce results. They are not stubs.
5. **Extensible via FFI.** Custom Metal kernels, C/Rust extensions, and new ops can be added and composed with existing infrastructure.

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

**Source origin:** The entirety of current `packages/mlx-ts/` — FFI, array, ops, transforms, fast, device, memory, metal, random, io, dtype, tree utils, shape utils, error types.

---

### Layer 1: Neural Network Framework

#### `@mlxts/nn`

Neural network layers, Module system, activations, losses.

| Concern | What it provides |
|---------|-----------------|
| Module base | `Module` class with property scanning, freeze/unfreeze, train/eval |
| Linear layers | `Linear` |
| Normalization | `LayerNorm`, `RMSNorm` (when mlx-c fused binding available) |
| Embeddings | `Embedding` with `asLinear()` for weight tying |
| Regularization | `Dropout` |
| Activations | `gelu`, `relu`, `silu`, `sigmoid`, `tanh` (free functions) |
| Losses | `crossEntropy`, `mse` |
| Attention | `MultiHeadAttention`, `CausalSelfAttention` |
| Module autograd | `nn.valueAndGrad` — the module-aware gradient transform |

**Dependencies:** `@mlxts/core` — nn modules import MxArray and ops directly.

**Source origin:** Current `packages/mlx-ts/src/nn/` and attention code from `packages/nanogpt/src/model/causal-self-attention.ts`.

#### `@mlxts/optimizers`

Gradient-based optimizers and learning rate schedules.

| Concern | What it provides |
|---------|-----------------|
| Optimizers | `SGD`, `Adam`, `AdamW`, `Adafactor` (future), `Lion` (future) |
| LR schedules | `cosineAnnealing`, `warmupCosine`, `linearWarmup`, `constant` |
| Optimizer base | `Optimizer` abstract class with state management and disposal |

**Dependencies:** `@mlxts/core`, `@mlxts/nn` — `Optimizer.update()` accepts `Module` for parameter extraction.

**Source origin:** Current `packages/mlx-ts/src/optimizers/` plus LR scheduling from `packages/nanogpt/src/train.ts`.

---

### Layer 2: Training Infrastructure

#### `@mlxts/train`

Model-agnostic training loop, checkpointing, gradient utilities.

| Concern | What it provides |
|---------|-----------------|
| Training loop | `train()` function with callbacks, gradient accumulation, NaN detection |
| Checkpointing | `saveCheckpoint`, `loadCheckpoint`, `applyCheckpoint` (atomic, validated) |
| Gradient utils | `clipGradNorm`, `accumulateGradients`, `scaleGradientTree` |
| Mixed precision | Precision context, dtype casting utilities |
| Metrics | `TrainEvent` discriminated union, step timing, memory telemetry |
| Config | `TrainConfig` base type, validation |

**Dependencies:** `@mlxts/core`, `@mlxts/nn`, `@mlxts/optimizers`

**Source origin:** Model-agnostic parts of current `packages/nanogpt/src/train.ts`, `packages/nanogpt/src/checkpoint.ts`.

#### `@mlxts/data`

Dataset loading, batching, and preprocessing.

| Concern | What it provides |
|---------|-----------------|
| Text data | `loadText`, `prepareData`, `getBatch` |
| Dataset base | `Dataset` interface, `IterableDataset` for streaming |
| Data collation | Padding, batching, attention mask generation |
| Preprocessing | Text normalization, train/val splitting |

**Dependencies:** `@mlxts/core`

**Source origin:** Current `packages/nanogpt/src/data.ts` generalized.

---

### Layer 3: Tokenization and Hub

#### `@mlxts/tokenizers`

Fast tokenization with support for HuggingFace tokenizer formats.

| Concern | What it provides |
|---------|-----------------|
| BPE | Byte-pair encoding from `tokenizer.json` |
| Char tokenizer | Simple character-level tokenizer |
| SentencePiece | SentencePiece `.model` loading (via FFI to C++ lib, or pure TS) |
| tiktoken compat | OpenAI tiktoken encoding support |
| Encoding | Batch encode/decode, offset tracking, special tokens |

**Dependencies:** `@mlxts/core` (minimal — mostly standalone)

**Source origin:** Current `packages/nanogpt/src/tokenizer.ts` (char tokenizer) plus new BPE implementation.

#### `@mlxts/hub`

HuggingFace Hub integration and model format I/O.

| Concern | What it provides |
|---------|-----------------|
| Hub client | Download models/datasets from HuggingFace Hub via REST API |
| safetensors | Read/write safetensors format (via native binding or pure TS) |
| GGUF | Read GGUF model files (parse header, extract tensors, dequantize) |
| Config | Parse HF `config.json`, `generation_config.json` |
| Caching | Local model cache with integrity checking |
| Conversion | HF weight name mapping to mlxts model architectures |

**Dependencies:** `@mlxts/core`, `@mlxts/tokenizers`

---

### Layer 4: Model Architectures

#### `@mlxts/transformers`

Pretrained model architectures — the mlxts equivalent of HuggingFace Transformers.

| Concern | What it provides |
|---------|-----------------|
| GPT-2 family | GPT-2 architecture (from nanoGPT reference implementation) |
| LLaMA family | LLaMA, LLaMA 2, LLaMA 3, Code Llama |
| Mistral family | Mistral, Mixtral (MoE) |
| Phi family | Phi-3, Phi-3.5 |
| Gemma family | Gemma, Gemma 2 |
| Qwen family | Qwen, Qwen2 |
| Auto dispatch | `AutoModel.fromPretrained(modelId)` — config-driven architecture selection |
| Generation | `model.generate()` with KV cache, sampling strategies (temperature, top-k, top-p, min-p) |
| Weight loading | Load from safetensors via `@mlxts/hub` |

**Dependencies:** `@mlxts/core`, `@mlxts/nn`, `@mlxts/hub`, `@mlxts/tokenizers`

**Architecture pattern:** Each model family is one file (e.g., `llama.ts`, `mistral.ts`, `gpt2.ts`). Config-driven instantiation from HF `config.json`. Common generation logic shared. Mirrors mlx-lm's clean design.

#### `@mlxts/diffusion` (Phase 10 — future)

Diffusion model pipelines.

| Concern | What it provides |
|---------|-----------------|
| Schedulers | DDPM, DDIM, DPM-Solver, Euler |
| UNet | UNet2D architecture |
| VAE | Variational autoencoder |
| Pipelines | `StableDiffusionPipeline`, `SDXLPipeline` |
| ControlNet | Conditional generation |

**Dependencies:** `@mlxts/core`, `@mlxts/nn`, `@mlxts/hub`

#### `@mlxts/audio` (Phase 10 — future)

Audio model support.

| Concern | What it provides |
|---------|-----------------|
| Whisper | Speech recognition |
| TTS | Text-to-speech |
| Audio I/O | Mel spectrograms, resampling |

**Dependencies:** `@mlxts/core`, `@mlxts/nn`, `@mlxts/hub`

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

#### `@mlxts/align` (Phase 8 — future)

Alignment and RLHF.

| Concern | What it provides |
|---------|-----------------|
| SFT | Supervised fine-tuning trainer |
| DPO | Direct Preference Optimization |
| Reward modeling | Reward model training |
| Data | Preference pair formatting, chat template support |

**Dependencies:** `@mlxts/core`, `@mlxts/nn`, `@mlxts/train`, `@mlxts/lora`

---

### Layer 6: Inference and Serving

#### `@mlxts/serve`

Production inference server with OpenAI-compatible API.

| Concern | What it provides |
|---------|-----------------|
| API server | `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings` |
| KV cache | PagedAttention-style cache management |
| Batching | Continuous batching for concurrent requests |
| Streaming | Server-sent events for token streaming |
| Model management | Load/unload models, model registry |
| Quantized inference | Run quantized models at full speed |

**Dependencies:** `@mlxts/core`, `@mlxts/nn`, `@mlxts/transformers`, `@mlxts/tokenizers`

**Runtime:** `Bun.serve()` — no Express, no Node HTTP.

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

Command-line tools for the mlxts ecosystem.

| Concern | What it provides |
|---------|-----------------|
| `mlxts train` | Train any model with config-driven setup |
| `mlxts generate` | Interactive text generation |
| `mlxts serve` | Start an inference server |
| `mlxts convert` | Convert HF models to mlxts format |
| `mlxts quantize` | Quantize a model |
| `mlxts eval` | Run benchmarks |
| `mlxts download` | Download from HuggingFace Hub |

**Dependencies:** All packages as needed per subcommand.

---

## Examples Directory

Examples are complete, runnable applications that showcase the ecosystem. They are not library packages — they are user-facing educational projects.

nanoGPT is a workspace package (has its own `package.json` for dependency resolution) but is NOT published to npm. It imports from `@mlxts/*` like any user would.

```
examples/
  nanogpt/                    # The original reference GPT
    package.json              # Workspace package — NOT published to npm
    README.md                 # Educational walkthrough
    train.ts                  # Train GPT on Shakespeare
    generate.ts               # Generate text from trained model
    config.ts                 # GPT-tiny and GPT-small presets
    run/                      # Supervised run manager, soak, and acceptance infra
      manager.ts              #   (operational tooling for the example)
      supervisor.ts
      acceptance.ts
      soak.ts

  llama-chat/                 # Load and chat with LLaMA
    README.md
    chat.ts                   # Interactive chat with a local LLaMA model

  lora-finetune/              # Fine-tune a model with LoRA
    README.md
    finetune.ts               # LoRA fine-tuning on custom data
    merge.ts                  # Merge adapters and export

  stable-diffusion/           # Image generation (Phase 10)
    README.md
    generate.ts               # Generate images from text prompts

  whisper/                    # Speech recognition (Phase 10)
    README.md
    transcribe.ts             # Transcribe audio files

  custom-model/               # Implement a research paper
    README.md                 # "How to implement a custom architecture"
    model.ts                  # Example custom transformer variant
    train.ts                  # Train it using @mlxts/train
```

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
        train.ts                      # Training loop
        checkpoint.ts                 # Save/load/apply checkpoints
        gradient.ts                   # Gradient accumulation, clipping, scaling
        metrics.ts                    # TrainEvent, telemetry
        config.ts                     # TrainConfig base type

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
        dataset.ts                    # Dataset interface
        text.ts                       # Text data loading/batching
        collation.ts                  # Batching, padding

    hub/                              # @mlxts/hub
      package.json
      src/
        index.ts
        client.ts                     # HuggingFace Hub REST client
        safetensors.ts                # safetensors parser
        gguf.ts                       # GGUF parser
        config.ts                     # HF config.json parsing
        cache.ts                      # Local model cache
        convert.ts                    # Weight name mapping

    transformers/                     # @mlxts/transformers
      package.json
      src/
        index.ts
        auto.ts                       # AutoModel, AutoTokenizer dispatch
        generate.ts                   # Generation with KV cache
        models/
          gpt2.ts                     # GPT-2 family
          llama.ts                    # LLaMA family
          mistral.ts                  # Mistral family
          phi.ts                      # Phi family
          gemma.ts                    # Gemma family

    lora/                             # @mlxts/lora
      package.json
      src/
        index.ts
        lora.ts                       # LoRA layer injection
        qlora.ts                      # QLoRA (quantized LoRA)
        merge.ts                      # Merge adapters
        config.ts                     # LoRAConfig

    serve/                            # @mlxts/serve
      package.json
      src/
        index.ts
        server.ts                     # Bun.serve() with OpenAI-compat API
        kv-cache.ts                   # KV cache management
        batching.ts                   # Continuous batching
        streaming.ts                  # SSE token streaming

    quantize/                         # @mlxts/quantize
      package.json
      src/
        index.ts
        quantize.ts                   # Quantization routines
        gguf-export.ts                # GGUF file creation
        calibrate.ts                  # Calibration dataset

    eval/                             # @mlxts/eval
      package.json
      src/
        index.ts
        harness.ts                    # Eval harness runner
        tasks/                        # Individual benchmark tasks
        metrics.ts                    # Metric computation

    cli/                              # @mlxts/cli
      package.json
      src/
        index.ts
        commands/                     # Subcommand implementations

  examples/
    nanogpt/                          # The reference GPT application
    llama-chat/                       # Chat with LLaMA
    lora-finetune/                    # Fine-tuning example
    custom-model/                     # Research paper implementation example

  scripts/                            # Validation gates (repo-level)
    check-coverage.ts
    check-type-assertions.ts
    check-runtime-review.ts
    check-visible-tensor-lifetimes.ts

  docs/                               # Architecture and planning docs
    ecosystem-structure.md            # This document
    backend-abstraction.md            # Multi-backend design
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
@mlxts/core                           # Leaf — MxArray, ops, transforms, FFI, everything MLX
  ├── @mlxts/nn                       # Core only (imports MxArray directly)
  ├── @mlxts/optimizers               # Core + nn (Optimizer.update accepts Module)
  ├── @mlxts/tokenizers               # Core (minimal)
  └── @mlxts/data                     # Core only

@mlxts/train                          # Core + nn + optimizers
@mlxts/hub                            # Core + tokenizers
@mlxts/lora                           # Core + nn

@mlxts/transformers                   # Core + nn + hub + tokenizers
@mlxts/serve                          # Core + nn + transformers + tokenizers
@mlxts/quantize                       # Core + nn
@mlxts/eval                           # Core + transformers + tokenizers + data

@mlxts/cli                            # All packages (imports per subcommand)
```

---

## Migration from Current Structure

| Current location | Destination | Package |
|-----------------|-------------|---------|
| `packages/mlx-ts/src/core/dtype.ts` | `packages/core/src/dtype.ts` | `@mlxts/core` |
| `packages/mlx-ts/src/core/error.ts` | `packages/core/src/error.ts` | `@mlxts/core` |
| `packages/mlx-ts/src/core/array.ts` | `packages/core/src/array.ts` | `@mlxts/core` |
| `packages/mlx-ts/src/core/ffi/lib.ts` | `packages/core/src/ffi/lib.ts` | `@mlxts/core` |
| `packages/mlx-ts/src/core/ffi/symbols.ts` | `packages/core/src/ffi/symbols.ts` | `@mlxts/core` |
| `packages/mlx-ts/src/core/ffi/index.ts` | `packages/core/src/ffi/index.ts` | `@mlxts/core` |
| `packages/mlx-ts/src/core/ffi/pointer.ts` | `packages/core/src/ffi/pointer.ts` | `@mlxts/core` |
| `packages/mlx-ts/src/core/ffi/closure-bridge.ts` | `packages/core/src/ffi/closure-bridge.ts` | `@mlxts/core` |
| `packages/mlx-ts/src/core/ops/` | `packages/core/src/ops/` | `@mlxts/core` |
| `packages/mlx-ts/src/core/transforms.ts` | `packages/core/src/transforms.ts` | `@mlxts/core` |
| `packages/mlx-ts/src/core/fast.ts` | `packages/core/src/fast.ts` | `@mlxts/core` |
| `packages/mlx-ts/src/core/device.ts` | `packages/core/src/device.ts` | `@mlxts/core` |
| `packages/mlx-ts/src/core/memory.ts` | `packages/core/src/memory.ts` | `@mlxts/core` |
| `packages/mlx-ts/src/core/metal.ts` | `packages/core/src/metal.ts` | `@mlxts/core` |
| `packages/mlx-ts/src/core/random.ts` | `packages/core/src/random.ts` | `@mlxts/core` |
| `packages/mlx-ts/src/core/io.ts` | `packages/core/src/io.ts` | `@mlxts/core` |
| `packages/mlx-ts/src/utils/tree.ts` | `packages/core/src/tree.ts` | `@mlxts/core` |
| `packages/mlx-ts/src/utils/format-shape.ts` | `packages/core/src/shape.ts` | `@mlxts/core` |
| `packages/mlx-ts/src/nn/` | `packages/nn/src/` | `@mlxts/nn` |
| `packages/mlx-ts/src/nn/checkpoint.ts` | `packages/nn/src/checkpoint.ts` | `@mlxts/nn` |
| `packages/mlx-ts/src/optimizers/` | `packages/optimizers/src/` | `@mlxts/optimizers` |
| `packages/nanogpt/src/train.ts` | `packages/train/src/train.ts` | `@mlxts/train` |
| `packages/nanogpt/src/checkpoint.ts` | `packages/train/src/checkpoint.ts` | `@mlxts/train` |
| `packages/nanogpt/src/safetensors.ts` | `packages/hub/src/safetensors.ts` | `@mlxts/hub` |
| `packages/nanogpt/src/data.ts` | `packages/data/src/text.ts` | `@mlxts/data` |
| `packages/nanogpt/src/tokenizer.ts` | `packages/tokenizers/src/char.ts` | `@mlxts/tokenizers` |
| `packages/nanogpt/src/model/mlp.ts` | `examples/nanogpt/model/mlp.ts` | Example app |
| `packages/nanogpt/src/model/transformer-block.ts` | `examples/nanogpt/model/transformer-block.ts` | Example app |
| `packages/nanogpt/src/model/causal-self-attention.ts` | `examples/nanogpt/model/causal-self-attention.ts` | Example app |
| `packages/nanogpt/src/model/init.ts` | `examples/nanogpt/model/init.ts` | Example app |
| `packages/nanogpt/src/config.ts` | `examples/nanogpt/config.ts` | Example app |
| `packages/nanogpt/src/cli.ts` | `examples/nanogpt/cli.ts` | Example app |
| `packages/nanogpt/src/generate.ts` | `examples/nanogpt/generate.ts` | Example app |
| `packages/nanogpt/src/optimizer-defaults.ts` | `examples/nanogpt/optimizer-defaults.ts` | Example app |
| `packages/nanogpt/src/run/files.ts` | `examples/nanogpt/run/files.ts` | Example app |
| `packages/nanogpt/src/run/supervisor.ts` | `examples/nanogpt/run/supervisor.ts` | Example app |
| `packages/nanogpt/src/run/acceptance.ts` | `examples/nanogpt/run/acceptance.ts` | Example app |
| `packages/nanogpt/src/run/soak.ts` | `examples/nanogpt/run/soak.ts` | Example app |
| `packages/nanogpt/src/bench/` | `examples/nanogpt/bench/` | Example app |
| All `*.test.ts` files | Move with their source file | (same as source) |

---

## Package Sizing Principles

- **If it has no consumer yet, don't create it.** Packages are extracted when a second consumer needs them, not when we can imagine one.
- **`@mlxts/core` contains everything MLX-related:** FFI, array, ops, transforms, device, memory, random, I/O, fast fused ops. This is one package because MLX is one coherent system — splitting it would create artificial boundaries.
- **Phase 5 creates:** `core`, `nn`, `optimizers`, `train`, `data`, `tokenizers`. These have a proven consumer: the nanoGPT example.
- **Phase 7 creates:** `hub`, `transformers`. These appear when pretrained model loading lands.
- **Phase 8 creates:** `lora`, `align`. These appear when fine-tuning lands.
- **Phase 9 creates:** `serve`, `quantize`. These appear when inference serving lands.
- **Phase 10 creates:** `diffusion`, `audio`. These appear when multi-modal lands.
- **Phase 12 creates:** `eval`. This appears when benchmark evaluation lands.
- **`cli` grows incrementally** — subcommands arrive as their backing packages ship.

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
