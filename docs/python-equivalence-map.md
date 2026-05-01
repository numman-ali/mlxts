# Python-to-mlxts Equivalence Map

A comprehensive mapping of the Python ML ecosystem to mlxts (@mlxts/* npm scope) -- what exists, what's planned, and what's intentionally out of scope.

## 1. Overview

mlxts is a TypeScript-native ML ecosystem built on Bun, optimized first for Apple Silicon via MLX, with multi-backend support planned for later phases.

The goal is not to replicate Python's entire ML ecosystem line-for-line. It is to provide the building blocks a TypeScript developer needs to train, fine-tune, serve, and evaluate ML models without ever dropping into Python. Where Python's ecosystem grew organically over a decade across multiple competing frameworks, mlxts has the advantage of designing a coherent stack from the ground up -- consistent APIs, shared type safety, and a single runtime.

**Design principles**:
- TypeScript-native from the ground up (not a wrapper around Python)
- Apple Silicon via MLX. No multi-backend abstraction — MLX-native throughout.
- Correctness and clarity over raw performance parity
- One canonical way to do each thing
- Every package is a product, not a prototype

**Current state**: The repo is in a package-first Phase 5 posture. `@mlxts/core`, `@mlxts/nn`, `@mlxts/optimizers`, `@mlxts/train`, `@mlxts/data`, and `@mlxts/tokenizers` all exist as extracted workspace packages and now back the committed in-repo nanoGPT example directly. `examples/nanogpt` stays intentionally thin while the reusable package surfaces settle, and the repo can still split broader examples out later if that becomes useful.

**Prior art**: @frost-beta/mlx provides Node.js MLX bindings with a camelCase API. Transformers.js v4 (Feb 2026) provides inference at ~60 tok/s on M4 via WebGPU. mlxts differentiates on training capability, native MLX performance, and a complete ecosystem -- not just inference.

---

## 2. Equivalence Map

### 2a. Core Compute

| Python Package | What It Does | mlxts Equivalent | Phase | Status | Notes |
|---|---|---|---|---|---|
| **NumPy** | N-dimensional array operations, linear algebra, math | @mlxts/core | 4 (exists) | Exists | MxArray wraps MLX arrays. Covers creation (zeros, ones, arange, full), element-wise ops, reductions, reshaping, slicing, broadcasting. Not a full NumPy clone -- focused on what ML needs. |
| **PyTorch** (tensor layer) | GPU tensors, autograd, eager execution | @mlxts/core | 4 (exists) | Exists | MLX's lazy evaluation model differs from PyTorch's eager mode. Autograd is functional (JAX-style `grad(fn)`) rather than tape-based (`loss.backward()`). This is a feature, not a limitation. |
| **JAX** | Functional transforms (grad, jit, vmap), XLA backend | @mlxts/core | 4 (exists) | Exists | MLX's computation model is closest to JAX. `mx.grad()`, `mx.valueAndGrad()`, and lazy eval all mirror JAX idioms. vmap and custom_vjp are future work. |
| **MLX** (Python) | Apple Silicon GPU compute, unified memory | @mlxts/core | 4 (exists) | Exists | This is the direct equivalent. mlxts binds to the same C++ MLX library via mlx-c, so the GPU kernels and Metal backend are identical. The difference is the language above the FFI boundary. |
| **TensorFlow** | Full ML framework, graph execution, Keras API | No direct equivalent | -- | Not planned | TensorFlow's graph-mode execution is handled by MLX's lazy evaluation. Keras-style high-level APIs emerge from @mlxts/nn. There is no reason to replicate TensorFlow's specific abstractions. |

### 2b. Neural Network Layers and Modules

| Python Package | What It Does | mlxts Equivalent | Phase | Status | Notes |
|---|---|---|---|---|---|
| **torch.nn** | Module system, layers, activations | @mlxts/nn | 4 (exists) | Exists | Module base class with parameter scanning, freeze/unfreeze, train/eval. Linear, Embedding, LayerNorm, GroupNorm, Conv1d, Conv2d, Dropout, GELU, ReLU, SiLU, crossEntropy, MSE. |
| **torch.optim** | SGD, Adam, AdamW, LR schedulers | @mlxts/optimizers | 4 (exists) | Exists | SGD (with momentum, weight decay), Adam, AdamW. LR schedulers (cosine with warmup) arrive in Phase 4. |
| **mlx.nn** | MLX's own nn module system | @mlxts/nn | 4 (exists) | Exists | Our nn layer is a TypeScript rewrite inspired by MLX's own nn design, not a binding to it. Same Module pattern, same parameter tree semantics. |
| **timm** | Pre-trained vision model zoo (ViT, ResNet, EfficientNet) | @mlxts/transformers | 10+ | Future | Vision encoders belong in the transformers package because packages are organized by generation paradigm, not modality. |

### 2c. Model Architectures

| Python Package | What It Does | mlxts Equivalent | Phase | Status | Notes |
|---|---|---|---|---|---|
| **HuggingFace Transformers** | 200k+ pre-trained models and a unified architecture API | @mlxts/transformers | 7 | Exists | The current surface covers text decoders plus the first Qwen image-preparation path. Later autoregressive understanding families remain in the same package. |
| **mlx-lm** | LLM loading, generation, fine-tuning for MLX | @mlxts/transformers + @mlxts/lora + @mlxts/align | 7-8 | Exists / Hardening | Dense text loading and generation exist now. LoRA/QLoRA/SFT/DPO proof surfaces exist and continue to harden against official checkpoints. |
| **mlx-vlm** | Vision-language models for MLX | @mlxts/transformers | 10 | Partial / Future | The first Qwen image-preparation and serving path exists. Broader VLM families remain Phase 10 work in the transformers package, not a separate modality package. |
| **diffusers** | Stable Diffusion, FLUX, Qwen-Image, Z-Image, and broader media generation pipelines | @mlxts/diffusion | 10 | Partial / Hardening | Stable Diffusion / SDXL package surfaces exist: scheduler/config loading, local Diffusers manifest inspection, VAE/UNet construction and loading, sampling, pipeline loading, and an example proof command. FLUX.1 also has package-owned FlowMatch scheduling, transformer, VAE loading/decoding, sampling, and an example proof command. Real checkpoint image evidence plus Qwen-Image, Z-Image-Turbo, SD3/3.5, video, and audio families remain Phase 10 work. |

### 2d. Training Infrastructure

| Python Package | What It Does | mlxts Equivalent | Phase | Status | Notes |
|---|---|---|---|---|---|
| **HF Trainer** | High-level training loop with logging, checkpointing, eval | @mlxts/train | 5 (exists) | Exists | `@mlxts/train` now holds the extracted schedule, loop, gradient, and checkpoint primitives. The reusable surface is intentionally explicit and composable rather than a magic base class. |
| **HF Accelerate** | Multi-GPU/multi-node training orchestration | Not applicable | -- | Not planned | mlxts targets single Apple Silicon machines. See Section 4. |
| **DeepSpeed** | Distributed training, ZeRO optimizer, model parallelism | Not applicable | -- | Not planned | Multi-node distributed training is out of scope. See Section 4. |
| **FSDP** | Fully Sharded Data Parallelism | Not applicable | -- | Not planned | Single-machine focus. See Section 4. |
| **Unsloth** | Faster fine-tuning via kernel optimization | @mlxts/train (built-in) | 8 | Future | Unsloth's value is fast LoRA on consumer hardware. mlxts's fine-tuning will use MLX's fused kernels natively -- the optimization is built into the backend, not bolted on. |
| **Weights & Biases / MLflow** | Experiment tracking, metrics logging | @mlxts/telemetry | 7+ | Future | Phase 4 has structured telemetry in the run manager. A proper experiment tracking integration is later work. W&B API integration is possible without Python. |

### 2e. Fine-Tuning

| Python Package | What It Does | mlxts Equivalent | Phase | Status | Notes |
|---|---|---|---|---|---|
| **PEFT** (LoRA, QLoRA) | Parameter-efficient fine-tuning | @mlxts/lora | 8 | Exists / Hardening | LoRA and QLoRA surfaces exist with adapter injection, preservation checks, and proof reporting. The official-checkpoint proof path continues to harden before being promoted to heavier CI. |
| **TRL** (DPO, PPO, SFT) | Reinforcement learning from human feedback | @mlxts/align | 8+ | Exists / Hardening | SFT and DPO recipe/evaluation helpers exist for the proof surfaces. PPO is lower priority; DPO is the practical alignment target. |

### 2f. Tokenization and Data

| Python Package | What It Does | mlxts Equivalent | Phase | Status | Notes |
|---|---|---|---|---|---|
| **HF tokenizers** | Fast BPE/WordPiece/Unigram tokenization (Rust backend) | @mlxts/tokenizers | 5-7 | Exists | The package now supports char tokenization plus Hugging Face `tokenizer.json`, SentencePiece `.model`, and `tekken.json` loading for the current decoder families. |
| **tiktoken** | OpenAI's BPE tokenizer | @mlxts/tokenizers | -- | Not planned | The tokenizer package focuses on Hugging Face tokenizer artifacts (`tokenizer.json`, `tokenizer.model`, `tekken.json`) rather than a dedicated tiktoken compatibility layer. |
| **SentencePiece** | Google's subword tokenizer | @mlxts/tokenizers | 7 | Planned | SentencePiece model files can be parsed. Many models that "use SentencePiece" actually ship HF tokenizer.json files, which we already plan to support. |
| **HF Datasets** | Dataset loading, streaming, preprocessing | @mlxts/data | 5 (exists) | Exists | `@mlxts/data` now holds the extracted text-data loading and batching helpers. Structured dataset loading (HF Hub REST API, Parquet) still comes later. |
| **Data Collators** | Batch assembly, padding, masking | @mlxts/data | 5+ | Planned | Collation logic lives in the data pipeline. Padding and attention mask generation are part of the training infrastructure. |

### 2g. Inference and Serving

| Python Package | What It Does | mlxts Equivalent | Phase | Status | Notes |
|---|---|---|---|---|---|
| **vLLM** | High-throughput LLM serving with PagedAttention | @mlxts/serve | 9 | Exists / Future backend | `@mlxts/serve` already provides OpenAI-compatible serving, streaming, admission, prompt-prefix cache, and cache-generic continuous scheduling. Paged/batch-native cache backends remain future Phase 9 work. |
| **llama.cpp** | CPU/GPU inference for quantized LLMs (GGUF) | @mlxts/quantize + @mlxts/serve | 7-9 | Planned | GGUF support is now future quantization/interoperability work rather than a standalone hub package. llama.cpp's value is broad hardware support; mlxts uses MLX natively but GGUF loading still matters for the quantized ecosystem. |
| **Ollama** | Local LLM runner with REST API | @mlxts/serve | 9 | Exists / Hardening | `@mlxts/serve` provides REST APIs, model management, streaming generation, local model-root discovery, and lazy model loading on Bun's native HTTP server. |
| **TGI** (Text Generation Inference) | HuggingFace's production inference server | @mlxts/serve | 9 | Partial / Hardening | Streaming, continuous scheduling, OpenResponses, Anthropic Messages, tools, and benchmark/report surfaces exist in bounded form. Broader production scheduling and protocol breadth remain Phase 9 work. |
| **TensorRT-LLM** | NVIDIA-optimized LLM inference | Not applicable | -- | Not planned | NVIDIA-exclusive. See Section 4. |
| **SGLang** | Structured generation, constrained decoding | @mlxts/serve | 9+ | Future | Structured generation (JSON mode, grammar-constrained decoding) is a feature of the serving layer, not a separate package. |

### 2h. Quantization

| Python Package | What It Does | mlxts Equivalent | Phase | Status | Notes |
|---|---|---|---|---|---|
| **MLX quantize** | MLX's built-in quantization (4-bit, 8-bit) | @mlxts/core now, `@mlxts/quantize` later | 6.5 / 9 | Partial | Phase 6.5 exposes the raw MLX quantize/dequantize bindings in `@mlxts/core`. The dedicated `@mlxts/quantize` package is still the later ergonomics and GGUF-focused phase. |
| **bitsandbytes** | CUDA 4-bit/8-bit quantization (QLoRA foundation) | Not applicable | -- | Not planned | CUDA-only. See Section 4. MLX's native quantization covers the same use cases on Apple Silicon. |
| **GPTQ** | Post-training quantization for GPU inference | @mlxts/quantize | 9 | Future | GPTQ model weights can be loaded. The quantization algorithm itself can be implemented against MLX ops. |
| **AWQ** | Activation-aware weight quantization | @mlxts/quantize | 9 | Future | AWQ is a quantization format. Loading pre-quantized AWQ weights is the practical need. |
| **GGUF** (format) | llama.cpp's quantized model format | @mlxts/quantize | 7-9 | Planned | GGUF is a file format, not a runtime. The future implementation belongs with quantization/interoperability work, not a standalone hub package. |

### 2i. Hub and Interoperability

| Python Package | What It Does | mlxts Equivalent | Phase | Status | Notes |
|---|---|---|---|---|---|
| **HuggingFace Hub** | Model/dataset repository, download, upload | official `@huggingface/hub` + `@mlxts/transformers` | 7 | Exists | mlxts now uses the official Hugging Face JavaScript hub client directly from the transformers pretrained loading surface. |
| **safetensors** | Safe, fast tensor serialization format | `@mlxts/core` + `@mlxts/transformers` | 7 | Exists | mlxts reads sharded safetensors snapshots directly and iterates weights shard by shard for pretrained loading. |
| **ONNX** | Cross-framework model interchange format | @mlxts/onnx | 11+ | Future | ONNX import would allow loading models from PyTorch/TensorFlow. Lower priority than safetensors and GGUF, which cover the practical model loading needs. |
| **ONNX Runtime** | Optimized inference engine for ONNX models | @mlxts/onnx | 11+ | Future | If ONNX loading is implemented, ORT-style optimized execution is possible via MLX. |

### 2j. Evaluation and Benchmarks

| Python Package | What It Does | mlxts Equivalent | Phase | Status | Notes |
|---|---|---|---|---|---|
| **lm-eval-harness** | Standardized LLM evaluation (MMLU, HellaSwag, etc.) | @mlxts/eval | 12 | Planned | Phase 12 builds an evaluation framework. The core need is running standard benchmarks against mlxts models to verify correctness and compare performance. |
| **HELM** | Holistic evaluation of language models | @mlxts/eval | 12 | Planned | HELM's evaluation scenarios can be implemented as @mlxts/eval tasks. The scoring infrastructure is shared. |

### 2k. Applications and Frameworks

| Python Package | What It Does | mlxts Equivalent | Phase | Status | Notes |
|---|---|---|---|---|---|
| **LangChain** | LLM application framework (chains, agents, RAG) | Not a mlxts concern | -- | Not planned | LangChain is an application framework, not an ML framework. TypeScript already has LangChain.js. mlxts provides the inference backend that LangChain.js or similar frameworks can call via OpenAI-compatible APIs. |
| **LlamaIndex** | Data indexing and RAG framework | Not a mlxts concern | -- | Not planned | Same rationale as LangChain. LlamaIndex has a TypeScript version. mlxts provides the model serving layer. |
| **Gradio** | Quick ML demo UI builder | Not a mlxts concern | -- | Not planned | The GUI surface (Phase post-6) serves a similar demo/exploration purpose, but it is part of the product surface design, not a generic UI builder. |
| **OpenAI API** (compatibility) | Standard API format for LLM serving | @mlxts/serve | 9 | Exists | @mlxts/serve exposes OpenAI-compatible completions, chat completions, and text Responses slices. This is the interop surface for existing tools (LangChain, ChatGPT-compatible clients, etc.). |
| **Anthropic Messages API** (compatibility) | Standard Anthropic chat/message API shape | @mlxts/serve | 9 | Partial | @mlxts/serve exposes bounded Messages text, user image blocks, client tool-use formatting, and SSE framing; richer documents/audio, server tools, and broader content blocks remain future work. |

### 2l. llama.cpp Ecosystem

| Python Package / Tool | What It Does | mlxts Equivalent | Phase | Status | Notes |
|---|---|---|---|---|---|
| **ggml** | C tensor library (llama.cpp's compute backend) | @mlxts/core (MLX backend) | 4 (exists) | Exists (different backend) | mlxts uses MLX, not ggml, as the compute backend. Same role (GPU tensor math), different implementation. MLX's Metal kernels are optimized for Apple Silicon. |
| **llama.cpp** | LLM inference engine | @mlxts/serve | 9 | Planned | mlxts's inference path is native MLX, not llama.cpp. GGUF loading provides model compatibility without depending on llama.cpp as a runtime. |
| **stable-diffusion.cpp** | Image generation inference | @mlxts/diffusion | 10 | Future | Native MLX diffusion is preferable to wrapping a C++ implementation. |

### 2m. WebGPU and Browser ML

| Python Package / Tool | What It Does | mlxts Equivalent | Phase | Status | Notes |
|---|---|---|---|---|---|
| **ONNX Runtime Web** | Browser-based ML inference via WebAssembly/WebGPU | @mlxts/webgpu | Future (see future-backends.md) | Future | WebGPU as a second backend would enable mlxts models to run in browsers. |
| **Transformers.js** | HuggingFace Transformers ported to JS (ONNX Runtime) | @mlxts/webgpu + @mlxts/transformers | Future (see future-backends.md) | Future | Transformers.js uses ONNX Runtime. mlxts's approach would be native WebGPU compute, potentially sharing model architecture code between server (MLX) and browser (WebGPU). |
| **WebLLM** | In-browser LLM inference via WebGPU | @mlxts/webgpu | Future (see future-backends.md) | Future | WebLLM demonstrates that WebGPU LLM inference is viable. mlxts's WebGPU backend would enable the same capability with shared TypeScript model code. |

---

## 3. Layer Dependency Map

### Python Ecosystem Layers

```
Applications        LangChain, LlamaIndex, Gradio, custom apps
                         |
Evaluation          lm-eval-harness, HELM
                         |
Serving             vLLM, TGI, Ollama, SGLang
                         |
Fine-tuning         PEFT (LoRA), TRL (DPO/SFT), Unsloth
                         |
Training            HF Trainer, Accelerate, DeepSpeed
                         |
Models              HF Transformers, mlx-lm, timm, diffusers
                         |
Hub / Interop       HF Hub, safetensors, ONNX, tokenizers, future GGUF
                         |
NN Framework        torch.nn, torch.optim, mlx.nn
                         |
Autograd            torch.autograd, jax.grad, mlx.grad
                         |
Compute             NumPy, PyTorch tensors, JAX arrays, MLX arrays
                         |
Hardware            CUDA, ROCm, Metal, CPU, WebGPU
```

### mlxts Ecosystem Layers (mirrored structure)

```
Applications        OpenAI-compatible clients, LangChain.js, custom Bun apps
                         |
Evaluation          @mlxts/eval                                    [Phase 12]
                         |
Serving             @mlxts/serve (REST API, streaming, KV cache)   [Phase 9]
                         |
Fine-tuning         @mlxts/lora (LoRA, QLoRA)                  [Phase 8]
                    @mlxts/align (DPO, SFT)                        [Phase 8+]
                         |
Training            @mlxts/train (training loop, checkpoints)      [Phase 5]
                    @mlxts/data (datasets, collation)               [Phase 5]
                         |
Models              @mlxts/transformers (text decoders and             [Phase 7+]
                    autoregressive understanding families)
                    @mlxts/diffusion                                   [Phase 10]
                         |
Hub / Interop       official @huggingface/hub + @huggingface/jinja [Phase 7]
                    @mlxts/tokenizers (BPE, tokenizer.json)         [Phase 7]
                    @mlxts/quantize (MLX quantize, GGUF dequant)    [Phase 9]
                         |
NN Framework        @mlxts/nn (Module, layers, activations, losses)[Phase 4 - exists]
                    @mlxts/optimizers (SGD, Adam, AdamW)            [Phase 4 - exists]
                         |
Compute             @mlxts/core (MxArray, ops, autograd, random)   [Phase 4 - exists]
                         |
Hardware            MLX (Metal, Apple Silicon)                      [Phase 4 - exists]
                    WebGPU                                          [Future]
                    CUDA                                            [Future]
```

### Key structural differences from Python

1. **Unified type system.** In Python, you juggle NumPy arrays, PyTorch tensors, JAX arrays, and MLX arrays -- four incompatible types for the same concept. mlxts has one array type (MxArray) from bottom to top.

2. **Single autograd model.** Python has PyTorch's tape-based autograd, JAX's functional transforms, and MLX's own functional grad. mlxts has one: functional grad, matching MLX and JAX.

3. **No framework fragmentation.** In Python, choosing PyTorch vs JAX vs TensorFlow is a fork that determines your entire library ecosystem. mlxts makes that choice once (MLX backend) and every package above it benefits.

4. **Hub interop without Python.** HuggingFace Hub, safetensors, and tokenizer artifacts are documented formats with HTTP or binary specs. Future GGUF support follows the same principle. mlxts accesses them directly, not through Python bindings.

---

## 4. What We Don't Replicate (and Why)

Some Python packages exist to solve problems that don't arise in mlxts's context. Replicating them would be wasted effort.

### Multi-GPU Distributed Training

**Packages**: DeepSpeed, FSDP, HF Accelerate, Megatron-LM, Ray Train

**Why not**: mlxts targets Apple Silicon machines -- single-socket, unified memory. There is no multi-GPU to distribute across. Apple's Ultra chips (M2 Ultra, M4 Ultra) provide more compute within a single unified memory space, not through discrete GPU scaling. If you need multi-node training, you need a different stack.

Note: MLX does have mx.distributed with MPI, Ring, and JACCL (RDMA over Thunderbolt 5) backends. Multi-Mac clusters are a real and growing use case. Distributed training is not in scope for the current roadmap but is technically feasible with MLX and may be added if demand warrants.

### NVIDIA-Specific Tooling

**Packages**: TensorRT-LLM, bitsandbytes, CUDA kernels, cuDNN, Triton (OpenAI's GPU compiler)

**Why not**: These are CUDA-exclusive. They don't run on Apple Silicon. MLX's Metal backend provides the equivalent GPU acceleration for our target hardware. If a future CUDA backend is added, some of these become relevant -- but they would be backend-specific optimizations, not core abstractions.

### Legacy Compatibility Layers

**Packages**: TensorFlow 1.x compatibility, Caffe model loaders, Theano

**Why not**: mlxts has no legacy to maintain. Starting fresh is an advantage. We support the formats that matter today (safetensors and tokenizer artifacts) and stage GGUF with the later quantization work instead of pretending it already exists.

### Python-Specific Infrastructure

**Packages**: pip, conda, virtual environments, Jupyter notebooks, Colab

**Why not**: mlxts runs on Bun. Package management is npm/bun. The development environment is a TypeScript project, not a Python environment. This eliminates an entire category of tooling complexity.

### Generic Application Frameworks

**Packages**: LangChain, LlamaIndex, Gradio, Streamlit

**Why not**: These are application-layer tools, not ML infrastructure. They already have TypeScript equivalents (LangChain.js, LlamaIndex.TS). mlxts provides the model serving layer that these frameworks consume via OpenAI-compatible and Anthropic-compatible API slices. We don't need to rebuild them.

---

## 5. Interoperability Strategy

mlxts does not exist in isolation. The Python ML ecosystem has produced millions of trained models, datasets, and tokenizer configurations. Our interop strategy is: **consume the artifacts, not the runtime.**

### safetensors Loading

**What**: Load pre-trained model weights from HuggingFace models directly.

**How**: safetensors is a simple binary format -- a JSON header describing tensor names, shapes, dtypes, and byte offsets, followed by raw tensor data. mlx-c has built-in safetensors I/O. TypeScript-side, the header is standard JSON parsing.

**When**: Phase 7 (model loading).

**Covers**: Every HuggingFace model that ships safetensors weights (which is most of them now).

### GGUF Loading

**What**: Load quantized models from the llama.cpp ecosystem.

**How**: GGUF is a documented binary format with a header, metadata key-value pairs, and quantized tensor data. A TypeScript parser reads the format; MLX handles the dequantized or quantized compute.

**When**: Phase 9 (quantized inference).

**Covers**: The thousands of quantized models on HuggingFace (TheBloke, bartowski, etc.).

### tokenizer.json Parsing

**What**: Use HuggingFace tokenizer definitions without running the Rust tokenizer library.

**How**: HuggingFace's `tokenizers` library saves its configuration as `tokenizer.json` -- a JSON file describing the vocabulary, merge rules, special tokens, and pre/post-processing. BPE encoding from a merge table is straightforward to implement in TypeScript.

**When**: Phase 7 (tokenizer infrastructure).

**Covers**: Any model that ships a `tokenizer.json` on HuggingFace (GPT-2, LLaMA, Mistral, Phi, Gemma, etc.).

### HuggingFace Hub REST API

**What**: Download models, read model cards, list repository files, and resolve model revisions without Python.

**How**: HF Hub exposes a REST API (`https://huggingface.co/api/...`). Model file downloads are direct HTTPS. Authentication uses a bearer token. No `huggingface_hub` Python package needed.

**When**: Phase 7 (hub client).

**Implementation sketch**:
```typescript
import { resolvePretrainedSource } from "@mlxts/transformers";

const directory = await resolvePretrainedSource("meta-llama/Llama-3-8B", {
  revision: "main",
});
```

### OpenAI-Compatible Serving API

**What**: Serve mlxts models through an API that existing tools (ChatGPT clients, LangChain, Continue.dev, etc.) can consume without modification.

**How**: Implement the `/v1/chat/completions` and `/v1/completions` endpoints with streaming support. Bun's native HTTP server handles this efficiently.

**When**: Phase 9 (serving).

**Covers**: Any client or framework that speaks the OpenAI API format, which is effectively the universal LLM API standard.

### Weight Format Decision Tree

When loading a pre-trained model, mlxts follows this priority:

1. **safetensors** -- preferred. Clean format, exact weight representation, fast loading.
2. **GGUF** -- for quantized models. Broad ecosystem support, smaller files.
3. **MLX native** -- for models converted/quantized within the mlxts ecosystem.
4. **ONNX** -- future, lower priority. For models that only exist in ONNX format.
5. **PyTorch .bin / .pt** -- not supported. These require pickle deserialization, which is a security and complexity hazard. Most models now ship safetensors alternatives.

---

## 6. Research Paper Implementation Path

A mature ML ecosystem is not just about running existing models. It is about implementing new ideas from research papers. Here is how mlxts's building blocks support that workflow.

### Custom Model Architectures

The @mlxts/nn Module system provides composable building blocks:

```typescript
// Implementing a new attention variant from a paper
class SlidingWindowAttention extends Module {
  qProj: Linear;
  kProj: Linear;
  vProj: Linear;
  windowSize: number;

  forward(x: MxArray): MxArray {
    // Custom attention pattern using core ops
    const q = this.qProj.forward(x);
    const k = this.kProj.forward(x);
    const v = this.vProj.forward(x);
    // Window masking, scaled dot product, etc.
    // All built from @mlxts/core primitives
  }
}
```

**Available after Phase 6.5**: RoPE, RMS norm, GQA/MQA-style attention, and SwiGLU as reusable components that new architectures compose.

### Custom Training Loops

Phase 4's training loop is explicit and hand-written by design. This makes it easy to modify for research:

- Custom learning rate schedules (just a function of step count)
- Gradient accumulation (configurable)
- Mixed-precision training (MLX handles this at the backend level)
- Custom logging and checkpointing intervals
- Curriculum learning, dynamic batching, or any other training modification

### Custom Loss Functions and Optimizers

```typescript
// Custom loss: just a function from MxArrays to MxArray
function focalLoss(logits: MxArray, targets: MxArray, gamma: number): MxArray {
  const ce = crossEntropy(logits, targets);
  const pt = exp(negative(ce));
  return mean(multiply(pow(subtract(1, pt), gamma), ce));
}

// Autograd works on any differentiable function
const gradFn = mx.valueAndGrad(focalLoss);
```

Custom optimizers extend the Optimizer base class. The parameter tree and gradient tree structures are well-typed and inspectable.

### Custom FFI Extensions for Novel Kernels

When a paper requires a truly novel GPU operation that MLX doesn't provide:

1. **First**: compose from existing ops. MLX's lazy evaluation and kernel fusion often make composed operations fast enough.
2. **If needed**: write a custom Metal kernel and expose it through mlx-c's extension mechanism.
3. **mlxts binds it**: the FFI layer can load additional symbols from custom .dylib files.

This is the same path MLX Python users follow -- most research does not need custom kernels, but the escape hatch exists.

### The Research Workflow

```
Paper describes new method
         |
         v
Identify which building blocks exist
(@mlxts/nn layers, @mlxts/core ops, @mlxts/optimizers)
         |
         v
Implement the novel parts as Module subclasses,
custom loss functions, or custom training logic
         |
         v
Train using @mlxts/train with standard infrastructure
(checkpointing, logging, LR scheduling)
         |
         v
Evaluate using @mlxts/eval against standard benchmarks
         |
         v
Serve using @mlxts/serve for inference
         |
         v
Share weights as safetensors on HuggingFace Hub
```

The goal is that steps 1-2 (understanding what exists and what's novel) take minutes, not days. The ecosystem provides the boring parts so researchers can focus on the interesting parts.

---

## Appendix: Package Phase Summary

| Package | Phase | Status | Dependencies |
|---|---|---|---|
| @mlxts/core | 4 | Exists | MLX (via mlx-c FFI). Includes MxArray, ops, autograd (grad, valueAndGrad), random, eval. |
| @mlxts/nn | 4 | Exists | @mlxts/core |
| @mlxts/optimizers | 4 | Exists | @mlxts/core, @mlxts/nn |
| @mlxts/train | 5 | Exists | @mlxts/core, @mlxts/nn, @mlxts/optimizers |
| @mlxts/data | 5 | Exists | @mlxts/core |
| @mlxts/tokenizers | 5 | Exists | tokenizer.json, SentencePiece, Tekken, and char tokenizer surfaces |
| official `@huggingface/hub` + `@huggingface/jinja` | 7 | Exists | external JS packages used by `@mlxts/transformers` |
| @mlxts/transformers | 7 | Exists | @mlxts/nn, @mlxts/tokenizers, @huggingface/hub, @huggingface/jinja |
| @mlxts/lora | 8 | Exists / Hardening | @mlxts/nn, @mlxts/train, @mlxts/transformers |
| @mlxts/align | 8+ | Exists / Hardening | @mlxts/lora, @mlxts/train, @mlxts/data |
| @mlxts/quantize | 9 | Exists / Hardening | @mlxts/core, GGUF tensor dequantization |
| @mlxts/protocols | 9 | Exists | shared zero-dependency wire helpers for serve and agent |
| @mlxts/serve | 9 | Exists / Hardening | @mlxts/transformers, @mlxts/protocols, @mlxts/quantize |
| @mlxts/agent | 9 | Exists / Hardening | @mlxts/protocols |
| @mlxts/diffusion | 10 | Exists / Hardening | @mlxts/core, @mlxts/nn |
| @mlxts/webgpu | Future | Future | @mlxts/core (WebGPU backend) |
| @mlxts/eval | 12 | Planned | @mlxts/transformers, @mlxts/tokenizers |
| @mlxts/telemetry | 7+ | Future | (standalone) |
| @mlxts/onnx | 11+ | Future | @mlxts/core |
