# Design Reasoning

How we think about API design, abstraction choices, and system architecture across the mlxts ecosystem. This document captures the reasoning agents and contributors should apply when facing design decisions — not formatting rules or surface-specific UX (those live in [code-standards.md](./code-standards.md) and [product-surfaces.md](./product-surfaces.md)).

Every package in `@mlxts/*` should feel like it was designed by the same mind. This document is how.

---

## The Core Premise

**Machines write the code. Humans read the control flow.**

TypeScript's type system and fast validation loops (`bun run validate` in seconds) mean that agents can generate and verify code at high throughput. But the human — Nomi, a contributor, a researcher using the library — reads the result. They need to understand what the code does by reading it top to bottom, without chasing through framework internals.

This means:

- **Visible beats hidden.** If something happens during training, serving, or inference, it should be visible in the user's script — not buried in a lifecycle hook three inheritance levels deep.
- **Explicit beats implicit.** Configuration that affects behavior should be passed as arguments, not discovered through class introspection or runtime defaults.
- **Linear beats branching.** A script that reads as a straight sequence of operations is easier to understand than one that dispatches through abstract interfaces. Branching is fine when the domain requires it (model architecture dispatch), not when the framework imposes it (trainer lifecycle phases).

---

## Composition Over Inheritance — And Why

"Composition over inheritance" is a common slogan. Here's what it means concretely in this codebase and why.

### The problem with trainer base classes

HuggingFace's `Trainer` is ~4,000 lines. It has `training_step`, `compute_loss`, `evaluation_loop`, `save_model`, `log`, and dozens of hooks. To customize training, you subclass and override. The problem:

1. **You can't read it.** To understand what happens when you call `.train()`, you need to trace through the base class, your overrides, the callback system, and the argument object — simultaneously.
2. **You can't compose it.** Want gradient accumulation + mixed precision + gradient checkpointing? Each is a different mixin or flag, and their interaction is defined by the framework, not by you.
3. **You can't debug it.** When loss goes NaN, the stack trace shows framework internals, not your training logic.

PyTorch Lightning has the same structural issue. So does Keras (pre-3.0). The pattern is: a framework owns the loop, and you fill in the blanks.

### What we do instead

The user owns the loop. The library provides **composable primitives** that the user calls explicitly.

A training script reads like this:

```typescript
for (let step = 0; step < config.maxSteps; step++) {
  const batch = getBatch(data, config.batchSize);
  const [loss, grads] = computeLossAndGrads(model, batch);
  const clipped = clipGradNorm(grads, config.maxGradNorm);
  optimizer.update(model, clipped);
  mx.eval(model.parameters());

  if (step % config.logInterval === 0) {
    onEvent({ type: "step", step, loss: loss.item() });
  }
}
```

Every step is visible. Gradient accumulation is an explicit loop. Mixed precision is an explicit cast. Checkpointing is an explicit call. The user sees and controls every operation.

### When classes ARE right

Classes are the right choice when identity and state matter:

- **`Module`** — a neural network layer has parameters, a training/eval mode, and recursive structure. That's identity.
- **`Optimizer`** — an optimizer carries per-parameter state (momentum, variance) across steps. That's state.
- **`MxArray`** — a tensor wraps a native handle with lifecycle management. That's ownership.

Classes are wrong when they're used as a framework entry point that hides control flow. The test: if a user has to read the class source to understand what calling a method does, the abstraction is wrong.

### The decision framework

When designing a new API surface, ask:

| Question | If yes → | If no → |
|----------|----------|---------|
| Does this thing have identity that persists across calls? | Class | Function |
| Does the user need to see the control flow to understand what happens? | Functions they call explicitly | Method that encapsulates |
| Are there multiple valid strategies for this operation? | Strategy functions passed as arguments | Don't abstract yet |
| Will users need to combine this with other operations in arbitrary order? | Standalone function | Can be a method |

---

## Influence Chain

Our design draws from specific ecosystems and explicitly rejects others.

### We learn from

- **MLX Python / mlx-lm (Apple):** Functional, explicit, no magic. Training is a visible loop. The library provides ops and utilities, not a framework. This is our primary influence.
- **JAX / Flax (Google):** Functional transforms (`grad`, `jit`, `vmap`). State is explicit. Training loops are user code, not framework code. Flax's `TrainState` is a value, not a base class.
- **PyTorch core (Meta):** The tensor API and autograd design are excellent. The training ecosystem built on top (Lightning, etc.) is what we avoid.

### We reject

- **HuggingFace Trainer / TRL pattern:** Base class with lifecycle hooks. Customization through subclassing and 200-field argument objects. The user fills in blanks instead of writing logic.
- **PyTorch Lightning pattern:** "Just implement `training_step`." Hides the training loop, optimizer step, gradient accumulation, logging, checkpointing — all behind framework methods.
- **Keras Sequential/Functional pattern:** Build a graph of layers, then `.fit()`. The gap between "define model" and "training happens" is a black box.

The rejection is not about quality — these are excellent projects. It's about the user model. Their user is someone who wants training to *happen*. Our user is someone who wants to *see* training happen and *understand* why.

---

## How This Applies Per Layer

### `@mlxts/core` — Tensor Operations

Already correct: pure functions, explicit evaluation, no hidden state. `mx.add(a, b)` does what it says. `mx.eval(x)` forces computation. The lazy evaluation model is the one place where "hidden" computation is acceptable because it's MLX's fundamental design — but we make it visible through documentation and the explicit `eval` call.

### `@mlxts/nn` — Neural Network Modules

Modules are classes (identity + state). But the forward pass is explicit — you call `model.forward(input)`, not `model(input)` with hidden `__call__` hooks. `nn.valueAndGrad` is a function that wraps a loss function, not a method you override.

Pattern to follow:
```typescript
const [loss, grads] = nn.valueAndGrad(model, lossFn)(model.parameters(), input, target);
```

Pattern to avoid:
```typescript
class MyTrainer extends Trainer {
  computeLoss(model, input, target) { ... }  // When does this get called? Who calls it?
}
```

### `@mlxts/optimizers` — Gradient Optimizers

Optimizers are classes (per-parameter state). But `optimizer.update(model, grads)` is a single explicit call, not a framework-managed step. The user controls when it happens, what gradients go in, and what happens after.

### `@mlxts/train` — Training Infrastructure

This is where the reasoning matters most. The package provides:

- **Composable primitives**: gradient clipping, accumulation, scaling — standalone functions the user calls
- **LR schedules**: pure functions from `(step, config) → learningRate`
- **Checkpoint I/O**: save and load functions, not a checkpoint "manager"
- **Training events**: a callback type, not a hook system
- **Config types**: plain TypeScript interfaces, not a 200-field argument class

It does NOT provide a `Trainer` base class. The training loop lives in the user's code (or in an example like `examples/nanogpt/`). The package makes writing that loop easy, not invisible.

### `@mlxts/serve` — Inference Serving

Same principle: explicit server configuration, not an opaque "just works" black
box. Serving is now a first-class package surface with protocol adapters,
admission limits, streaming, scheduling, metrics, and endpoint benchmarks.

```typescript
// What we want: visible, configurable, protocol-neutral inside.
const server = serveLoadedModel({
  model,
  tokenizer,
  modelId: "qwen-local",
  port: 8080,
  maxGeneratedTokens: 2048,
  maxPromptTokens: 32768,
});
await server.ready;

// What we avoid: hidden policy.
serve("meta-llama/Llama-3.2-1B", { magic: true });  // What tokenizer? What cache policy? What route semantics?
```

OpenAI completions, chat completions, text Responses, and bounded Anthropic
Messages should normalize directly into the shared generation request model.
Adapters own wire shape. Engines own generation.

### `@mlxts/lora`, `@mlxts/align` — Fine-Tuning

LoRA injection should be a visible transformation:
```typescript
const adapted = applyLoRA(model, { targetLayers: ["attention.wq", "attention.wv"], rank: 8 });
```
Not a decorator or mixin that silently replaces layers.

SFT/DPO helpers in `@mlxts/align` compose training primitives from
`@mlxts/train`; they are recipe helpers, not a different framework. A user who
reads an SFT training script should still see the same
`getBatch -> computeLoss -> getGrads -> update` structure, with the loss
function and data shaping being the SFT/DPO-specific parts.

---

## Graph Shape Is Performance

MLX uses lazy evaluation: tensor operations build a computation graph, and nothing executes on the GPU until `mx.eval()` is called. This means the **structure of the graph is the performance**. Two implementations that produce the same output tensor can have dramatically different throughput depending on how many nodes they create, which GPU kernels they trigger, and how many intermediate tensors they allocate.

This is not intuitive for developers coming from eager-mode frameworks (PyTorch, NumPy) where "correct output = done." In MLX, correct output is necessary but not sufficient. A functionally correct forward pass that creates 376 graph nodes per token will be 3-4x slower than one that creates 64 — even though both produce identical results.

### Why this matters for model families

When the first model family is implemented (e.g., Llama), the hot-path patterns get careful attention because everything is new. When subsequent families arrive (Gemma, Phi, Mistral variants), the focus shifts to functional correctness — making the math right, handling the config differences, getting the weight loading working. The performance patterns from the first family don't automatically transfer.

This creates a specific failure mode: **the first model is fast, subsequent models are functionally correct but slow**, and the gap only shows up in benchmarks after the code is already written. The fix is to make the performance-critical patterns explicit and checkable *before* benchmarking. See [runtime-safety.md § Forward pass performance invariants](./runtime-safety.md#forward-pass-performance-invariants) for the specific rules.

### Concrete examples

| Pattern | Correct but slow | Correct and fast |
|---------|-----------------|-----------------|
| SDPA mask during single-token decode | Dense all-true boolean tensor (routes to slow masked kernel) | `null` (routes to fast maskless kernel) |
| KV cache update per token | `concatenate([existing, new], axis)` — O(n), new allocation each step | Pre-allocated buffer + `sliceUpdateDynamic` — O(1) amortized |
| Multi-op activation (GELU) | 8 separate graph nodes, 8 intermediate tensors | `compile({ shapeless: true })` — 1 fused kernel, 0 intermediates |
| Weight-derived constant (1 + weight) | Recomputed every forward call — extra graph node per layer per token | Computed once after weight loading, stored and reused |

The takeaway: when implementing a new model family, the reference parity audit against mlx-lm is not optional polish — it's how you avoid shipping a 3x regression that's invisible until someone runs the benchmark.

---

## Abstraction Timing

**Don't abstract before the second consumer.** The first implementation teaches you what the API needs. The second consumer proves the abstraction is right.

This is why `@mlxts/train` is extracted when the training code is proven (nanoGPT works), not when the training code is first written. And why pretrained loading now sits inside `@mlxts/transformers`, with the official `@huggingface/hub` package handling remote snapshot transport instead of a separate repo-owned hub package.

Premature abstraction is worse than duplication. Three similar training loops in three examples is better than one generic `TrainingPipeline<TModel, TData, TConfig>` that nobody can read.

The same rule applies to runtime execution choices inside inference code:

- keep runtime selection private until there is a validated winner
- do not widen public API around a hot-path idea that has not earned permanence
- let measured wins decide which execution choices deserve long-term surface
- keep semantic function names about math and model behavior; execution
  strategy such as eager, keyed compile reuse, or native help should stay
  behind those semantic names unless the API itself is explicitly a transform utility

There is a second rule that matters just as much:

**do not let runtime optimization consume the teaching surface of the repo.**

If an optimization makes the main model-family or training flow unreadable, the
optimization is living in the wrong layer. The right answer is usually not "do
less optimization"; it is "move the strategy behind a better seam."

---

## Contract Boundaries

Contracts describe what a thing **does** (predict next tokens from a sequence), not **how** it does it internally (dense MLP vs sparse expert routing, attention variant, norm placement).

### Runtime strategy is not model identity

There is a similar distinction lower in the stack:

- model config describes checkpoint truth
- runtime strategy describes how we execute that checkpoint
- backend implementation is the concrete mechanism that realizes that strategy

That means we do **not** create duplicate model configs for:

- managed cache versus native-assisted cache
- eager helpers versus selective compiled transform reuse
- later, dense KV versus compressed or quantized KV

Those are execution choices, not model identity.

This matters because it keeps the architecture composable. A future
TurboQuant-style KV path should arrive as a runtime/backend pairing between
cache representation and attention compute path, not as "another Gemma
config."

The same naming rule applies lower down in the stack: the semantic function
name should describe the math, while compile or native selection stays an
internal execution choice. We want people reading `swiglu` or `crossEntropy`,
not `compiledSwiglu` or `compiledCrossEntropy`.

The same separation should hold one level higher too:

- reference surfaces explain inference and training flow
- backend surfaces implement execution strategy

This is how the repo can stay both understandable and fast.

### CausalLM is a text generation contract

`CausalLM` defines a model that takes token IDs and returns logits over a vocabulary. This is the right contract for:

- **Dense text models** (LLaMA, Mistral, Gemma, Phi) — the initial Phase 7 families
- **MoE text models** (Mixtral, DeepSeek) — MoE routing happens inside the decoder block, invisible at the contract level. The forward pass is still `(tokenIds, cache?) → logits`.
- **The text decoder inside a vision-language model** — a VLM wraps a CausalLM, it doesn't change what "causal language model" means.

MoE does not require a new contract because the block-level MLP → MoE swap does not change the model's input/output signature. The generation pipeline, KV cache, and sampling all work identically for dense and MoE models.

### Multimodal composes CausalLM, doesn't replace it

A vision-language model:

1. Encodes images via a vision encoder (a separate transformer module)
2. Projects vision embeddings into the text model's embedding space
3. Feeds merged embeddings to `CausalLM.forward()` via an optional `inputEmbeddings` parameter
4. After the initial prefill, generation is pure text — no further image processing

This is why multimodal understanding is an extension of `@mlxts/transformers`, not a separate package. Vision encoders (CLIP, SigLIP, ViT) and VLM wrappers (LLaVA, PaliGemma) are transformer architectures that compose with existing text decoder families.

### Implications for package design

- **MoE** is additive domain work inside `@mlxts/transformers` — new block types and family configs, same contract
- **Multimodal understanding** is an extension of `@mlxts/transformers` — new encoder families and VLM wrappers, same contract
- **Diffusion-based generation** (images, video, audio) is a different generation paradigm with its own contract — `@mlxts/diffusion`
- Don't widen contracts preemptively for consumers that don't exist yet

---

## Generation Paradigms

The mlxts ecosystem recognizes two fundamental generation paradigms. This distinction drives the package boundary between `@mlxts/transformers` and `@mlxts/diffusion`, and is designed to hold as models become fully multimodal.

### Autoregressive generation (`@mlxts/transformers`)

Predict the next token/unit, one step at a time. The model sees everything generated so far and produces the next element.

- **Inference loop**: single forward pass per token, KV cache grows linearly
- **Training objective**: next-token prediction loss
- **Covers**: text generation, multimodal understanding (VLMs), autoregressive image/audio/video token generation (GPT-4o style), encoder-decoder models (Whisper, T5)

All of these share the same fundamental pattern: sequence in → distribution over next element out. The `CausalLM` contract and generation pipeline handle all of them.

### Diffusion/flow generation (`@mlxts/diffusion`)

Iteratively denoise from random noise to structured signal. The model takes a noisy input and predicts the noise (or the clean signal, or a velocity field).

- **Inference loop**: N denoising steps with a scheduler, fixed-size latent tensors
- **Training objective**: noise prediction loss (or flow matching loss, or v-prediction)
- **Covers**: image generation (Stable Diffusion, Flux), video generation (LTX-Video), audio generation (Stable Audio), 3D generation (future)

All of these share the same fundamental pattern: noise + conditioning → denoised signal. The diffusion contract (backbone + scheduler + VAE) handles all of them.

### Why this split is durable

Modalities come and go — a model that generates images today might generate video tomorrow. But the generation paradigm is structural:

- **Autoregressive models** need KV caches, token sampling, and grow memory linearly with sequence length.
- **Diffusion models** need schedulers, fixed-size latent spaces, and have constant memory per denoising step.

These are different enough that sharing a contract would mean either: (a) a contract so generic it's useless, or (b) runtime type-checking to figure out which paradigm you're actually using. Neither is acceptable.

**Packages are named for generation paradigms, not input/output modalities.** Modalities come and go. The paradigm boundary is durable.

The future of fully multimodal models (any modality in, any modality out, on device) maps cleanly:

| Model type | Paradigm | Package |
|---|---|---|
| Autoregressive multimodal (GPT-4o style) | Autoregressive | `@mlxts/transformers` — still next-token prediction, richer vocabulary |
| Diffusion-based media generation | Diffusion/Flow | `@mlxts/diffusion` — same paradigm regardless of output modality |
| Hybrid (understanding via transformer, generation via diffusion) | Both | Composes both packages at the application layer |

---

## The Readability Test

Before shipping any API:

1. **The 30-second test.** Open any file. Can a TypeScript developer understand what it does within 30 seconds?
2. **The top-to-bottom test.** Read a script that uses the API from top to bottom. Does the control flow make sense without jumping to library source?
3. **The debug test.** If something goes wrong, does the stack trace show the user's code, or framework internals?
4. **The composition test.** Can two features be combined by the user writing code, or do they need to configure the framework to compose them?

If any test fails, the abstraction needs rethinking.

---

## Summary

| Principle | Means | Doesn't mean |
|-----------|-------|-------------|
| Visible control flow | User sees every step in their script | No convenience functions allowed |
| Composition over inheritance | Functions and values over base classes | Never use classes |
| Explicit over implicit | Configuration is passed, not discovered | Verbose boilerplate everywhere |
| User owns the loop | Training control flow stays visible; serving policy stays explicit and package-owned | No reusable training utilities or serving helpers |
| Don't abstract early | Wait for the second consumer | Copy-paste forever |
| Contracts describe behavior | CausalLM = "predicts next tokens", regardless of internals | One contract per model variant |
| Paradigm-based packages | `transformers` (autoregressive) and `diffusion` (denoising) | One package per modality |

The goal is not minimalism for its own sake. The goal is that a TypeScript developer can pick up any `@mlxts/*` package, read a usage example, and understand exactly what happens — then modify it with confidence because nothing is hidden.
