# Serving and Runtime Strategy

This document defines how `mlxts` should absorb new serving, inference, and
training techniques without turning the repo into runtime-strategy soup.

The goal is simple: a TypeScript developer should be able to train, fine-tune,
serve, inspect, and agentically use serious ML models on Apple Silicon without
falling through to Python or opaque framework magic. The implementation can be
fast and native, but the surface still needs to read like TypeScript.

## North Star

`mlxts` exists to make the ML ecosystem feel first-class from TypeScript:

- local-first on Mac hardware through MLX
- no Python runtime dependency for package behavior
- Hugging Face and community checkpoint interoperability
- readable package seams, not monolithic framework inheritance
- examples that behave like ML workbooks and proof surfaces
- serving and agent loops as product packages, not demos
- native MLX or custom native code when evidence says the boundary needs it

The repo should eventually support dense text models, MoE, multimodal
understanding, diffusion/flow media generation, fine-tuning, serving, and local
agent loops. The package split should make that expansion boring rather than
fragile.

## Layer Responsibilities

`@mlxts/core` owns MLX runtime truth: tensor wrappers, FFI, native build,
streams, transforms, memory controls, and custom native helpers when needed.

`@mlxts/nn` owns semantic neural-network building blocks. Runtime helpers may
exist underneath, but public names should read like math, not execution plans.

`@mlxts/transformers` owns autoregressive model architecture truth: config
parsing, weight loading, tokenizer/chat-template integration, model families,
generation, cache contracts, MoE blocks, vision encoders, and VLM wrappers.

`@mlxts/quantize` owns checkpoint and tensor quantization utilities. It should
not become the serving scheduler and should not fork model identity.

`@mlxts/lora`, `@mlxts/align`, and `@mlxts/train` own fine-tuning primitives and
recipe helpers. Training loops should stay explicit and readable.

`@mlxts/serve` owns model serving: protocol adapters, request normalization,
model routing, admission limits, scheduling, streaming lifecycle, benchmark
surfaces, and operator telemetry. It should not execute tools and should not
hide model-family prompt logic.

`@mlxts/agent` owns agent loops: messages, tools, tool parsing, observations,
max-iteration behavior, CLI presentation, and future approval/sandbox policy.

`examples/*` own workbooks and proof flows. They demonstrate real usage and feed
pressure back into packages, but reusable behavior moves into packages.

`@mlxts/diffusion` will own diffusion and flow generation when that phase
starts. It should not be modeled as an extension of `CausalLM`.

## Strategy Is Not Identity

Keep these three concepts separate:

- model config: checkpoint truth
- runtime strategy: how we choose to run it on this machine
- backend implementation: the concrete mechanism that executes that strategy

Model config includes architecture facts such as layers, heads, rope settings,
attention pattern, recurrent state, sliding windows, and vocab size.

Runtime strategy includes cache backend, cache precision, attention execution,
prefill policy, decode policy, batching policy, memory policy, and compile or
native helper selection.

Backend implementation includes eager TypeScript-composed MLX ops, compiled MLX
closures, mlx-c bindings, private native helpers, custom Metal kernels, and
future cache modules.

Do not create duplicate model configs for managed cache, native cache, quantized
KV, TurboQuant-style KV, FlashAttention-like kernels, or scheduler variants.
Those are execution choices over the same model.

## Strategy Surfaces

Future operator flags should map into typed strategy configuration. They should
not be scattered boolean checks across model-family files.

Provisional strategy axes:

- cache backend: `simple`, `static-batch`, `paged`, `ssd`
- cache precision: `model`, `fp16`, `bf16`, `q8`, `q4`, `turboquant`
- attention backend: `auto`, `sdpa`, `native`, `quantized-kv`
- scheduler: `serial`, `static-batch`, `continuous`
- prefill: `chunked`, `interleaved`, `sparse`
- decode: `greedy`, `sampled`, `speculative`, `mtp`
- memory policy: `admit-only`, `active-guard`, `evicting-pool`

These names are directional, not a public API commitment. A strategy becomes
public only after it has an implementation, tests, benchmark evidence, and a
clear fallback story.

The user-facing rule should be:

- default to `auto` where the system can make a truthful choice
- expose explicit flags for reproducibility and experiments
- reject unsupported combinations before generation starts
- report the selected strategy in `/info`, logs, and benchmark artifacts

## Interface Seams

The stable seams should be designed before the advanced strategy lands.

Cache backends need a shared shape that can represent trimmable KV layers,
non-trimmable recurrent layers, static batch caches, paged blocks, prefix cache
hits, and future quantized storage.

Schedulers need a model-agnostic request lifecycle: waiting, prefill, running,
streaming collector, cancellation, completion, and error. HTTP concurrency is
not the same as token-level continuous batching.

Attention backends need a semantic call site: query, key/value state, mask or
causal marker, scale, and optional cache metadata. The call site should not know
whether the backend uses MLX SDPA, compiled helpers, or a native quantized-KV
kernel.

Decoding strategies need an explicit cache-trim or cache-restore contract before
speculative decoding, MTP, or prompt-lookup drafting can be correct.

Protocol adapters need one internal request model. OpenAI completions, chat
completions, Responses, and Anthropic Messages should normalize into it
directly, not through each other.

Training strategies need the same discipline. SFT, DPO, LoRA, QLoRA, and future
recipe helpers should compose batch, loss, gradients, optimizer, checkpointing,
and evaluation primitives without hiding the loop behind a framework base class.

## Native Code Rule

Native code is a tool, not a personality trait.

Use this order:

1. Check whether MLX or mlx-c already exposes the operation.
2. Try a readable TypeScript composition when the work is host-side or cold.
3. Try `compile({ shapeless: true })` for repeated pure tensor subgraphs.
4. Add a narrow native helper only for a hot semantic stage that evidence shows
   cannot be handled cleanly through MLX/compile.
5. Consider custom Metal only when the strategy requires a new compute path,
   such as attention directly over compressed KV states.

Keep native seams private until the strategy is proven. The public call site
should still read in terms of model behavior.

## Evidence Bar

No runtime strategy earns permanence by sounding modern.

Before a strategy is kept or exposed, record:

- correctness tests against a small oracle
- model-family coverage for at least the families it claims to support
- paired benchmark evidence against the current baseline
- long-context or boundary-sensitive reruns when cache behavior matters
- memory peak, active memory, and cache memory where relevant
- quality evidence when the strategy is lossy or approximate
- an explicit unsupported-combination rejection path

TurboQuant-style KV compression is the right kind of future capability for this
architecture. It should enter as a cache representation plus attention backend,
not as a model-family fork. Because it changes memory layout and may change
attention math, it needs both quality evidence and long-context evidence before
it is an operator-facing default.

## Current Execution Order

The next work should stay ordered around architecture truth:

1. Keep repo alignment current as package surfaces evolve; stale roadmap text is
   a real engineering hazard.
2. Lock serving baselines for Qwen/Gemma across greedy and model-default
   sampled generation, buffered and streaming output, `@1/@2/@4/@8`
   concurrency, staggered arrivals, short/long output rungs, and separate
   client-observed plus server-side stream evidence.
3. Keep the typed internal serving/runtime strategy seam as the path for new
   backend choices. The current seam reports implemented behavior only:
   scheduler `auto`, managed model-precision cache, attention `auto`,
   model-native decoding, streaming decode cadence, and admit-only memory
   preflight.
4. Harden the scheduler: continuous routes now use one model-level reservation
   budget with separate prompt, completion, and aggregate total caps. The next
   passes should add stronger fairness controls and keep explicit per-row
   decode state for sampler, stop, reasoning, and future logits processors.
5. Build cache backends behind stable contracts: dense managed cache first,
   then prefix cache, rotating/max-KV policy, quantized KV, paged KV, and later
   TurboQuant or tiered SSD storage.
6. Use `/metrics` as the production observability baseline. It now covers
   request, generation, scheduler, batch, memory, and streaming lifecycle
   signals; deepen it as cache backends, cancellation state, and scheduler
   fairness gain more first-class state.
7. Expand protocols through the shared request model: fuller Responses,
   Anthropic Messages, structured output/logprobs, then multimodal serving.

That order protects the thing that matters most: every new capability should
make the stack feel more coherent, not more accidental.
