# Phase 8 LoRA: Realistic, Readable, and Family-Proven

## Summary

The repo already has real LoRA mechanics:

- `@mlxts/nn` provides `LoRALinear`
- `@mlxts/lora` provides explicit apply / merge / remove / save / load helpers
- `@mlxts/align` provides SFT and DPO recipe helpers
- `examples/train-proof/` already proves the path on an official Llama checkpoint

So the next LoRA phase is **not** "invent LoRA." It is:

1. keep the current readable surfaces
2. harden the adapter path against real model families
3. make the proof path realistic across official checkpoints
4. add interoperability where it is worth it

The reference hierarchy for this work is:

- `peft` for adapter semantics, QLoRA targeting guidance, merge behavior, and checkpoint conventions
- `datasets` for realistic loading and pinned subsets
- `alignment-handbook` for proof recipes and official-model task shapes
- `trl` for SFT / DPO dataset and trainer expectations
- `transformers` for family-specific module naming truth

`ml-intern` is useful workflow inspiration, not the architecture source of truth for LoRA itself.

## What Stays

- Keep LoRA as a **visible transformation**:
  `applyLoRA(model, ...)`, `mergeLoRA(model)`, `removeLoRA(model)`
- Keep `@mlxts/lora` generic and module-tree oriented. It should not become a transformer-only control layer.
- Keep `@mlxts/train` explicit and model-agnostic.
- Keep `@mlxts/align` recipe-level, not framework-like.
- Keep merge explicit and non-magical. The merged model should be an ordinary model tree again.
- Keep QLoRA semantics simple in mlxts: quantized base stays quantized by default, adapters stay trainable, and merge only dequantizes when explicitly requested.

## Implementation Direction

### 1. Add transformer-owned LoRA target presets

Do **not** bury family knowledge inside the generic `@mlxts/lora` traversal layer.

Add a transformer-owned preset resolver that maps a loaded causal LM to explicit target paths or target keys. The generic `applyLoRA()` surface remains unchanged; the preset resolver exists to produce explicit selections for it.

Presets should be:

- `attention`
  - target attention projections only
  - canonical set: query, key, value, output projection
- `attention+mlp`
  - target attention projections plus MLP projections
  - canonical set: query, key, value, output, gate, up, down
- `all-linear`
  - target every `Linear` or `QuantizedLinear` inside the decoder stack
  - exclude embeddings and tied output projection unless explicitly requested

Default usage should be:

- standard LoRA proof/examples: `attention`
- QLoRA proof/examples: `all-linear`

This follows current PEFT guidance more closely than the older "q/v only" convention while staying readable.

### 2. Keep adapter config small and explicit

The first-class mlxts config should stay minimal:

- rank
- alpha
- dropout
- explicit target selection or preset
- optional layer slicing such as `lastLayers`

Do **not** add PEFT-style surface area just because PEFT has it.

Specifically, defer these unless a real mlxts use case demands them:

- `bias` training modes
- `modules_to_save`
- multi-adapter stacking
- adapter composition / arithmetic
- sequence-classification-specific saved heads

For the current roadmap, the real target is CausalLM fine-tuning, not every PEFT feature.

### 3. Treat QLoRA as "LoRA over QuantizedLinear", not a separate framework

mlxts already has the right shape for this:

- quantized base layers are represented directly as `QuantizedLinear`
- non-adapter parameters are frozen before wrapping
- merge preserves quantized base form by default

That means mlxts does **not** need a PyTorch-style `prepare_model_for_kbit_training()` clone as a public concept.

Instead, harden the existing path:

- keep adapter weights in stable training dtype
- prove that untouched quantized base weights remain frozen
- prove that merge returns a quantized base by default and a dense base only when requested

### 4. Add PEFT-compatible adapter I/O as an interoperability path

Current adapter I/O is repo-native:

- `adapter_config.json`
- `adapters.safetensors`
- `format: "mlxts-lora"`

That format should stay supported.

But the next LoRA pass should also add **optional PEFT-compatible import/export** for causal LM adapters:

- `adapter_config.json`
- `adapter_model.safetensors`

This should be an interoperability path, not a forced replacement of the mlxts-native format.

The goal is simple:

- mlxts can load/save its own adapters cleanly
- mlxts can exchange LoRA adapters with the wider Hugging Face ecosystem where path naming matches

### 5. Move from one proof anchor to a proof matrix

The repo should prove LoRA in three tiers.

#### Tier 1: structural matrix for every supported family

Run tiny-snapshot structural coverage across:

- Llama
- Mistral
- Mistral 3 text path
- Gemma
- Gemma 3 text path
- Gemma 4 text path
- Phi

For every family, prove:

- apply adapters
- forward still runs
- save/load adapter round-trip
- merge returns usable base layers
- remove restores base layers

This is the "works across all supported families" floor.

#### Tier 2: official-checkpoint training canaries by family shape

Use small official checkpoints where practical:

- `meta-llama/Llama-3.2-1B-Instruct`
- `google/gemma-3-1b-it`
- `google/gemma-4-E2B-it`
- `microsoft/Phi-4-mini-instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`

Each canary should prove:

- apply LoRA on the official checkpoint
- one short training run decreases held-out loss
- merge produces a generation-equivalent adapter result
- save/load round-trip preserves the adapted behavior

For `mistral3`, keep structural tiny-snapshot coverage plus official load/generation canary until we intentionally adopt a practical text-only proof anchor.

#### Tier 3: canonical CI proof anchor

Keep one CI-sized official anchor as the canonical regression gate:

- `meta-llama/Llama-3.2-1B-Instruct`

That remains the fast, stable gate for:

- LoRA
- QLoRA
- SFT
- DPO

The wider official-checkpoint matrix should run as an explicit proof surface, not as the default CI wall.

## Testing and Proof Expectations

The next LoRA implementation slice should leave the repo with these checks:

- focused unit coverage for `LoRALinear`, traversal, merge/remove, and adapter I/O
- family-matrix structural tests across every supported causal LM family
- official-checkpoint LoRA and QLoRA proof runs on the chosen anchors
- proof output that records:
  - trainable parameter count
  - selected target paths
  - loss before / after
  - merge parity check
  - save/load round-trip check

The long-term expectation is:

- tiny-snapshot family matrix stays in normal test coverage
- official Llama proof stays in the canonical `proof:training` gate
- wider family anchors remain an explicit proof suite run before major Phase 8 releases

## Explicit Defaults

- Keep `applyLoRA()` explicit and visible; do not introduce a hidden PEFT-model wrapper.
- Keep `@mlxts/lora` generic; family-specific target presets belong above it.
- Use `attention` as the default standard-LoRA preset.
- Use `all-linear` as the default QLoRA preset.
- Keep the mlxts-native adapter format, but add PEFT-compatible I/O as an optional second path.
- Do not add bias training, modules-to-save, or multi-adapter management in the first hardening pass.
- Treat Mistral 3 as structural-smoke coverage first, not as a forced heavy official training proof.
