# @mlxts/transformers

## Performance Workflow

Generation performance is a staged research loop. Work starts with reference
parity, changes one bounded seam, measures, and removes losing experiments.

Reference parity precedes model hot-path changes. Qwen-family work compares
against `mlx_lm/models/qwen3_5.py`, `qwen3_next.py`, `gated_delta.py`,
`cache.py`, and `base.py`.

Qwen capability claims require prompt rungs, output rungs, and long-context
retrieval. `128/128` alone is insufficient.

Semantic model code stays readable. Runtime strategy lives behind helpers such
as native gated-delta, mask builders, quantized projection helpers, and cache
utilities.

## Package Boundaries

`@mlxts/transformers` owns autoregressive architecture truth: configs, weights,
family models, generation, caches, MoE blocks, vision encoders, VLM wrappers,
chat templates, and checkpoint loading. Diffusion belongs to a future
`@mlxts/diffusion` package, not here.

`CausalLM` is the universal autoregressive behavior contract. MoE is a
block-level decoder swap. VLMs compose a vision encoder and projector with a
CausalLM through prepared embeddings. Do not widen `CausalLM` for anticipated
future consumers, and do not fork model identity for cache backend, attention
backend, compile choice, or KV precision.

Lean families (`llama`, `mistral`, `mistral3`, `phi`, `gemma`) stay as
config/weights adapters over `llama-like/`. Full families (`gemma3`, `gemma4`,
`qwen3_5`) own their model, attention, block, MLP, norm, config, weights, and
types. When a full family grows past a small readable surface, split by role:
`runtime/` for compile or shape-keyed helpers, `cache/` for family cache state,
`multimodal/` for vision/media composition, and family-native attention
subfolders where needed.

`infrastructure/cache/` owns cache contracts and snapshot/fork primitives.
Families implement family-specific caches by composing those contracts; no
family imports another family. `infrastructure/generation/` stays model-agnostic
and does not grow family-specific generation rules.

Future attention or cache backends start from semantic infrastructure seams
before touching every family attention file. Paged KV, quantized KV, speculative
decode, and MTP need contracts around cache trim/restore, layer trimmability,
and attention dispatch; they do not become model-config flags.

## Qwen 3.5 / 3.6 Notes

- Native gated-delta is the canonical fast path when Metal and shape constraints
  allow it; the TypeScript recurrence remains the oracle/fallback.
- Non-window cached full-attention prefill uses the `"causal"` SDPA marker,
  not an explicit boolean mask.
- Qwen 3.6 advertises long context through nested
  `text_config.max_position_embeddings`, currently `262144` for the tested
  checkpoint.
- Long-context retrieval benchmarks disable Qwen thinking and grade the
  first non-empty generated answer line while still printing the full response.
  Broad Qwen context-window claims need early, middle, and late marker evidence,
  not only the default late-position needle.
