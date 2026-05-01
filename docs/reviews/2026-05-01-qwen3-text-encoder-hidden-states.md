# Qwen3 Text Encoder Hidden States

## Summary

Added dense Qwen3 as a lean `llama-like/` family adapter so Diffusers-style
`Qwen3ForCausalLM` text encoders can load without widening `CausalLM`. The
shared LLaMA-like backbone now exposes an explicit hidden-state output for
encoder-style conditioning, and the shared attention path supports Qwen3's
per-head query/key RMSNorm weights.

## Files Reviewed

- `packages/transformers/src/families/llama-like/attention.ts`
- `packages/transformers/src/families/llama-like/model.ts`
- `packages/transformers/src/families/llama-like/types.ts`
- `packages/transformers/src/families/qwen3/config.ts`
- `packages/transformers/src/families/qwen3/load.ts`
- `packages/transformers/src/families/qwen3/weights.ts`
- `packages/transformers/src/registry.ts`
- `packages/transformers/src/index.ts`

## Reference Checks

- Hugging Face `Tongyi-MAI/Z-Image-Turbo` publishes `text_encoder/config.json`
  with `architectures: ["Qwen3ForCausalLM"]`, `model_type: "qwen3"`,
  `hidden_size: 2560`, `num_hidden_layers: 36`, and tied embeddings.
- `.reference/transformers/src/transformers/models/qwen3/modeling_qwen3.py`
  applies `q_norm` and `k_norm` over `head_dim` before transposing attention
  heads.
- `.reference/diffusers/src/diffusers/pipelines/z_image/pipeline_z_image.py`
  uses the text encoder hidden states for Z-Image prompt conditioning.

## Tensor Lifetime Audit

`LlamaLikeModel.runWithHiddenStates` returns retained hidden-state tensors owned
by the caller. Error paths free retained hidden states and final outputs. The
ordinary logits path still disposes hidden-state bookkeeping before returning
the final hidden tensor to the caller.

`LlamaLikeAttention` uses explicit `using` scopes for normalized query/key
heads and retains unnormalized heads only when no per-head norm is configured.
Cache-owned key/value views retain the existing ownership contract.

## Memory / Performance Evidence

No throughput claim is made. The new default path keeps query/key norm disabled
for existing LLaMA-like families, and focused tests cover the enabled Qwen3
path. This tranche adds a retained hidden-state extraction path for conditioning
callers; ordinary causal-LM generation still runs through `model.run()` and
returns logits only.

`bench:generation` and `bench:generation:parity` were not run because this
tranche does not change the default execution path for existing registered
generation checkpoints. Qwen3 is newly registered for text-encoder use before
the Z-Image proof command, so there is no prior Qwen3 generation baseline in the
repo to compare.

Validation:

- `bun test packages/transformers/src/families/llama-like/model.test.ts packages/transformers/src/families/qwen3/config.test.ts`
- `bun run --filter @mlxts/transformers typecheck`

## Independent Review

Franklin reviewed the Qwen/Z-Image conditioning seam and flagged that official
Z-Image uses dense `qwen3`, not the existing Qwen 3.5/3.6 hybrid family. The
review also caught Qwen3 `q_norm`/`k_norm` and the hidden-state indexing
semantics needed before Z-Image consumes `hiddenStates[-2]`.

## Remaining Risks / Follow-ups

This tranche enables Qwen3 loading and hidden-state extraction but does not yet
claim a real `Tongyi-MAI/Z-Image-Turbo` image proof. The next tranche wires the
example-owned prompt conditioning and proof command over these surfaces.
