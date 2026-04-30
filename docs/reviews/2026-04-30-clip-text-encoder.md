# CLIP Text Encoder

## Summary

Added a transformer-owned CLIP text encoder family for Stable Diffusion and
future multimodal composition. The new surface parses Hugging Face CLIP text
configs, constructs plain and projected CLIP text models, keeps CLIP out of the
CausalLM registry, and exposes disposable output helpers for conditioning
tensors. Diffusion remains decoupled: prompt tokenization, text encoding, and
conditioning composition stay outside `@mlxts/diffusion`.

## Files Reviewed

- `packages/transformers/src/families/clip/attention.ts`
- `packages/transformers/src/families/clip/block.ts`
- `packages/transformers/src/families/clip/config.ts`
- `packages/transformers/src/families/clip/mlp.ts`
- `packages/transformers/src/families/clip/model.ts`
- `packages/transformers/src/families/clip/types.ts`
- `packages/transformers/src/index.ts`

## Reference Audit

- `.reference/transformers/src/transformers/models/clip/modeling_clip.py`
  defines token plus position embeddings, biased q/k/v/out projections, causal
  text attention, pre-LN encoder blocks, final layer norm, explicit EOS pooling,
  and the optional projection head. The TypeScript model mirrors those runtime
  semantics without adding a decoder/generation contract.
- `.reference/transformers/src/transformers/models/clip/configuration_clip.py`
  is the config source for defaults and geometry validation.
- `.reference/mlx-examples/stable_diffusion/stable_diffusion/clip.py` confirms
  the MLX execution shape used by Stable Diffusion: CLIP text is encoder-owned
  but causal-masked, and SDXL conditioning needs intermediate hidden states.

## Tensor Lifetime Audit

`CLIPTextModel.run()` owns the embedding/layer hidden cursor and frees each
previous hidden tensor after the next layer returns. Requested hidden states are
retained explicitly and disposed through `disposeCLIPTextModelOutput()`.
`lastHiddenState`, `pooledOutput`, and projected `textEmbeds` are caller-owned
outputs. Error paths free staged final hidden states and retained hidden-state
snapshots before rethrowing. Attention, MLP, pooling, and embeddings keep
runtime tensor intermediates visible through local `using` declarations.

## Memory / Performance Evidence

This tranche adds an encoder construction path and does not make throughput
claims. The attention path uses the existing fused `scaledDotProductAttention`
with the semantic `"causal"` marker, and quick GELU stays a small tensor motif
matching CLIP math rather than widening shared activation contracts. Focused
tests cover shape, pooling, hidden-state collection, projection, and future-token
causal isolation.

`bench:generation` and `bench:generation:parity` were not run for this tranche:
CLIP text encoding is not a CausalLM generation path, does not register in the
decoder family registry, and does not modify token generation, cache, sampling,
or serving decode behavior.

## Independent Review

Socrates reviewed the CLIP text boundary before commit. The review called out
the key decisions implemented here: keep CLIP out of the CausalLM registry, use
an explicit `families/clip/` folder, expose plain and projected text models,
support `quick_gelu`, preserve causal text attention, expose hidden states for
SDXL, and leave tokenizer plus Diffusers snapshot composition for follow-up
tranches. The review also noted that projection is selected by the model class,
not merely by the presence of `projection_dim`.

## Validation

- `bun test packages/transformers/src/families/clip`
- `bun run --filter @mlxts/transformers typecheck`
- `bun run check:file-lines`

## Remaining Risks / Follow-ups

- CLIP text weight loading is not yet implemented. The next transformer tranche
  should map `text_model.*` safetensor names, ignore `position_ids`, and keep
  `text_projection.weight` tied to the projected class.
- CLIP tokenization is not yet product-ready. `@mlxts/tokenizers` still needs
  vocab/merges loading, CLIP normalization, the CLIP regex, `</w>` BPE suffix
  handling, and prompt padding/truncation to the model's position limit.
- Stable Diffusion prompt conditioning still needs a higher-level composition
  surface that feeds tensors into `@mlxts/diffusion` without importing
  transformers from diffusion.
