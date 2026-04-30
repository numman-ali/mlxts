# CLIP Text Weight Loading

## Summary

Added explicit CLIP text encoder loading for `@mlxts/transformers`. The new
loader parses a local or Hub CLIP text config, constructs either
`CLIPTextModel` or `CLIPTextModelWithProjection`, maps Hugging Face
`text_model.*` safetensor names into the package-owned camelCase module tree,
loads `text_projection.weight` only for the projected class, ignores exported
`position_ids`, and reports missing, mismatched, or strict unexpected weights.

This remains transformer-owned. `@mlxts/diffusion` still receives conditioning
tensors only and does not import CLIP, tokenizers, or transformer loaders.

## Files Reviewed

- `packages/transformers/src/families/clip/load.ts`
- `packages/transformers/src/families/clip/weights.ts`
- `packages/transformers/src/index.ts`

## Reference Audit

- `.reference/transformers/src/transformers/models/clip/modeling_clip.py`
  defines the public state-dict names for `CLIPTextModel` and
  `CLIPTextModelWithProjection`: `text_model.embeddings`,
  `text_model.encoder.layers`, `text_model.final_layer_norm`, and optional
  `text_projection.weight`.
- `.reference/mlx-examples/stable_diffusion/stable_diffusion/model_io.py`
  treats projection as an architecture choice rather than as a generic config
  field. The TypeScript loader follows that split by exposing separate plain
  and projected entry points.
- `.reference/mlx-examples/stable_diffusion/stable_diffusion/clip.py`
  confirms that Linear weights do not need a PyTorch-to-MLX transpose for this
  text encoder tree; `[out, in]` matches `@mlxts/nn.Linear`.

## Tensor Lifetime Audit

Checkpoint tensors are assigned into model parameter slots with
`assignWeightPath()`, which frees the replaced parameter before installing the
loaded tensor. Unsupported or ignored checkpoint tensors are freed immediately.
If assignment fails, the staged checkpoint tensor is freed before rethrowing.
Top-level `loadCLIPTextModel*()` disposes the partially loaded model on any
loader error. After successful assignment, the model switches to eval mode and
all parameters are evaluated once through `mxEval()`.

## Memory / Performance Evidence

This tranche adds safetensor hydration and does not make throughput claims.
Loading iterates shards through the existing `iterateSafetensorWeights()` path
instead of materializing a whole checkpoint outside MLX tensor ownership.

`bench:generation` and `bench:generation:parity` were not run for this tranche:
CLIP text loading is not a CausalLM generation path, does not register in the
decoder family registry, and does not modify token generation, cache, sampling,
or serving decode behavior.

## Independent Review

Socrates reviewed the CLIP boundary before the model tranche and identified the
same weight mapping used here: `text_model.embeddings.*`,
`text_model.encoder.layers.N.self_attn.{q,k,v,out}_proj`,
`layer_norm{1,2}`, `mlp.fc{1,2}`, `text_model.final_layer_norm`, and optional
`text_projection.weight`. Beauvoir performed a follow-up read-only review of
the weight/tokenizer boundary during this tranche; tokenizer support remains a
separate follow-up because CLIP BPE needs vocab/merges loading, CLIP regex,
normalization, and `</w>` suffix semantics.

## Validation

- `bun test packages/transformers/src/families/clip`
- `bun run --filter @mlxts/transformers typecheck`
- `bun run check:file-lines`
- `bun run check:tensor-lifetimes`

## Remaining Risks / Follow-ups

- CLIP tokenization is still the next required prompt-conditioning step. The
  tokenizer package needs a CLIP vocab/merges loader and prompt padding/truncate
  helper before real text prompts can drive Stable Diffusion.
- Diffusers root snapshot composition should load the CLIP text encoder from
  component subfolders such as `text_encoder/`; remote full-root Diffusers
  downloads need supported-file selection widened deliberately before claiming
  Hub root support.
- Stable Diffusion prompt conditioning still needs a higher-level composer that
  feeds CLIP outputs into `@mlxts/diffusion` without reversing the dependency.
