# Qwen2.5-VL Text Encoder Runtime Review

## Summary

`@mlxts/transformers` now has a lean Qwen2-family text loader for Qwen2 and
the text-decoder portion of Qwen2.5-VL conditional-generation checkpoints. This
is the missing text-conditioning rung for Qwen-Image proof work: it maps
`model.language_model.*` checkpoint tensors into the shared LLaMA-like runtime,
keeps Qwen2 q/k/v biases without inventing an output-projection bias, and
rejects sliding-window Qwen2 shapes until the shared backbone owns that runtime
semantics deliberately.

This does not implement Qwen2.5-VL vision inputs, video inputs, multimodal
placeholder scattering, or generated text product claims for Qwen2.5-VL.

## Files Reviewed

- `packages/transformers/src/families/llama-like/attention.ts`
- `packages/transformers/src/families/llama-like/types.ts`
- `packages/transformers/src/families/qwen2/config.ts`
- `packages/transformers/src/families/qwen2/load.ts`
- `packages/transformers/src/families/qwen2/weights.ts`
- `packages/transformers/src/registry.ts`
- `packages/transformers/src/index.ts`

## Tensor Lifetime Audit

The tranche does not add new tensor-producing forward logic. The only runtime
module change is the bias flag passed to `Linear` construction for the existing
LLaMA-like output projection. Qwen2 config and weight mapping are host-side
translation logic, and loading flows through the existing shard-iterator-first
`loadPreparedCausalLM()` path with established assigned/skipped/error tensor
ownership.

## Memory / Performance Evidence

No performance optimization claim is made. The path reuses the existing
LLaMA-like decoder implementation and does not add extra per-token work for
existing families because `attentionOutputBias` defaults to the previous
`attentionBias` behavior.

Focused validation:

- `bun test packages/transformers/src/families/qwen2/config.test.ts packages/transformers/src/families/config.test.ts`
- `bun run --filter '@mlxts/transformers' typecheck`
- `bun run bench:generation -- --model /Users/numman/.cache/huggingface/hub/models--mlx-community--Llama-3.2-1B-Instruct-4bit/snapshots/08231374eeacb049a0eade7922910865b8fce912 --prompt-tokens 16 --generation-tokens 4 --trials 1`: prompt `1192.073` tok/s, decode `576.020` tok/s, peak memory `0.717` GB, evals/token `1.00`.
- `bun run bench:generation:parity -- --model /Users/numman/.cache/huggingface/hub/models--mlx-community--Llama-3.2-1B-Instruct-4bit/snapshots/08231374eeacb049a0eade7922910865b8fce912 --prompt-tokens 16 --generation-tokens 4 --trials 1 --skip-mlx-lm-reference`: prompt `1236.122` tok/s, decode `635.160` tok/s, peak memory `0.717` GB, evals/token `1.00`; MLX-LM reference was intentionally skipped for this tiny local hot-path smoke.

## Independent Review

Bohr the 2nd reviewed the Qwen-Image proof boundary and identified Qwen2.5-VL
text encoding as the only blocker to an honest prompt-string Qwen-Image command.
Rawls the 2nd reviewed the loader delta and found the Qwen2 `rope_parameters`
variant check; the tranche now rejects non-default `rope_parameters` and derives
`rope_theta` from the record when present.

## Remaining Risks / Follow-ups

- The first Qwen-Image proof command still needs real prompt-template
  conditioning, the Diffusers 34-token prefix drop, and final hidden-state
  extraction wired at the example layer.
- Sliding-window Qwen2 configs remain rejected until the shared LLaMA-like
  runtime owns a layer-pattern cache contract for that family.
- Qwen2.5-VL multimodal image/video inputs remain out of scope for this text
  encoder path.
