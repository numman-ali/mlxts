# Runtime Review: CLIP Tokenizer

## Summary

Added package-owned CLIP vocab/merges tokenization in `@mlxts/tokenizers` and
wired root transformer snapshots to discover `vocab.json` plus `merges.txt`.
The tokenizer implements the Hugging Face CLIP semantics needed for Stable
Diffusion conditioning: NFC normalization, whitespace collapse, lowercasing,
CLIP regex splitting, byte-level BPE with `</w>`, BOS/EOS/PAD defaults, decode
suffix stripping, and fixed-length prompt preparation.

The boundary remains tokenizer/transformer-owned. `@mlxts/diffusion` still
consumes conditioning tensors only.

## Files Reviewed

- `packages/tokenizers/src/bpe/bpe.ts`
- `packages/tokenizers/src/bpe/clip.ts`
- `packages/tokenizers/src/index.ts`
- `packages/tokenizers/src/load.ts`
- `packages/transformers/src/load.ts`
- `packages/transformers/src/pretrained/snapshot-inspection.ts`
- `packages/transformers/src/pretrained/snapshot-supported-files.ts`
- `packages/transformers/src/pretrained/types.ts`

## Reference Audit

- `.reference/transformers/src/transformers/models/clip/tokenization_clip.py`
  defines CLIP's normalizer, regex split, byte-level BPE, `</w>` end-of-word
  suffix, and Roberta-style BOS/EOS post-processing.
- `.reference/mlx-examples/stable_diffusion/stable_diffusion/tokenizer.py`
  confirms the Stable Diffusion CLIP prompt path uses byte-level BPE pieces with
  the final piece suffixed by `</w>`.
- `.reference/diffusers/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py`
  and the SDXL pipeline use max-length CLIP tokenization with truncation and
  padding before text encoder conditioning.

## Tensor Lifetime Audit

This tranche is host-side tokenization and snapshot metadata plumbing. It does
not allocate or retain `MxArray` tensors, native handles, or MLX streams. The
fixed-length helper returns JavaScript arrays only; tensor materialization stays
with the future conditioning composer.

## Memory / Performance Evidence

No throughput claims are made. The CLIP BPE merge loop is bounded by each regex
segment and uses the same ranked-pair merge helper as the existing BPE
implementation. Tokenizer package tests and transformer loader tests cover
auto-detection, malformed merge files, fixed-length padding/truncation, and root
snapshot tokenizer discovery.

`bench:generation` and `bench:generation:parity` were not run for this tranche:
no CausalLM model path, cache path, sampling path, or serving decode loop
changed.

## Independent Review

Raman performed a read-only second-opinion review for this tranche and
recommended a dedicated `@mlxts/tokenizers` CLIP tokenizer under `src/bpe/`,
generic loader discovery for `vocab.json` plus `merges.txt`, CLIP-specific
normalization/regex/`</w>` semantics, and root transformer snapshot artifact
selection. The implementation follows that boundary and leaves Diffusers
component-subfolder prompt composition for the next tranche.

## Validation

- `bun test packages/tokenizers/src/bpe/clip.test.ts packages/tokenizers/src/load.test.ts`
- `bun test packages/transformers/src/pretrained/snapshot.test.ts packages/transformers/src/load.test.ts --test-name-pattern "CLIP|Tokenizer|snapshot"`
- `bun run --filter @mlxts/tokenizers typecheck`
- `bun run --filter @mlxts/transformers typecheck`
- `bun test packages/tokenizers/src packages/transformers/src/pretrained packages/transformers/src/load.test.ts`

## Remaining Risks / Follow-ups

- Diffusers component-subfolder composition still needs a higher-level prompt
  conditioning loader that pairs `tokenizer/` and `tokenizer_2/` with CLIP text
  encoders without adding transformer/tokenizer imports to `@mlxts/diffusion`.
- Real checkpoint parity should be checked once the composer feeds CLIP outputs
  into the Stable Diffusion pipeline.
