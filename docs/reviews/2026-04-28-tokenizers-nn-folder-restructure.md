# Tokenizers and NN Folder Restructure Review

## Summary

Moved BPE tokenizer files under `packages/tokenizers/src/bpe/`, moved standard
NN layers under `packages/nn/src/layers/`, and moved quantized NN layers under
`packages/nn/src/quantized/`. Public barrels still expose the same semantic
package API names.

## Files Reviewed

- `packages/nn/src/conv1d.ts`
- `packages/nn/src/dropout.ts`
- `packages/nn/src/embedding.ts`
- `packages/nn/src/grouped-query-attention.ts`
- `packages/nn/src/index.ts`
- `packages/nn/src/layer-norm.ts`
- `packages/nn/src/linear.ts`
- `packages/nn/src/lora-linear.ts`
- `packages/nn/src/quantized-embedding.ts`
- `packages/nn/src/quantized-linear.ts`
- `packages/nn/src/rms-norm.ts`
- `packages/nn/src/rope.ts`
- `packages/nn/src/layers/conv1d.ts`
- `packages/nn/src/layers/dropout.ts`
- `packages/nn/src/layers/embedding.ts`
- `packages/nn/src/layers/grouped-query-attention.ts`
- `packages/nn/src/layers/layer-norm.ts`
- `packages/nn/src/layers/linear.ts`
- `packages/nn/src/layers/lora-linear.ts`
- `packages/nn/src/layers/rms-norm.ts`
- `packages/nn/src/layers/rope.ts`
- `packages/nn/src/quantized/quantized-embedding.ts`
- `packages/nn/src/quantized/quantized-linear.ts`
- `packages/tokenizers/src/bpe-added-tokens.ts`
- `packages/tokenizers/src/bpe-base.ts`
- `packages/tokenizers/src/bpe-load.ts`
- `packages/tokenizers/src/bpe-merges.ts`
- `packages/tokenizers/src/bpe.ts`
- `packages/tokenizers/src/byte-level.ts`
- `packages/tokenizers/src/index.ts`
- `packages/tokenizers/src/load.ts`
- `packages/tokenizers/src/tekken.ts`
- `packages/tokenizers/src/bpe/bpe-added-tokens.ts`
- `packages/tokenizers/src/bpe/bpe-base.ts`
- `packages/tokenizers/src/bpe/bpe-load.ts`
- `packages/tokenizers/src/bpe/bpe-merges.ts`
- `packages/tokenizers/src/bpe/bpe.ts`
- `packages/tokenizers/src/bpe/byte-level.ts`

## Tensor Lifetime Audit

No tensor-producing expressions, parameter ownership rules, cached projection
invalidations, quantized storage layouts, tokenizer matching semantics, or
decode tables were changed intentionally. The production changes are file moves,
relative import updates, public barrel updates, and path-sensitive validation
script updates.

## Memory / Performance Evidence

This tranche makes no performance claim. `bench:generation` and
`bench:generation:parity` were not run because the diff is a folder-only move
with unchanged layer and tokenizer implementations.

Focused validation passed:

- `bun test packages/tokenizers/src packages/nn/src` passed: 171 tests.

## Independent Review

The 2026-04-28 architectural audit identified this as a pure folder-discipline
move. I checked the resulting layout against the package AGENTS boundaries:
BPE stays a tokenizer-internal subpackage with no ML dependencies, standard
layers stay in `nn/src/layers/`, quantized variants stay in
`nn/src/quantized/`, and `Module`/checkpoint/value-and-grad remain at the NN
root.

## Remaining Risks / Follow-ups

Historical review artifacts and memory entries still mention the old paths as
past references. The final continuity and memory update tranche should point
future readers at the new locations without rewriting older review history.
