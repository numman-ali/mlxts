# Runtime Review: Whisper Encoder-Decoder Foundation

## Summary

This tranche adds the first executable Whisper encoder-decoder model surface in
`@mlxts/transformers`. Whisper remains outside the `CausalLM` registry: the new
family path owns audio encoder blocks, text decoder blocks, cross-attention,
tied decoder-embedding logits projection, local loading, and Hugging Face
weight mapping with Conv1d layout transforms.

This is a model-execution foundation, not a production ASR claim. Tokenizer
prompting, greedy transcription, `examples/whisper`, and real decoded-text
evidence remain the next Phase 10 audio tranche.

## Files Reviewed

- `packages/transformers/src/families/whisper/attention.ts`
- `packages/transformers/src/families/whisper/block.ts`
- `packages/transformers/src/families/whisper/load.ts`
- `packages/transformers/src/families/whisper/mlp.ts`
- `packages/transformers/src/families/whisper/model.ts`
- `packages/transformers/src/families/whisper/types.ts`
- `packages/transformers/src/families/whisper/weights.ts`
- `packages/transformers/src/index.ts`

## Tensor Lifetime Audit

Whisper attention keeps projection, reshape, transpose, scale, SDPA, and output
merge tensors visible with `using`. Encoder and decoder blocks keep residual
sublayer intermediates visible, and return one caller-owned tensor per block.
`WhisperModel.run` returns caller-owned encoder and decoder outputs; disposal
helpers free logits, hidden states, and retained hidden-state traces explicitly.

Weight loading keeps each shard tensor owned until either it is assigned into
the model tree or freed on skip/error. Raw Hugging Face Conv1d kernels are
transposed and made contiguous before assignment, with the original tensor freed
after a replacement tensor is created.

`bun run check:tensor-lifetimes` passes with no suspicious nested
tensor-producing calls.

## Memory / Performance Evidence

Focused validation passed:

- `bun test packages/transformers/src/families/whisper`
- `bun run --filter @mlxts/transformers typecheck`
- `bun run check:tensor-lifetimes`

The tests cover tiny encoder-decoder logits execution, malformed audio and
decoder input rejection, Hugging Face weight-name mapping, Conv1d transposition,
missing/mismatched/unexpected weight failures, and local snapshot loading.

`bun run bench:generation` and `bun run bench:generation:parity` were not run
for this tranche because Whisper is an encoder-decoder family outside the
`CausalLM` generation benchmark surface and the patch does not touch the shared
text decoder generation loop.

## Independent Review

Hypatia the 2nd performed a read-only reference pass over MLX Whisper, Hugging
Face Whisper, and the current transformer family patterns. The review
recommended this bounded execution tranche, called out HF Conv1d layout
transposition, tied `proj_out.weight` handling through `Embedding.asLinear()`,
and keeping Whisper out of the CausalLM registry until a real encoder-decoder
contract exists. The implemented patch follows those recommendations.

## Remaining Risks / Follow-ups

- `examples/whisper` still needs an AXI-shaped finite transcription proof.
- Greedy Whisper decoding still needs tokenizer/generation-config handling,
  forced decoder prompt IDs, EOS-list stopping, and real decoded-text evidence.
- Decoder self-attention and cross-attention caching remain future work before
  any production throughput claim.
- Long-form transcription, language detection, timestamps, batching, streaming
  ASR, and serving audio routes remain out of scope for this foundation tranche.
