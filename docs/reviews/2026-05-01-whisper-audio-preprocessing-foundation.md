# Runtime Review: Whisper Audio Preprocessing Foundation

## Summary

This tranche adds the MLX primitive bindings needed for Whisper-style audio
feature extraction and introduces the first `@mlxts/transformers` Whisper
family surface: config parsing, feature-extractor config parsing, Slaney mel
filter creation, and channel-last log-mel feature preparation. It deliberately
does not add Whisper to the `CausalLM` registry or advertise transcription; the
next tranche can build the encoder-decoder model on top of this preprocessing
path.

## Files Reviewed

- `packages/core/src/ffi/lib.ts`
- `packages/core/src/ffi/symbols.ts`
- `packages/core/src/index.ts`
- `packages/core/src/ops/arithmetic.ts`
- `packages/core/src/ops/fft.ts`
- `packages/core/src/ops/index.ts`
- `packages/core/src/ops/strides.ts`
- `packages/core/src/ops/windows.ts`
- `packages/transformers/src/families/whisper/config.ts`
- `packages/transformers/src/families/whisper/preprocessing.ts`
- `packages/transformers/src/families/whisper/types.ts`
- `packages/transformers/src/index.ts`

## Tensor Lifetime Audit

Tensor-producing core primitives use per-call `OutSlot` ownership through
`readResultArrayWithMetadata`. `asStrided` stores shape and stride buffers as
lexical typed arrays for the duration of the FFI call; `rfft` normalizes the
axis before metadata inference and FFI dispatch. Whisper preprocessing keeps
each disposable intermediate visible with `using`, including audio padding,
framing, windowing, FFT, magnitude, mel projection, and log scaling. The returned
`inputFeatures` tensor is not disposed by the helper and remains caller-owned.

`bun run check:tensor-lifetimes` passes with no suspicious nested
tensor-producing calls.

## Memory / Performance Evidence

Focused validation passed:

- `bun test packages/core/src/ops/ops.test.ts`
- `bun test packages/transformers/src/families/whisper packages/core/src/ops/ops.test.ts`

The Whisper preprocessing tests use a tiny feature-extractor geometry to keep
the gate finite while still exercising pad/trim, reflect framing, native FFT,
mel projection, log compression, and caller-owned disposal. No generation
loop or model attention files were changed, so `bun run bench:generation` and
`bun run bench:generation:parity` were not run for this foundation tranche.

## Independent Review

James the 2nd performed a read-only implementation review of the uncommitted
diff, focused on core binding correctness, tensor lifetime visibility, and
Whisper preprocessing semantics. Two findings were integrated before commit:
`asStrided` now allows MLX-compatible negative strides with explicit offsets,
and Whisper preserves list-shaped `eos_token_id` values instead of silently
collapsing checkpoint truth to the first token.

## Remaining Risks / Follow-ups

- The full `WhisperForConditionalGeneration` encoder-decoder path remains the
  next Phase 10 audio tranche.
- The first real checkpoint proof still needs `examples/whisper/`, tokenizer
  handling, weight mapping, and decoded text output.
- Long-form transcription, timestamps, streaming ASR, batching, and serving
  audio routes remain out of scope for this foundation tranche.
