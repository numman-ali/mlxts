# Runtime Review: Whisper Transcription Proof

## Summary

This tranche adds the first finite Whisper transcription proof surface. The
package owns special-token prompting and bounded greedy decoding over prepared
Whisper features; `examples/whisper` owns local WAV decoding and AXI command
composition. This is not a production ASR claim: cached decoder state,
timestamp segmentation, language detection, resampling, and long-form chunking
remain follow-up work.

## Files Reviewed

- `packages/transformers/src/families/whisper/generation.ts`
- `packages/transformers/src/families/whisper/model.ts`
- `packages/transformers/src/families/whisper/tokenizer.ts`
- `packages/transformers/src/index.ts`
- `examples/whisper/index.ts`
- `examples/whisper/wav.ts`

## Tensor Lifetime Audit

`generateWhisperGreedyTranscription` keeps encoder output ownership explicit
and frees it after the decode loop. Each decoder step owns its input-id tensor,
decoder hidden state, projected logits, last-logit slice, and argmax token in
local scope. The example frees WAV audio and prepared log-mel features in
`finally` blocks.

The first proof intentionally recomputes the full decoder prefix per generated
token because Whisper KV cache plumbing is not yet implemented. That keeps the
capability bounded and honest while avoiding a hidden incomplete cache contract.

## Memory / Performance Evidence

- `bun test packages/transformers/src/families/whisper`
- `bun test examples/whisper`
- `bun run --filter @mlxts/transformers typecheck`
- `tsc -p tsconfig.phase10-examples.json`
- `bun run check:tensor-lifetimes`

`bench:generation` and `bench:generation:parity` are not applicable to this
tranche because Whisper remains outside the shared `CausalLM` generation loop.

## Independent Review

Hypatia the 2nd performed a read-only reference pass over MLX Whisper tokenizer
and decode behavior. The review pointed at the start-of-transcript, language,
task, and no-timestamps prompt sequence as the minimum honest proof path, and
warned against claiming timestamped or long-form ASR before cached decode and
segment logic exist.

## Remaining Risks / Follow-ups

- Whisper decode is greedy and finite, with no KV cache reuse yet.
- Audio input accepts 16 kHz WAV only; broad audio containers need a future
  media transport or explicit ffmpeg-backed surface.
- The example does not implement language detection, timestamp rules, beam
  search, compression-ratio fallback, or no-speech filtering.
