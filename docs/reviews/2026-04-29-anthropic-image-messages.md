# Anthropic Image Messages

## Summary

Anthropic Messages now accepts user image content blocks and normalizes them
into the existing protocol-neutral media content shape. The model path is not
new: Qwen conditional checkpoints still use the existing serve-owned media
transport plus `@mlxts/transformers` Qwen image-preparation adapter.

Text-only Anthropic Messages remain `messages` input and keep their existing
continuous-scheduler eligibility. Media-shaped Anthropic requests route as
single-model `media_input`, matching the OpenAI Chat and OpenResponses image
paths.

## Files Reviewed

- `packages/serve/src/protocols/media-content.ts`
- `packages/serve/src/protocols/anthropic-messages-input.ts`
- `packages/serve/src/protocols/anthropic-messages.ts`
- `packages/serve/src/http/route-info.ts`
- `packages/serve/src/http/server.test.ts`
- `packages/serve/src/model-loading/server.test.ts`
- `packages/serve/src/protocols/anthropic-messages.test.ts`
- `packages/serve/README.md`
- `PLAN.md`
- `docs/serving-runtime-strategy.md`

## Runtime Sensitivity Notes

The runtime-sensitive surface is request normalization before serving engine
routing. Anthropic image blocks now produce `GenerationInput.kind = "content"`,
which the transformer engine already routes through `loadContentGenerationRequest`
and the model-owned content adapter.

The change does not alter model forward math, Qwen image preprocessing, cache
tensor representation, sampling, stream writing, or continuous scheduler decode.
Remote URL and file-id image sources are parsed as protocol truth but still
rejected by the local media loader until explicit transport policy exists.

## Tensor Lifetime Audit

This tranche adds host-side JSON normalization and tests. It does not allocate,
retain, or dispose `MxArray` values in production code.

The loaded-model Anthropic image test proves the existing prepared-prompt path
still owns tensor disposal through the content adapter and single-request
generation machinery.

## Memory / Performance Evidence

- `bun test packages/serve/src/protocols/anthropic-messages.test.ts packages/serve/src/http/server.test.ts packages/serve/src/model-loading/server.test.ts`: passed, `61` tests.
- `bun test packages/serve/src/protocols/anthropic-messages.test.ts packages/serve/src/http/server.test.ts packages/serve/src/model-loading/server.test.ts packages/serve/src/engine/content.test.ts packages/serve/src/media/image.test.ts`: passed, `76` tests.
- `bun run --filter '@mlxts/serve' typecheck`: passed.
- `bun test packages/serve`: passed, `310` tests.
- `bun run validate`: passed.
- `bun run regression:qwen-gemma -- --profile real`: passed.

The real regression included the Anthropic stream rungs for both representative
families:

- Qwen `mlx-community/Qwen3.6-27B-4bit` Anthropic stream:
  `continuous:eligible`, `mean_post_ttft_completion_tps=30.972`,
  `peak_memory=15.516 GB`, `max_stream_chunk_gap_ms=78.8`.
- Gemma `google/gemma-4-E2B-it` Anthropic stream:
  `continuous:eligible`, `mean_post_ttft_completion_tps=82.759`,
  `peak_memory=9.308 GB`, `max_stream_chunk_gap_ms=28.9`.

The same real run also preserved long/short fairness budgets after this parser
change: Qwen `32768x128+128x32` passed with `peak_memory=19.277 GB` and
Gemma `5000x128+128x32` passed with `peak_memory=9.840 GB`.

## Independent Review

Boole reviewed the tranche before final validation and agreed that Anthropic
image blocks belong in the protocol parser and should flow through existing
`GenerationContentPart` plus Qwen content-adapter machinery. The review called
out the production file-line cap, which was addressed by extracting
`anthropic-messages-input.ts`, and kept remote images, Files API fetches,
tools, Gemma media, and multimodal continuous batching out of scope.

The implementation shape was checked against Anthropic's Messages documentation:
image blocks use `source.type` values `base64`, `url`, or `file`, with local
bytes represented by `base64`.

## Out-of-scope Drift Noticed

None.

## Remaining Risks / Follow-ups

Remote Anthropic URL images and `file` sources still need a deliberate local
transport policy before they are usable. Anthropic tools, tool results,
documents, audio, Gemma image support, and multimodal continuous batching remain
separate product tranches.
