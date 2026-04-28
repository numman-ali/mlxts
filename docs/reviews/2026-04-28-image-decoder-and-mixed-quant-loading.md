# Runtime Review: Image Decoder, Media Cache, and Mixed Quant Loading Fixes

## Summary

This tranche fixes real-world integration failures found while testing Qwen
vision and Gemma 4 A4B serving through local agent clients.

The serving image decoder now accepts the 32-bit `BI_BITFIELDS` BMP payloads
that macOS `sips` emits when resizing normal PNG/RGBA images. The previous
parser accepted only older uncompressed BGR BMP layouts, so PNG data URLs could
reach the Qwen media path and then fail during host decoding.

The quantized checkpoint loader now preserves explicit per-path quantization
parameters when it realizes quantized paths discovered from `.scales` tensors.
This keeps mixed Gemma 4 A4B checkpoints honest: split experts can remain
4-bit while router and dense MLP leaves remain 8-bit.

Media prompt-prefix caching is now enabled safely. Content/image requests do
not reuse token-only cache entries; they store ordered media identity keys and
only hit when the token prefix and prior media identity match. Prepared prompts
slice token ids, input embeddings, and position ids together on a cache hit, and
Qwen image RoPE deltas are stored on the Qwen cache/snapshot rather than relying
on stale model-global state.

## Files Reviewed

- `packages/serve/src/media-image.ts`
- `packages/serve/src/media-image.test.ts`
- `packages/serve/src/transformers-engine-content.ts`
- `packages/serve/src/transformers-engine-content.test.ts`
- `packages/serve/src/transformers-engine-generation.ts`
- `packages/serve/src/transformers-engine-prefix-cache.ts`
- `packages/serve/src/transformers-engine-prefix-cache.test.ts`
- `packages/serve/src/transformers-engine.test.ts`
- `packages/serve/src/transformers-engine.ts`
- `packages/transformers/src/families/qwen3_5/cache.ts`
- `packages/transformers/src/families/qwen3_5/conditional.ts`
- `packages/transformers/src/families/qwen3_5/conditional.test.ts`
- `packages/transformers/src/index.ts`
- `packages/transformers/src/load-quantized.ts`
- `packages/transformers/src/load.test.ts`
- `packages/transformers/src/prepared-prompt.ts`
- `packages/transformers/src/prepared-prompt.test.ts`

## Tensor Lifetime Audit

`media-image.ts` is host-side byte parsing and subprocess-backed image decoding
only. It does not create or retain MLX tensor handles. The change widens BMP
metadata parsing for a standard uncompressed bitfield representation and keeps
decoded RGB ownership as a plain `Uint8Array`.

`transformers-engine-content.ts` computes SHA-256 media identity from host
bytes plus deterministic Qwen preprocessor inputs before tensor preparation.
Decoded images remain host arrays until `prepareQwen3_5ImageBatch()` creates
request-local tensors inside the model lane.

`transformers-engine-generation.ts` now slices cache-hit prepared prompts with
`slicePreparedPrompt()` before generation. The sliced suffix owns tensor views
and is disposed separately from the original prepared prompt, so the existing
generation `finally` still releases the full prompt tensors.

`transformers-engine-prefix-cache.ts` still owns prompt-boundary snapshots and
forked cache disposal. Media identity is copied into entries as plain strings.
Media entries only allow full-entry hits, preventing partial reuse across image
boundaries until a richer token/media span key exists.

`families/qwen3_5/cache.ts` stores Qwen multimodal RoPE deltas as cache-owned
request state and clones them through snapshots/forks. `conditional.ts` refreshes
those deltas from provided position ids using the cache's logical offset, so a
forked image cache cannot accidentally use stale deltas from another request.

`load-quantized.ts` remains loader preparation logic. The change only preserves
already-parsed quantization metadata while deciding which module leaves should
be prepared for quantized checkpoint assignment. It does not change checkpoint
tensor ownership: skipped tensors are still freed by the existing load path,
assigned tensors still transfer ownership into the module tree, and error paths
still dispose partially loaded models.

## Memory / Performance Evidence

Focused tests passed:

- `bun test packages/serve/src/media-image.test.ts`
- `bun test packages/serve/src/media-image.test.ts packages/transformers/src/load.test.ts --test-name-pattern "mixed per-path quantization|media image helpers"`
- `bun test packages/serve/src/transformers-engine-prefix-cache.test.ts packages/serve/src/transformers-engine.test.ts packages/serve/src/transformers-engine-content.test.ts packages/transformers/src/prepared-prompt.test.ts packages/transformers/src/families/qwen3_5/conditional.test.ts`

Mechanical gates passed:

- `bun run typecheck`
- `bun run lint`
- `bun run check:file-lines`
- `bun run check:assertions`
- `bun run check:tensor-lifetimes`

Real local Qwen image proof passed after restarting the server with the decoder
fix:

- Server: `bun run packages/serve/src/cli.ts mlx-community/Qwen3.6-27B-4bit --model-id mlx-community/Qwen3.6-27B-4bit --port 8000 --local-files-only --max-generated-tokens 32768 --max-prompt-tokens 262144 --max-total-tokens 262144 --gpu-memory-utilization 0.85`
- Direct request: `/v1/chat/completions` with `data:image/png;base64,...`,
  one text part, one `image_url` part, and
  `chat_template_kwargs.enable_thinking=false`
- Result: HTTP `200`, assistant described the MLX logo, usage
  `prompt_tokens=866`, `completion_tokens=55`, `total_tokens=921`

Pi client proof passed:

- `pi --provider mlxts --model mlx-community/Qwen3.6-27B-4bit --thinking off --offline --tools read,grep,find,ls --print @.tmp/pi-vision-test.png "What is in this image? Answer in one sentence."`
- Result: assistant described the MLX logo in one sentence.

Real media prefix-cache proof passed on the local Qwen endpoint:

- Server: same Qwen command as above, with `--max-generated-tokens 256`
- Request: two identical `/v1/chat/completions` image requests against
  `.tmp/pi-vision-test.png`, `max_tokens=32`, `temperature=0`, and
  `chat_template_kwargs.enable_thinking=false`
- First result: HTTP `200`, duration `6873 ms`, usage
  `prompt_tokens=867`, `completion_tokens=30`, `cached_tokens=0`,
  `cache_write_tokens=866`
- Second result: HTTP `200`, duration `1945 ms`, same answer and usage
  `prompt_tokens=867`, `completion_tokens=30`, `cached_tokens=866`,
  `cache_write_tokens=0`
- Server logs showed the first request `cache miss` plus two prefill chunks
  (`512/866`, `866/866`), and the second request `cache hit` with no prefill
  progress events.

Real Gemma 4 A4B mixed-quant proof passed before this artifact:

- `bun run bench:generation:parity --model mlx-community/gemma-4-26b-a4b-it-4bit --prompt-tokens 128 --generation-tokens 16 --trials 1 --memory-sample-interval 4 --skip-mlx-lm-reference`
- Result: `generation_tps=102.517`, `prompt_tps=485.300`,
  `peak_memory=14.671 GB`, `evals_per_token=1.00`, active memory slope flat.

No new text decode hot-path optimization is claimed here. The media-cache
change reduces repeated prompt prefill for repeated image histories while
keeping cache reuse conservative and content-addressed.

## Independent Review

Rawls the 2nd independently identified the mixed-quant loading failure: realized
rules were rebuilt from discovered `.scales` paths as `{ enabled: true }`, which
dropped explicit per-path `{ bits: 8 }` overrides from Gemma 4 wrapper configs.
The implemented fix preserves translated explicit rule params while still using
discovered checkpoint paths for explicit-only loading.

Averroes the 2nd independently reviewed the image failure and confirmed that
supporting 32-bit `BI_BITFIELDS` BMP parsing is the correct layer. The review
also asked for contiguous-mask validation, pixel-data/header overlap checks, and
an RGBA PNG decoder regression; those checks are included in
`media-image.test.ts`.

Gauss the 2nd independently reviewed the proposed media cache path and flagged
the two critical safety points: token-only cache keys are unsafe for images, and
prepared-prompt cache hits must slice embeddings/position ids rather than just
token ids. That review also flagged Qwen image RoPE deltas as request state that
must not be left model-global; the landed cache/snapshot delta handling follows
that warning.

## Remaining Risks / Follow-ups

Interactive Pi TUI image entry is not the same as CLI `@file` arguments. Typing
`@.tmp/pi-vision-test.png` into an already-running TUI leaves it as text; Pi's
documented interactive image path is clipboard paste via `ctrl+v`, while CLI
startup and `--print` support `@image` arguments. The serving API and Pi
non-interactive path are proven; the next UX pass should document or automate
the TUI image workflow for local mlxts testing.

The serving media path still uses macOS `sips`, which is acceptable for local
Apple Silicon proof but not the final high-throughput media decoder. A future
package-owned decoder should keep the same `DecodedRgbImage` boundary and feed
transformer-owned preprocessing.

Media cache reuse is intentionally conservative: it only reuses whole stored
media prompt snapshots. A future richer key can model exact image-token spans
for safe partial-prefix reuse, but the current behavior is the right product
step for Pi/OpenCode image sessions that append turns after a repeated image
history.
