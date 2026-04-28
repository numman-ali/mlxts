# Runtime Review: Qwen Image Serving Path

## Summary

This tranche turns the prior protocol-neutral media seam into a real Qwen image-serving path for checkpoints that advertise `Qwen3_5ForConditionalGeneration` plus `vision_config`.

`@mlxts/serve` remains responsible for protocol/media transport and host image decoding. `@mlxts/transformers` remains responsible for Qwen image preprocessing, image token expansion, visual embeddings, and prepared prompt tensors.

Gemma image input is still rejected honestly until Gemma-family image preparation exists in `@mlxts/transformers`.

The serving product surface now advertises the capability narrowly: Qwen image data URLs are supported through OpenAI Chat Completions and Responses when the loaded checkpoint exposes the media adapter; Anthropic Messages and coding-agent client metadata remain text-only until those paths are implemented and smoked.

## Files Reviewed

- `packages/serve/src/index.ts`
- `packages/serve/src/media-image.ts`
- `packages/serve/src/model-server-options.ts`
- `packages/serve/src/model-server.ts`
- `packages/serve/src/model-sources.ts`
- `packages/serve/src/server-anthropic-messages.ts`
- `packages/serve/src/server-json.ts`
- `packages/serve/src/server-responses.ts`
- `packages/serve/src/server-info.ts`
- `packages/serve/src/server.ts`
- `packages/serve/src/transformers-engine-content.ts`
- `packages/serve/src/transformers-engine-generation.ts`
- `packages/serve/src/transformers-engine.ts`
- `packages/transformers/src/families/qwen3_5/conditional.ts`
- `packages/transformers/src/families/qwen3_5/load.ts`
- `packages/transformers/src/index.ts`

## Tensor Lifetime Audit

`media-image.ts` is host-only byte loading and BMP parsing. It does not create tensor handles. The macOS `sips` calls are asynchronous and tied to the request abort signal so cancellation can kill the subprocess rather than waiting for a synchronous decode wall.

`transformers-engine-content.ts` performs host media loading and decoding before the model lane is acquired, then creates Qwen prepared prompts inside the lane by calling transformer-owned helpers. The temporary `preparedImages.pixelValues` and `preparedImages.imageGridThw` tensors are freed in a `finally` block after `prepareQwen3_5ImagePrompt` returns.

`transformers-engine-generation.ts` now carries an optional `PreparedPrompt` through the single-request generation path. The prepared prompt's `inputEmbeddings` and `positionIds` are freed in the generation and streaming `finally` blocks. Admission failures during media prompt preparation free the prepared prompt before rethrowing.

`server-json.ts` adds bounded JSON body parsing before protocol normalization. It is host-side byte work only and does not touch MLX tensor handles.

## Memory / Performance Evidence

Focused validation run:

- `bun test packages/serve/src/server-json.test.ts packages/serve/src/media-image.test.ts packages/serve/src/transformers-engine-content.test.ts packages/serve/src/transformers-engine.test.ts packages/serve/src/model-server.test.ts packages/serve/src/model-sources.test.ts packages/serve/src/server.test.ts` passed: 122 tests.
- `bun test packages/serve/src/model-server.test.ts packages/serve/src/server.test.ts` passed after the Responses image and `/info` product-surface follow-up: 50 tests.
- `bun run check:coverage` passed. `@mlxts/serve` coverage was `95.18%` lines / `95.42%` functions, above the required `95%` line / `90%` function thresholds.
- `bun run typecheck` passed across all workspaces.
- `bun run lint` passed.
- `bun run check:file-lines` passed.
- `bun run check:assertions` passed.
- `bun run check:tensor-lifetimes` passed.
- `bun run check:runtime-review` passed.
- Real local Qwen image endpoint smoke passed:
  - Server: `bun run packages/serve/src/cli.ts mlx-community/Qwen3.6-27B-4bit --model-id mlx-community/Qwen3.6-27B-4bit --port 8000 --max-generated-tokens 16 --max-total-tokens 262144 --local-files-only`
  - Request: `/v1/chat/completions` with one tiny red `data:image/bmp;base64,...` image and `chat_template_kwargs.enable_thinking=false`
  - Result: HTTP `200`, assistant content `Red`, usage `prompt_tokens=89`, `completion_tokens=2`, `total_tokens=91`
  - Server route evidence: `route=single`, `reason=media_input`, `input=content`, `prompt_tokens=89 ready in 48.2ms`, peak memory `17.3 GB`

No large-model `bench:generation` or `bench:generation:parity` run is claimed for this tranche. The changed generation path is admission/routing and prepared-prompt plumbing for media requests; text generation performance is covered by existing scheduler tests. The next proof step is a larger image and multi-turn client smoke through Pi/OpenCode rather than another decode-only benchmark.

## Independent Review

James the 2nd reviewed the first working tree for package-boundary fidelity, tensor ownership, and missing tests. The review found five issues:

- The runtime review artifact had not been created yet. This artifact now lists the changed runtime-sensitive files.
- Remote image fetch was unsafe and unbounded. The first local serving path now rejects remote URLs and accepts byte-capped data URLs only.
- Media preparation could hold the model lane during host I/O. Host media loading and decoding now happen before the lane is acquired, with abort checks before loading and model prep.
- Qwen prompt/capability truth was leaking into serve. The image marker and conditional checkpoint detector now live in `@mlxts/transformers`.
- The loader could attach an image adapter without loading the conditional model. Qwen image adapter attachment now requires both the conditional loader and the preprocessor loader.

Sartre the 2nd reviewed the updated tranche after mechanical gates were green. That review found five further ship-readiness issues, all addressed before commit:

- JSON request bodies and image data URLs needed pre-decode byte limits. `server-json.ts` now bounds JSON reads across OpenAI, Responses, and Anthropic routes, and `media-image.ts` estimates base64 decoded size before allocating decoded bytes.
- `sips` decoding was synchronous and could not be interrupted mid-decode. The image decoder now uses asynchronous subprocesses and kills them on abort.
- The serve media layer returned a Qwen-named decoded image type. It now exposes a serve-local `DecodedRgbImage` and passes it structurally into Qwen-specific transformer helpers.
- Single-model `serveModelWithRuntime` conditional Qwen loading lacked a direct test. `model-server.test.ts` now covers the source-detected conditional loader, preprocessor loader, and attached image adapter.
- The validation section was stale. This artifact now reflects the latest focused tests and repo gates.

## Remaining Risks / Follow-ups

The first image decoding path uses macOS `sips`, which is acceptable for local Apple Silicon serving but not the final high-throughput media pipeline. If image serving becomes hot or batch-heavy, the next seam should be a native/image package decoder that still feeds transformer-owned preprocessing.

The content path is single-request only and deliberately avoids continuous/static batching while prepared media tensors are request-local. Future batching work should start in the transformer prepared-prompt contract, not by forcing media prompts through the text batching path.

Remote image URLs are intentionally not supported in this first local path. Enabling them later needs an explicit operator policy for SSRF, redirects, timeout, byte caps, content type, and request cancellation.
