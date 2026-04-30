# Remote Image Transport

## Summary

Serving media input now accepts bounded, allowlisted remote HTTP(S) image URLs
in the same protocol-neutral content path as data/base64 images. OpenAI Chat
Completions, OpenResponses, and Anthropic Messages can normalize URL image
inputs into `GenerationContentPart`; the media loader fetches the bytes under a
local transport policy before the existing host decode and Qwen prepared-prompt
path run.

File-id image sources remain rejected until a file-store policy exists. This
tranche does not add remote non-image files, audio, tool-result media, Gemma
media, VLM batching, or diffusion.

## Files Reviewed

- `packages/serve/src/media/remote-image.ts`
- `packages/serve/src/media/remote-image.test.ts`
- `packages/serve/src/media/image.ts`
- `packages/serve/src/media/image.test.ts`
- `packages/serve/src/engine/content.ts`
- `packages/serve/src/engine/index.ts`
- `packages/serve/src/model-loading/server-options.ts`
- `packages/serve/src/model-loading/server.ts`
- `packages/serve/src/model-loading/server.test.ts`
- `packages/serve/src/model-loading/sources.ts`
- `packages/serve/src/model-loading/sources.test.ts`
- `packages/serve/src/cli-options.ts`
- `packages/serve/src/cli.ts`
- `packages/serve/src/cli.test.ts`
- `packages/serve/README.md`
- `PLAN.md`
- `docs/serving-runtime-strategy.md`
- `continuity.md`
- `MEMORY.md`

## Runtime Sensitivity Notes

The runtime-sensitive surface is serving media transport before model-family
prompt preparation. The change does not alter model forward math, Qwen image
preprocessing, cache tensor representation, sampling, stream writing, or
continuous scheduler decode.

Remote URL transport is intentionally bounded:

- protocols are limited to `http` and `https`
- credentials and non-default ports are rejected
- hosts must match the operator-configured exact allowlist
- localhost, `.localhost`, single-label DNS names, `.local`, private/link-local,
  loopback, multicast, documentation, and reserved IP ranges are rejected
- DNS is checked before fetch and every redirect target is revalidated
- redirects are manual and capped
- requests carry an image accept header, timeout, abort signal, content-type
  check, content-length preflight, and streaming byte cap

## Tensor Lifetime Audit

The changed production code allocates host `Uint8Array` image bytes only. It
does not allocate, retain, or dispose `MxArray` values. Existing host decode,
resize, Qwen preprocessing, and prepared-prompt tensor ownership are unchanged.

## Memory / Performance Evidence

- `bun test packages/serve/src/media/remote-image.test.ts packages/serve/src/media/image.test.ts`: passed, `20` tests / `77` assertions.
- `bun test packages/serve/src/cli.test.ts packages/serve/src/model-loading/server.test.ts packages/serve/src/model-loading/sources.test.ts packages/serve/src/media/remote-image.test.ts packages/serve/src/media/image.test.ts`: passed, `49` tests / `254` assertions before the final remote-transport coverage additions.
- `bun test packages/serve`: passed, `346` tests / `1505` assertions.
- `bun run --filter '@mlxts/serve' typecheck`: passed.
- `bun run lint`: passed.
- `bun run check:runtime-review`: passed.
- `bun run check:file-lines`: passed.
- `bun run check:tensor-lifetimes`: passed.
- `bun run check:assertions`: passed.
- `bun run check:coverage`: passed. `@mlxts/serve` coverage is `95.05%` lines and `95.79%` functions.

## Independent Review

Banach reviewed the uncommitted diff and caught that DNS preflight alone was not
a sufficient SSRF boundary, DNS resolution was outside the timeout, timeout
errors could surface as internal errors, and malformed redirect locations could
escape as raw URL errors. The patch now requires an exact operator host
allowlist, time-bounds DNS resolution, maps fetch/body timeouts to `ServeError`,
and rejects malformed redirect locations through the media transport policy.

## Out-of-scope Drift Noticed

None.

## Remaining Risks / Follow-ups

Remote image fetching still depends on the platform fetch resolver after the
pre-fetch DNS check. The exact host allowlist makes remote transport an
operator-declared trust boundary, but the implementation still does not expose
the connected peer address after TLS handshake.

The next Phase 10 serving tranche should evaluate repeated-image Qwen
preparation or visual-embedding cache hardening, with strict `MxArray`
ownership and eviction review.
