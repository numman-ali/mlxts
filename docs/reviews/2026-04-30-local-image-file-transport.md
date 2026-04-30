# Local Image File Transport

## Summary

Serving media input now accepts image `file_id` values only when the operator
configures local image roots. OpenAI Responses image parts and Anthropic image
file sources can normalize into the existing protocol-neutral content path; the
media loader resolves the file ID under the configured roots before the existing
host decode and Qwen prepared-prompt path run.

This tranche does not add `/v1/files`, uploads, non-image file inputs, audio,
stateful Responses continuation, Gemma media, VLM batching, or diffusion.

## Files Reviewed

- `packages/serve/src/media/local-image.ts`
- `packages/serve/src/media/image.ts`
- `packages/serve/src/engine/content.ts`
- `packages/serve/src/engine/index.ts`
- `packages/serve/src/model-loading/server-options.ts`
- `packages/serve/src/model-loading/server.ts`
- `packages/serve/src/model-loading/sources.ts`
- `packages/serve/src/cli-options.ts`
- `packages/serve/src/cli-usage.ts`
- `packages/serve/src/cli.ts`

## Runtime Sensitivity Notes

This tranche enables image `file_id` payloads for local serving only when the
operator configures one or more local image roots. File IDs resolve as relative
paths under those roots, stay image-extension-gated, reject traversal and
absolute paths, and check canonical paths before reading bytes.

Local file IDs default to unsupported input until `localImageRoots` are
configured. Root paths are canonicalized by runtime option resolution before
server start. Request-time reads also canonicalize roots for direct helper use.

Symlink escape is rejected by comparing the target realpath against the
configured root realpath. Byte limits are checked before and after `Bun.file`
reads. Abort signals are checked before resolution, before the read, and after
the read.

The Qwen adapter still owns only protocol-neutral media loading and cache keys.
Model-family preprocessing remains in `@mlxts/transformers`.

## Tensor Lifetime Audit

The changed production code allocates host `Uint8Array` image bytes only. It
does not allocate, retain, or dispose `MxArray` values. Existing host decode,
resize, Qwen preprocessing, and prepared-prompt tensor ownership are unchanged.

## Memory / Performance Evidence

- `bun test packages/serve/src/media/local-image.test.ts packages/serve/src/media/image.test.ts packages/serve/src/engine/content.test.ts packages/serve/src/cli.test.ts packages/serve/src/model-loading/server.test.ts packages/serve/src/model-loading/sources.test.ts`: passed, `77` tests / `385` assertions.
- `bun run check:file-lines`: passed.
- `bun run check:runtime-review`: passed.
- `bun run check:coverage`: passed, including `@mlxts/serve` at `95.11%`
  lines and `95.91%` functions.
- `bun run validate`: passed.

## Independent Review

Dalton reviewed the roadmap and recommended the next serving capability tranche
be active-request memory pressure handling. This media tranche stayed bounded to
the already-open Phase 10 local image transport gap and leaves the memory
pressure follow-up as the next serving item.

## Out-of-scope Drift Noticed

- Active-request memory pressure shedding remains the next Phase 9 serving
  capability gap.
- `/v1/files` remains unsupported; image `file_id` values are local transport
  handles, not uploaded file objects.

## Remaining Risks / Follow-ups

Local file IDs intentionally use root-relative path handles rather than uploaded
file objects. A future `/v1/files` implementation must define upload ownership,
retention, cleanup, and metadata separately instead of reusing this local-root
transport policy as a stateful file store.
