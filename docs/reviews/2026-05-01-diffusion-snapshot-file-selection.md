# Diffusion Snapshot File Selection

## Summary

Tightened Hugging Face Diffusers snapshot resolution so remote downloads select
metadata plus exactly one component-local safetensor set per model component.
Root safetensor exports are skipped because current package loaders consume
Diffusers component folders, not single-file UI exports. The Stable Diffusion
and FLUX proof CLIs now expose `--variant <name>` for repositories that publish
both default and fp16 component weights.

## Files Reviewed

- `packages/diffusion/src/pretrained/snapshot-file-selection.ts`
- `packages/diffusion/src/pretrained/snapshot-source.ts`

## Reference Audit

Hugging Face repository layouts reviewed for `segmind/SSD-1B`,
`black-forest-labs/FLUX.1-schnell`, `Qwen/Qwen-Image`, and
`Tongyi-MAI/Z-Image`. The common Diffusers package shape is component folders
with `config.json`, tokenizer metadata, and component-local safetensors or
safetensors indexes. Root `.safetensors` files are separate single-file exports
or sidecars and are not loadable by the current manifest/loader path.

Diffusers variant behavior recognizes component-local filename variants such as
`model.fp16.safetensors`,
`diffusion_pytorch_model.fp16.safetensors.index.json`, and sharded names like
`model.fp16-00001-of-00002.safetensors`; tests cover these shapes.

## Tensor Lifetime Audit

This tranche does not create, retain, or free `MxArray` tensors. The changed
code runs before model construction and only selects host-side metadata and
weight filenames for download.

## Memory / Performance Evidence

Remote file selection now avoids downloading root monolith exports and
unselected duplicate variants. For SSD-1B-style repositories, `--variant fp16`
selects fp16 component weights and skips default fp32 siblings. The resolver
still emits bounded progress on stderr through the AXI example commands.

## Independent Review

Nash reviewed the tranche shape before validation. The review confirmed the
minimal forward-only rule: select small metadata plus exactly one
component-local weight set per component, skip root safetensors, preserve
sharded index plus shard sets, and leave semantic validation to
`loadDiffusionSnapshotManifest()`. The review also called out stale polluted Hub
caches as a remaining risk.

## Validation

- `bun test packages/diffusion/src/pretrained/snapshot-file-selection.test.ts packages/diffusion/src/pretrained/snapshot-source.test.ts`
- `bun run --filter @mlxts/diffusion typecheck`
- `bun run check:phase10-proofs`
- `bun run check:file-lines`
- `bun run validate`

## Remaining Risks / Follow-ups

- Existing Hub cache directories polluted by earlier broad downloads may still
  contain unselected component variants, because the resolver does not delete
  cached files. Fresh resolves and clean cache directories use the tightened
  selection.
- Root single-file safetensor exports remain out of scope until a dedicated
  single-file diffusion loader exists.
