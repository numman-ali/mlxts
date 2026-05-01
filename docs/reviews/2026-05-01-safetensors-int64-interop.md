# Runtime Review: Safetensors 64-bit integer interop

## Summary

The safetensors bridge now accepts valid `I64` and `U64` tensor headers and can save/load 64-bit integer tensors. This unblocks checkpoints that carry scalar bookkeeping tensors, such as FLUX.2 Klein's VAE `bn.num_batches_tracked`, without weakening header validation.

## Files Reviewed

- `packages/core/src/io-safetensors-format.ts`
- `packages/core/src/array-ffi-data.ts`
- `packages/core/src/ffi/symbols.ts`
- `packages/core/src/io-extra.test.ts`

## Tensor Lifetime Audit

The loader continues to copy safetensor byte ranges into typed views before creating owned MLX arrays through the existing `mlx_array_new_data` path. The saver continues to force contiguity, evaluate once, copy native bytes into JS-owned storage, and dispose the temporary contiguous tensor with `using`.

No new retained native state, transform cache, or handle lifetime path was introduced. The added FFI symbol is a const data pointer getter for existing array storage, matching the already-used signed 64-bit getter.

## Memory / Performance Evidence

- `bun test packages/core/src/io-extra.test.ts`
- `bunx tsc -p packages/core/tsconfig.json --pretty false`

The change only extends dtype tag mapping and direct byte extraction for scalar/metadata-style integer tensors. It does not change dense tensor streaming, model execution, or checkpoint shard iteration order.

## Independent Review

Gibbs the 2nd performed a read-only second-opinion review of the core interop
change. The review found no blocking correctness, lifetime, or ABI issue and
recommended adding an incident-shaped `I64` scalar fixture with exact byte
preservation, which is now covered in `io-extra.test.ts`.

## Remaining Risks / Follow-ups

FLUX.2 Klein real checkpoint evidence is recorded separately in
`docs/reviews/2026-05-01-flux2-klein-real-checkpoint-proof.md`.
