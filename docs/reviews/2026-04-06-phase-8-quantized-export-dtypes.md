# Runtime Review: Quantized Export Auxiliary Dtypes

## Summary

This follow-up review covers two runtime-sensitive fixes that unblocked real
checkpoint proof work after the earlier streaming exporter landed:

1. the quantized snapshot export path no longer hard-codes auxiliary tensor
   dtypes, and
2. the remote snapshot resolver no longer downloads heavyweight checkpoint
   formats that the current loader does not use.

The export crash was not caused by the streaming file-copy path itself. The
root cause was a wrong dtype assumption inside
`quantizePretrainedSnapshot()`: the export path hard-coded quantized `.scales`
and `.biases` tensors as `float32`, while MLX returns those auxiliary tensors in
the source floating dtype for `affine` quantization and in `uint8` for the
MXFP/NVFP modes.

That mismatch meant the safetensors bridge asked MLX for the wrong raw pointer
type when exporting real model weights. On bfloat16 checkpoints, the write path
would bridge `bfloat16` auxiliary tensors as if they were `float32`, which
caused the Bun segfaults seen during the Llama and Gemma export canaries.

The export fix makes the metadata and byte bridge respect the actual quantized
component dtypes that MLX produces, and it exposes the canonical
`toSupportedSafetensorsDType()` helper from `@mlxts/core` so the exporter can
reuse the same safetensors dtype mapping instead of duplicating it.

The resolver fix narrows remote downloads to the supported artifact set:
standard sidecars plus `.safetensors` weights. That matters for multi-format
repos like the official Meta Llama checkpoints, which may contain both
`model.safetensors` and `original/*.pth`. The loader should not waste time or
disk pulling unsupported originals when the supported safetensors file is
already available.

## Files Reviewed

- `packages/core/src/index.ts`
- `packages/core/src/io.ts`
- `packages/core/src/io-safetensors.ts`
- `packages/transformers/src/pretrained/snapshot.ts`
- `packages/transformers/src/pretrained/snapshot-supported-files.ts`
- `packages/transformers/src/quantize.ts`

## Tensor Lifetime Audit

- The export path still keeps one quantized tensor's output bytes live at a
  time inside `quantizedTensorEntries()`, and the cached byte record is dropped
  after the last descriptor for that source tensor is written.
- The new dtype fix does not add any hidden nested tensor-producing calls. It
  only changes which safetensors dtype tag and byte bridge are used for the
  already-materialized quantized auxiliaries.
- The resolver narrowing is host-side file selection only. It changes which
  files are downloaded into the cache, not tensor ownership once a supported
  snapshot is on disk.
- The one new exported helper, `toSupportedSafetensorsDType()`, is a pure dtype
  mapping function and does not affect native ownership.

## Memory / Performance Evidence

- Before the dtype fix, a direct repro on
  `model.layers.0.mlp.down_proj.weight` from the cached
  `mlx-community/Llama-3.2-1B-Instruct-bf16` checkpoint crashed immediately
  after bridging the quantized weight and scales, right when the exporter tried
  to bridge the bfloat16 bias tensor as `float32`.
- After the fix, the same direct repro completes and reports:
  `weight bytes=8388608`, `scale bytes=524288`, `bias bytes=524288`.
- After the fix, the full official export canary for
  `mlx-community/Llama-3.2-1B-Instruct-bf16` completes successfully with:
  `input_bytes=2480814925`, `output_bytes=1082053444`,
  `quantized_tensors=112`, `copied_tensors=34`, and an output/input ratio of
  `0.436x`.
- After the resolver narrowing, the official
  `meta-llama/Llama-3.2-1B-Instruct` canary downloads only the supported
  safetensors/tokenizer artifact set, quantizes successfully with
  `input_bytes=2480796386`, `output_bytes=1082020763`,
  `quantized_tensors=112`, `copied_tensors=34`, and an output/input ratio of
  `0.436x`.
- The quantized official Meta snapshot reloads through the normal pretrained
  path and answers coherently in `examples/chat`. A greedy `Hello there` canary
  produced: `Hello! It's nice to meet you. Is there something I can help you
  with or would you like to chat?`

## Independent Review

- Independent review is still pending for this follow-up.
- The root cause was narrowed with a direct one-tensor repro against a cached
  real checkpoint rather than inferred from the large-model crash alone.

## Remaining Risks / Follow-ups

- The exporter now respects MLX's current auxiliary dtypes, but the repo should
  keep a direct unit test around the quantization dtype contract so this does
  not regress when MLX or mlx-c changes behavior.
- The resolver now prefers supported files, but repos with no safetensors path
  at all still remain out of scope for the current loader. Supporting original
  checkpoint formats would be separate work.
- The official Gemma end-to-end quantize → reload → chat proof remains an
  important acceptance check for this phase now that the official Meta Llama
  path is green.
