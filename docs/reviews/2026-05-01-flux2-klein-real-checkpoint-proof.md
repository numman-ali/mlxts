# Runtime Review: FLUX.2 Klein real checkpoint proof

## Summary

Official `black-forest-labs/FLUX.2-klein-4B` now has bounded real checkpoint image evidence through `examples/flux2`. The proof resolved the Hugging Face Diffusers snapshot, loaded Qwen3 prompt conditioning, the FLUX.2 Klein transformer, FlowMatch scheduler, and FLUX.2 VAE, ran four 256x256 bfloat16 denoise steps, wrote a BMP artifact, and passed saved-report verification.

The run also exposed a valid safetensors `I64` scalar tensor in the VAE file. `@mlxts/core` now accepts 64-bit integer safetensors tags so skipped checkpoint bookkeeping tensors are parsed correctly before family-owned weight sanitizers decide whether to load them.

## Files Reviewed

- `packages/core/src/io-safetensors-format.ts`
- `packages/core/src/array-ffi-data.ts`
- `packages/core/src/ffi/symbols.ts`
- `examples/flux2/index.ts`
- `examples/flux2/conditioning-runtime.ts`
- `packages/diffusion/src/families/flux2/pipeline.ts`
- `packages/diffusion/src/families/flux2/autoencoder.ts`
- `packages/diffusion/src/families/flux2/weights.ts`
- `packages/diffusion/src/families/flux2/transformer.ts`
- `packages/diffusion/src/families/flux2/transformer-weights.ts`

## Checkpoint Evidence

```bash
bun run examples/flux2/index.ts black-forest-labs/FLUX.2-klein-4B \
  --prompt "a small red apple on a white table, product photo" \
  --output .tmp/flux2/flux2-klein-official-proof.bmp \
  --steps 4 \
  --height 256 \
  --width 256 \
  --guidance-scale 1 \
  --seed 7 \
  --dtype bfloat16 \
  --json > .tmp/flux2/flux2-klein-official-proof.json
```

- Resolved revision: `e7b7dc27f91deacad38e78976d1f2b499d76a294`
- Selected snapshot files: `18`
- Selected snapshot bytes: `15,980,131,745`
- Output: `.tmp/flux2/flux2-klein-official-proof.bmp`
- Output bytes: `196,662`
- Output SHA-256: `4c116a4b03e632fcf09e02b6533620125ed316e9f4365f354c0a578a45b140f4`
- Report SHA-256: `5eaf8e01a434cbd1987c394dc45b9f015e5d02ce6e98acb232d85e21e408fc93`
- Elapsed generation command time: `5,040.52 ms`

```bash
bun run examples/image-proof/verify-report.ts \
  .tmp/flux2/flux2-klein-official-proof.json \
  --expect-pipeline flux2-klein
```

Verifier result: `passed`, `18` passed checks, `0` failed checks.

## Tensor Lifetime Audit

The proof command keeps transformer, VAE, text encoder, prompt conditioning tensors, denoised image tensors, and output tensors behind explicit disposal scopes. Prompt embeddings crossing from the example into `@mlxts/diffusion` remain owned by `Flux2KleinPromptConditioningResult`.

The core safetensors fix preserves the existing byte-copy boundary before `mlx_array_new_data`. The saver still evaluates a temporary contiguous tensor and copies native storage into JS-owned bytes before writing.

## Memory / Performance Evidence

This is a bounded capability proof, not a throughput or quality benchmark. The run validates that the official 4B checkpoint can load and produce a finite, nonblank image artifact through the package-owned runtime.

Focused gates:

```bash
bun test packages/core/src/io-extra.test.ts
bunx tsc -p packages/core/tsconfig.json --pretty false
bun test packages/diffusion/src/families/flux2/weights.test.ts
bun run examples/image-proof/verify-report.ts .tmp/flux2/flux2-klein-official-proof.json --expect-pipeline flux2-klein
```

## Independent Review

Lorentz the 2nd performed a read-only second-opinion review of the proof
tranche. The review verified the saved artifact SHA, verifier result, selected
snapshot file count/bytes, and core safetensors ABI mapping, and found no
blocking issue.

## Remaining Risks / Follow-ups

- Reference-image conditioning and KV-cache variants remain separate FLUX.2 tranches.
- Larger/default-step image quality and performance characterization remains future evidence.
- The Qwen3 text-encoder path pads token ids when a pad token exists but does not yet pass a Diffusers-style attention mask into `LlamaLikeModel`.
