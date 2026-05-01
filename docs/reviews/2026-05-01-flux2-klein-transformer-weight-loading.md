# FLUX.2 Klein Transformer Weight Loading

## Summary

Added Diffusers safetensors loading for the FLUX.2 Klein transformer component.
The loader maps expanded Diffusers `Flux2Transformer2DModel` parameter names onto
the package-owned `Flux2KleinTransformer2DModel` tree, supports sharded
`.safetensors.index.json` manifests, and constructs the transformer directly
from inspected FLUX.2 Klein snapshot manifests.

This tranche is weight-loading only. It does not change transformer forward
execution, prompt conditioning, reference-image KV cache behavior, sampling, or
VAE loading.

## Files Reviewed

- `packages/diffusion/src/families/flux2/transformer-weights.ts`
- `packages/diffusion/src/families/flux2/weights.ts`
- `packages/diffusion/src/index.ts`
- `packages/diffusion/src/families/flux2/transformer-weights.test.ts`

## Tensor Lifetime Audit

Safetensors tensors are consumed one tensor at a time through
`iterateSafetensors`. Assigned tensors replace existing module parameters after
shape validation. Unexpected, mismatched, and error-path tensors are freed by
the assignment helpers before the error leaves the loader.

Snapshot construction disposes the partially constructed transformer when
configuration or weight loading fails.

## Memory / Performance Evidence

The loader preserves shard-iterator-first loading and does not eagerly
materialize complete shards. It performs direct tensor assignment only; FLUX.2
Klein does not need the FLUX.1 fused-QKV merge or final modulation half swap for
Diffusers-format safetensors.

Focused evidence:

```bash
bun test packages/diffusion/src/families/flux2/transformer-weights.test.ts
bun run typecheck
bun run check:file-lines
```

## Independent Review

Russell the 2nd reviewed the FLUX.2 reference implementation and current
package runtime. The review confirmed that Diffusers FLUX.2 transformer weights
map directly to the package tree, that bias tensors are not expected, that no
transpose/squeeze/split/merge transform is needed, and that FLUX.1's fused
loader behavior must not be copied into FLUX.2 Klein.

## Remaining Risks / Follow-ups

- Official checkpoint proof still needs Qwen3 prompt conditioning wired into an
  example/proof command and then a bounded real FLUX.2 Klein generation run.
- Diffusers original-converter or non-expanded checkpoint formats remain out of
  scope for this tranche.
- Reference-image KV cache semantics remain a separate runtime tranche.

## Out-of-scope Drift Noticed

`.reference/transformers` still has an existing unresolved merge state and was
not refreshed during this tranche.
