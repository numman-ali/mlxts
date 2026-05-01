# Runtime Review: Phase 10 Image Proof QA

## Summary

Phase 10 image proof commands now emit machine-checkable artifact evidence for
their BMP outputs. The shared example verifier records BMP geometry, byte
length, SHA-256, tensor range, and non-uniform pixel evidence, and
`examples/image-proof/verify-report.ts` verifies saved JSON reports against the
referenced BMP file without rerunning generation.

This tranche does not change diffusion sampling, model loading, scheduler
semantics, conditioning, or generated tensor values. It tightens the proof
surface around already-generated artifacts.

## Files Reviewed

- `examples/image-proof/artifact.ts`
- `examples/image-proof/verify-report.ts`
- `examples/stable-diffusion/image-output.ts`
- `examples/stable-diffusion/index.ts`
- `examples/flux/image-output.ts`
- `examples/flux/index.ts`
- `examples/z-image/image-output.ts`
- `examples/z-image/index.ts`
- `examples/qwen-image/image-output.ts`
- `examples/qwen-image/index.ts`
- `package.json`
- `tsconfig.phase10-examples.json`

## Tensor Lifetime Audit

The changed image-output helpers call `image.eval()` and `image.toTypedArray()`
at the same host artifact boundary that already wrote BMP bytes. The shared
helper computes stats during the existing pixel traversal and does not retain
MLX arrays or tensor handles beyond the caller-owned generated image scope.

The verifier CLI is host-only: it reads JSON and BMP files from disk, computes
header and payload checks, and never calls MLX tensor operations.

## Memory / Performance Evidence

The only added work on proof-command success is host-side SHA-256 over the BMP
bytes plus O(width * height) artifact checks folded into BMP writing. This is
outside model generation and does not add model forwards, denoising steps, or
extra tensor allocations on the MLX runtime path.

`bun run check:phase10-proofs` passed with 72 focused Phase 10 proof tests. No
throughput or image-quality claim is made.

## Validation

```bash
bun test examples/image-proof examples/stable-diffusion examples/flux examples/z-image examples/qwen-image
bun run check:phase10-proofs
bun run typecheck
bun run lint
bun run check:assertions
bun run check:file-lines
bun run check:skills
bun run check:per-package-agents
bun run check:cross-package-imports
bun run check:tensor-lifetimes
bun run check:runtime-review
bun run check:coverage
```

All commands passed.

## Independent Review

Linnaeus reviewed the intended image-proof QA tranche independently before the
implementation was finalized. The review recommended keeping the helper under
`examples/image-proof/`, checking report schema plus BMP header/byte evidence,
preserving AXI stdout/stderr behavior, and avoiding visual-quality claims.

## Out-of-Scope Drift Noticed

- Later Phase 10 families such as FLUX.2 Klein, SD3/3.5, video, and audio still
  need separate reference audits and runtime proofs before support claims.
- Existing real checkpoint proof artifacts need to be regenerated with the new
  JSON artifact evidence if they are to be verified by this CLI.

## Remaining Risks / Follow-ups

The verifier proves artifact integrity and non-uniform image bytes. It does not
judge semantic image quality, prompt adherence, or visual aesthetics.
