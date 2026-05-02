# LTX-Video Proof Verifier Review

## Summary

This tranche adds an LTX-owned offline verifier for saved `examples/ltx-video`
JSON reports. The verifier checks classic LTX BMP preview evidence and LTX-2 BMP
plus PCM16 WAV evidence without rerunning generation. It keeps the shared
`examples/image-proof` verifier image-only and adds the audio-video proof logic
at the workbook boundary that emits those reports.

## Files Reviewed

- `examples/ltx-video/verify-report.ts`
- `examples/ltx-video/verify-report.test.ts`
- `examples/ltx-video/README.md`
- `packages/diffusion/README.md`
- `docs/gates-and-milestones.md`
- `PLAN.md`
- `continuity.md`
- `MEMORY.md`

## Evidence

- `bun test examples/ltx-video/verify-report.test.ts`

The focused test covers AXI help, parser expectations, classic LTX verification,
LTX-2 BMP plus WAV verification, CLI execution, usage errors, expectation
mismatches, truncated WAV failures, and schema failures.

## Runtime Scope

No production tensor runtime changes are made. The verifier reads saved JSON,
BMP, and WAV files from disk and computes structural/header/hash checks. It does
not load MLX model weights, run denoising, decode media tensors, or score visual
or audio quality.

## Independent Review

Kuhn the 2nd recommended this as the next low-risk product tranche after LTX-2
proof assembly: make the newly emitted BMP/WAV evidence machine-checkable before
attempting a heavy real LTX-2 checkpoint run. That review also noted that the
visible local LTX-2.3 and FLUX.2 KV caches are incomplete, so a verifier/docs
tranche is a cleaner next commit than forcing a new model branch prematurely.

## Remaining Risks

Real LTX-2 checkpoint evidence still needs an operator run with a complete local
snapshot and the shared runtime lock. The verifier proves saved artifact
integrity, not media quality, throughput, or LTX-2.3-specific runtime behavior.
