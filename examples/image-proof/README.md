# Image Proof Verifier

Shared verifier for Phase 10 image proof reports.

The image workbooks write uncompressed BMP artifacts and include an `artifact`
section in `--json` output. The verifier reads that JSON report, inspects the
referenced BMP file on disk, checks header geometry, byte length, SHA-256, and
non-uniform pixel evidence, then emits compact AXI-shaped output.

```bash
bun run examples/image-proof/verify-report.ts .tmp/qwen-image/proof.json \
  --expect-pipeline qwen-image
```

This verifier proves artifact integrity only. It does not rerun model
generation or make visual-quality claims.
