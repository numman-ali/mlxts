# Model Load Memory Preflight

## Summary

Source-backed `@mlxts/serve` model loading now performs a best-effort memory
preflight after snapshot resolution and before MLX weights load. The estimate
uses local safetensor bytes plus serving headroom and compares projected load
memory against the existing `gpuMemoryUtilization` budget. Missing MLX telemetry
or missing safetensor sizing skips the preflight rather than making an unsafe
claim.

## Files Reviewed

- `packages/serve/src/model-loading/memory-preflight.ts`
- `packages/serve/src/model-loading/server.ts`
- `packages/serve/src/model-loading/sources.ts`
- `packages/serve/src/model-loading/memory-preflight.test.ts`
- `packages/serve/src/model-loading/server.test.ts`
- `packages/serve/src/model-loading/sources.test.ts`

## Runtime Sensitivity

The changed production path runs before model load, not during generation. It
does not touch tensor lifetimes, generation loops, cache mutation, protocol
formatting, or streaming cadence. It reads MLX allocator telemetry through the
existing best-effort serving telemetry helper and scans host-side snapshot
metadata only.

## Tensor Lifetime Audit

No tensor-producing calls changed. The new preflight reads host-side file
metadata and optional serving memory telemetry before model construction, so it
does not add hidden `MxArray` intermediates or alter disposal responsibility.

## Memory / Performance Evidence

The preflight rejects only when all required facts are present:

- MLX memory telemetry is readable.
- The resolved local source has `.safetensors` files.
- Hugging Face snapshot symlinks resolve to local safetensor bytes.
- `activeBytes + ceil(safetensorBytes * 1.25)` exceeds
  `limitBytes * gpuMemoryUtilization`.

When those facts are absent, source-backed loading proceeds unchanged.

No generation hot path changed. The added work is one host-side directory scan
per source-backed model before loading. Existing already-loaded serving,
request routing, scheduler operation, prefix-cache restore, media preparation,
and SSE streaming paths are unchanged.

## Tests

- `bun test packages/serve/src/model-loading/memory-preflight.test.ts packages/serve/src/model-loading/server.test.ts packages/serve/src/model-loading/sources.test.ts`
- `bun test packages/serve`
- `bun run typecheck`
- `bun run lint`
- `bun run check:tensor-lifetimes`

## Independent Review

Bohr reviewed the working tree diff and found that Hugging Face snapshot
symlinks were skipped by the first scanner shape. The scanner now stats
`.safetensors` paths directly, and the test suite covers a symlinked snapshot
entry before final validation.

## Out-of-scope Drift Noticed

The full Phase 9d model pool still needs lazy loading, LRU eviction, pinning,
TTL, and active-request abort policy. This tranche lands only the pre-load
memory estimate needed before those controls can be made operator-facing.

## Remaining Risks / Follow-ups

Safetensor file size is an estimate, not exact resident MLX allocation size.
The 25% headroom follows the documented serving roadmap but remains a
bookkeeping guardrail rather than a replacement for active memory telemetry
during generation.
