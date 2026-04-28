# Runtime Review: Serve CacheLayerKind Routing

## Summary

Serve routing now reads cache-shape decisions from `TransformerCache.layerKinds`
instead of Qwen family identifiers or raw attention-type config strings. The
route probe creates one disposable cache per model, copies the semantic layer
kinds, and memoizes that copy for later routing decisions.

## Files Reviewed

- `packages/serve/src/engine/routing.ts`

## Tensor Lifetime Audit

`routing.ts` creates the probe cache with `using cache = model.createCache()` and
copies only the string-valued `layerKinds` into a `WeakMap`. No `MxArray`
handles cross the helper boundary. `packages/serve/src/engine/routing.test.ts`
asserts the probe cache is created once per model and disposed after the copy.

## Memory / Performance Evidence

- `bun test packages/serve`: 288 pass, 0 fail.
- `bun run typecheck`: all workspaces passed.
- `bun run lint`: 632 files checked, no warnings.
- `bun run validate`: passed.
- `bun run regression:qwen-gemma -- --profile real`: passed.
- Qwen decode smoke: 29.004 generation tok/s, 1.00 eval/token, peak memory 17.184 GB.
- Gemma decode smoke: 81.951 generation tok/s, 1.00 eval/token, peak memory 9.893 GB.
- Qwen serve continuous sweeps passed through 8 concurrent streaming rows with
  `routes=continuous:eligible`.
- Gemma serve continuous sweeps passed through 8 concurrent streaming rows with
  `routes=continuous:eligible`.
- Mixed long/short fairness smokes passed: Qwen 32768x128+128x32 and Gemma
  5000x128+128x32 both reported `routes=continuous:eligible=2`.

## Independent Review

Schrodinger reviewed the tranche before implementation and recommended probing
`cache.layerKinds` once, memoizing by model, requiring model-owned batch cache
support for `linear-recurrent` routes, and retaining model-type guards only for
known Gemma layer-pattern batch support.

## Remaining Risks / Follow-ups

Gemma layer-pattern eligibility still carries known implementation guards by
Gemma model type. A future non-Gemma full/sliding layer-pattern backend needs a
transformers-owned capability marker before serve can route it purely from
`CacheLayerKind`.

## Out-of-scope drift noticed

None.
