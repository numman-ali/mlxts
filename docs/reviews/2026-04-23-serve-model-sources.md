# Runtime Review: Serve Model Sources

## Summary

Added `serveModels()` and `serveModelsWithRuntime()` as the first-class
programmatic API for loading multiple local directories or Hugging Face model
repositories and exposing them through one OpenAI-compatible endpoint.

This is a thin owned-loading wrapper over `serveLoadedModels()`: it validates all
source entries first, loads models sequentially to avoid avoidable RAM spikes,
adds source/model context to load progress callbacks, and then hands the loaded
entries to the existing multi-model serving path. It does not add new routing,
batching, or generation behavior.

## Files Reviewed

- `packages/serve/src/model-sources.ts`
- `packages/serve/src/index.ts`

## Tensor Lifetime Audit

`packages/serve/src/model-sources.ts` does not allocate tensors directly. It
owns model object lifetime while loading and delegates all generation tensor work
to the existing `serveLoadedModels()` path.

If a tokenizer/profile load fails after a model is loaded, that model is
disposed before the original error is rethrown. If a later source fails after
earlier models loaded, all earlier loaded models are disposed. If final server
startup fails, all loaded models are disposed. Cleanup failures are surfaced as
an `AggregateError` so the original failure is not silently lost.

## Memory / Performance Evidence

Validated with:

- `bun test packages/serve/src/model-sources.test.ts`
- `bun run typecheck`
- `bun run lint`

The tests prove:

- multiple sources load sequentially and are served as owned models
- global load options can be overridden per source
- progress callbacks include source index, source id, and model id
- empty source lists, blank source values, and duplicate model ids fail before
  runtime loading begins
- already-loaded models are disposed when a later tokenizer load fails
- all loaded models are disposed when final `serveLoadedModels()` startup fails

No generation benchmark is required because this is an operator-facing loading
and composition layer, not a transformer generation hot-path change.

## Independent Review

Jason independently audited this slice and confirmed the correct boundary:
`serveModels()` should stay a thin sequential owned-loading wrapper over
`serveLoadedModels()`, with no new routing or batching behavior. The review also
called out the important edge cases around validation-before-loading and cleanup
when later loads or server startup fail; the implementation and tests cover
those paths.

## Remaining Risks / Follow-ups

Serving multiple large checkpoints in one process can exceed local Apple Silicon
memory even when loading is sequential. This API makes the ergonomics correct,
but operators still need to choose model sizes and quantization levels that fit
the machine.

The CLI still serves one source at a time. A future CLI tranche can add a config
file or repeated model flags once the desired operator UX is clear.
