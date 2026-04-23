# Runtime Review: Serve CLI Multi-Model Entries

## Summary

Added repeatable `--model <source>` / `--model <model-id=source>` support to
the `mlxts-serve` CLI. The old positional command remains the single-model
shorthand, including `--model-id` / `--served-model-name`.

The CLI now routes multiple `--model` entries into the existing `serveModels()`
programmatic loader instead of inventing a separate loading path. It also splits
CLI parsing into a small options module so the CLI runner stays below the repo
line-limit and remains focused on operator I/O.

## Files Reviewed

- `packages/serve/src/cli.ts`
- `packages/serve/src/cli-options.ts`

## Tensor Lifetime Audit

This change does not allocate tensors directly. Multi-model CLI startup
delegates model ownership and cleanup to `serveModels()`, which was reviewed in
`docs/reviews/2026-04-23-serve-model-sources.md`.

The CLI runner only adapts parsed options into `serveModel()` or `serveModels()`
calls and formats progress/events. No generation hot path, cache behavior, or
tensor ownership rules changed.

## Memory / Performance Evidence

Validated with:

- `bun test packages/serve/src/cli.test.ts`
- `bun run typecheck`
- `bun run lint`

The tests prove:

- legacy positional serving still parses with `--model-id`
- repeatable `--model` entries preserve order and call `serveModels()`
- duplicate model ids, empty specs, positional plus `--model`, and `--model-id`
  plus `--model` fail clearly before loading
- ready output lists all served model ids and keeps the curl example pointed at
  the first configured model
- multi-model progress logs include model index and model id

No generation benchmark is required because this change only affects CLI
parsing/startup orchestration.

## Independent Review

Hume independently audited the CLI shape and recommended repeatable
`--model <source-or-id=source>` while keeping the positional form as the
single-model shorthand. The audit also recommended rejecting ambiguous
`--model-id` plus `--model` usage and routing the CLI through `serveModels()`
rather than adding duplicate loading behavior.

## Remaining Risks / Follow-ups

All configured models load into one process. Sequential loading reduces startup
spikes, but operators still need to choose model sizes and quantization levels
that fit local unified memory.

Per-model revisions, cache directories, and auth options are intentionally not
part of this CLI tranche. If that becomes important, prefer a config-file
surface over making the flag grammar clever.
