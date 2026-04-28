# Runtime Review: Cross-Example Corpus Extraction

## Summary

Extracted the tiny training proof corpus and UltraChat row parser from
`examples/train-proof` into `@mlxts/data` so sibling examples no longer import
from another example. The train-proof example now consumes the package-owned
helpers while retaining its proof-specific dataset loading and UltraFeedback
preference parsing.

## Files Reviewed

- `packages/data/src/index.ts`
- `packages/data/src/training-proof.ts`

## Tensor Lifetime Audit

The changed data-package files are host-side row parsing and static chat corpus
construction only. They do not allocate `MxArray` values, call `mxEval`, hold
native handles, or alter ownership of collation outputs.

## Memory / Performance Evidence

No runtime tensor hot path changed. Focused validation covered the new data
helper tests plus the train-proof, LoRA fine-tune, and chat-canary consumers.
`bun run typecheck`, `bun run lint`, and `bun run check:file-lines` passed before
the full validation rerun.

## Independent Review

No sub-agent review was used because this session did not include explicit
delegation authorization. The change is a pure extraction with mechanical
cross-example import checks and package tests as the review backstop.

## Remaining Risks / Follow-ups

None for the extraction itself. Broader data-package doctrine can be tightened
later if the repo wants a more explicit home for reusable proof fixture parsers.
