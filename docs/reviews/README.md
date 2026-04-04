# Runtime Review Artifacts

This folder holds review records for runtime-sensitive changes.

Use `docs/reviews/_template.md` when a diff touches production code in:

- `packages/core/src/`
- `packages/nn/src/`
- `packages/optimizers/src/`
- `packages/train/src/`
- `packages/data/src/`
- `packages/tokenizers/src/`
- `packages/nanogpt/src/` (temporary validation fixture)

The artifact is part of the deliverable. `bun run check:runtime-review` expects at least one changed review file with the required sections whenever those runtime-sensitive files change.
