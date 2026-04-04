# Runtime Review Artifacts

This folder holds review records for runtime-sensitive changes.

Use `docs/reviews/_template.md` when a diff touches production code in:

- `packages/mlx-ts/src/core/`
- `packages/mlx-ts/src/nn/`
- `packages/mlx-ts/src/optimizers/`
- `packages/nanogpt/src/`

The artifact is part of the deliverable. `bun run check:runtime-review` expects at least one changed review file with the required sections whenever those runtime-sensitive files change.
