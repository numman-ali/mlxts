## Summary

Migrated the legacy nanoGPT validation surface from `packages/nanogpt` to
`examples/nanogpt`, rewired the repo to treat it as a committed in-repo example
instead of a package, and removed root-level example command ownership.

The migration also exposed a real operator bug: the outer `acceptance` and
`soak` wrappers were holding the shared runtime lock while launching the
supervised run, which caused the detached supervisor to die immediately with a
`dead-supervisor` health state. That orchestration bug is now fixed by letting
the supervised run own the lock and reacquiring it only for post-run local MLX
work.

## Files Reviewed

- examples/nanogpt/src/bench/memory.ts
- examples/nanogpt/src/checkpoint.ts
- examples/nanogpt/src/cli.ts
- examples/nanogpt/src/cli/commands.ts
- examples/nanogpt/src/cli/help.ts
- examples/nanogpt/src/cli/session.ts
- examples/nanogpt/src/cli/shared.ts
- examples/nanogpt/src/cli/train-events.ts
- examples/nanogpt/src/config.ts
- examples/nanogpt/src/generate.ts
- examples/nanogpt/src/index.ts
- examples/nanogpt/src/model/causal-self-attention.ts
- examples/nanogpt/src/model/gpt.ts
- examples/nanogpt/src/model/init.ts
- examples/nanogpt/src/model/mlp.ts
- examples/nanogpt/src/model/transformer-block.ts
- examples/nanogpt/src/optimizer-defaults.ts
- examples/nanogpt/src/run/acceptance-options.ts
- examples/nanogpt/src/run/acceptance-runtime.ts
- examples/nanogpt/src/run/acceptance.ts
- examples/nanogpt/src/run/files-health.ts
- examples/nanogpt/src/run/files-json.ts
- examples/nanogpt/src/run/files-paths.ts
- examples/nanogpt/src/run/files-types.ts
- examples/nanogpt/src/run/files.ts
- examples/nanogpt/src/run/manager-args.ts
- examples/nanogpt/src/run/manager-run.ts
- examples/nanogpt/src/run/manager-status.ts
- examples/nanogpt/src/run/manager.ts
- examples/nanogpt/src/run/soak.ts
- examples/nanogpt/src/run/supervisor-events.ts
- examples/nanogpt/src/run/supervisor-streams.ts
- examples/nanogpt/src/run/supervisor.ts
- examples/nanogpt/src/safetensors.ts
- examples/nanogpt/src/train.ts

## Tensor Lifetime Audit

`bun run check:tensor-lifetimes` passes after the move, so the example now
participates in the same path-based anonymous-intermediate audit under
`examples/nanogpt/src/` that it previously had under the package path.

Spot-checked the runtime-sensitive operator flow changes in
`run/acceptance.ts`, `run/soak.ts`, `run/manager-run.ts`, and
`run/supervisor.ts` to confirm the migration did not hide tensor-producing work
inside nested expressions. The only behavioral fix in this pass was lock
ownership around orchestrator wrappers, not tensor math changes.

## Memory / Performance Evidence

- `bun run typecheck`
- `bun run check:tensor-lifetimes`
- `bun run check:coverage`
- `bun test examples/nanogpt/src/run/acceptance.test.ts`
- `cd examples/nanogpt && bun test`

The end-to-end example evidence is especially important here because the move
surfaced the lock-ownership regression. After the fix:

- the full nanoGPT example suite passed
- the acceptance wrapper completed a tiny supervised run and emitted a sample
- the soak wrapper completed and reported stable throughput / memory slope

## Independent Review

Parallel sidecar audits informed the migration:

- Kepler verified the repo checks now treat `examples/nanogpt/src/` as the
  runtime-sensitive path for tensor-lifetime and runtime-review enforcement.
- Boole audited the example-local command surface and path assumptions so the
  example no longer points users back to root-owned scripts.

No separate human review has happened yet.

## Remaining Risks / Follow-ups

- Git currently sees the move as deleting `packages/nanogpt` and adding
  `examples/nanogpt`, so this artifact has to review the full example runtime
  surface rather than a smaller semantic diff. That should shrink naturally once
  the move is committed.
- The example still uses the workspace package name `nanogpt`. That is fine for
  now because it is private, but we can rename it later if we want the workspace
  identity to reinforce the example boundary more strongly.
