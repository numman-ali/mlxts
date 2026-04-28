# nanoGPT Example

Committed GPT training and regression surface for `mlxts`.

This directory is intentionally an in-repo example, not a publishable package.
It wires a small GPT model onto the reusable `@mlxts/*` packages and keeps the
GPT-specific supervised run policy: presets, acceptance thresholds, sample
generation, checkpoint naming, and `.nanogpt-runs/` operator state.

## Common Commands

Run commands from this directory:

```bash
bun test
bun run typecheck
bun run build
```

The build writes ignored `dist/` output for local editor and release-readiness
checks. `dist/` is not source and is not committed.

## Training And Generation

```bash
bun run src/cli.ts train --preset gpt-tiny
bun run src/cli.ts generate --checkpoint <checkpoint-dir> --prompt "To be"
```

Snapshot checkpoints are lightweight model saves. Resume checkpoints include
optimizer state for exact continuation.

## Supervised Runs

The run manager is the canonical long-run surface:

```bash
bun run manager start --preset gpt-small --max-steps 5000
bun run manager status --name <run-id>
bun run manager watch --name <run-id> --interval 600
bun run manager stop --name <run-id>
bun run manager resume --from <run-id> --max-steps 10000
```

The manager writes run-local status, control, event, stderr, pid, and checkpoint
files under `.nanogpt-runs/`.

## Soak And Acceptance

```bash
bun run bench:memory
bun run soak:gpt-tiny
bun run soak:gpt-small
bun run acceptance:gpt-tiny
bun run acceptance:gpt-small
```

These commands are heavy MLX runtime checks. Run them one at a time.
