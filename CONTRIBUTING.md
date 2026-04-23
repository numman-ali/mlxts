# Contributing

mlxts is package-first. The public `@mlxts/*` workspaces are the canonical
product surface, and `examples/nanogpt` is a committed example/regression
surface rather than a publishable package.

## Local workflow

```bash
bun install
bun run build:native
bun run validate
```

Use these additional checks when you touch package ergonomics:

```bash
bun run build
bun run docs:api
bun run pack:dry-run
```

## Expectations

- keep production files at or under 500 physical lines
- keep runtime-sensitive tensor lifetimes readable by eye
- update runtime review artifacts when hot-path production files change
- prefer improving `@mlxts/*` packages over deepening example-only code
- delete stale surfaces rather than preserving compatibility layers we no longer want

## Native runtime note

`@mlxts/core` still relies on a local MLX native build. That is intentional for
now. The repo is responsible for building and validating the native layer; live
package publishing or binary distribution is a separate step.

## CI note

The Apple Silicon validation workflow expects self-hosted runner labels:

- `self-hosted`
- `macOS`
- `ARM64`

Fast repo checks run separately and do not replace native validation.

The heavier Phase 8 training proof is intentionally separate for now. Use the
manual `Training Proof` GitHub workflow on a self-hosted Apple Silicon runner,
or run `bun run examples/train-proof/index.ts` locally when you want the
canonical LoRA / QLoRA / SFT / DPO proof without making every push pay that
cost.
