# Contributing

mlxts is package-first. The public `@mlxts/*` workspaces are the canonical
product surface, and `packages/nanogpt` is a private validation fixture rather
than the long-term app surface.

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
- prefer improving `@mlxts/*` packages over deepening temporary fixture code
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
