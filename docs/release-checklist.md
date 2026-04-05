# Release Checklist

This repo now supports local release-readiness checks for the public
`@mlxts/*` packages even though the actual npm publish step is still manual.

## Before a release

```bash
bun install
bun run build:native
bun run release:check
```

## What `release:check` covers

- repo-wide typecheck, lint, assertion, file-size, tensor-lifetime, runtime-review, and coverage gates
- `dist/` builds for all workspace packages
- TypeDoc generation into `.tmp/api-docs/`
- dry-run tarball packing for the public packages

## Packaging expectations

- public packages export only their root entrypoints in this phase
- `packages/nanogpt` is private and is not a publish target
- `@mlxts/core` still expects an explicit native build step
- Changesets are configured for the scoped public packages only

## Manual spot checks

- open one generated tarball dry-run log and confirm only `dist/`, package metadata, and intended support files are included
- install the packed tarballs into a temp project if you want an extra pre-publish check
- run the tiny tensor smoke path and, on Apple Silicon, confirm `bun run build:native` plus `@mlxts/core` still works end to end
