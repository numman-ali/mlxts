# Serve Pressure Timeout CLI

## Summary

The lazy model-pool pressure release timeout is now an operator-facing package
and CLI setting. `serveModels()` accepts `modelPressureReleaseTimeoutMs`, and
`mlxts-serve --model-pressure-release-timeout-ms <n>` forwards it to the lazy
source pool. The default remains unchanged when the option is omitted.

## Files Reviewed

- `packages/serve/src/cli-options.ts`
- `packages/serve/src/cli.ts`
- `packages/serve/src/cli-usage.ts`
- `packages/serve/src/model-loading/sources.ts`
- `packages/serve/src/model-loading/source-pool-server.ts`

## Tensor Lifetime Audit

This tranche changes option parsing, option validation, documentation, and lazy
pool wiring only. It does not add tensor-producing operations, alter model
forward paths, or change `MxArray` ownership.

## Memory / Performance Evidence

Default serving behavior is unchanged: omitted timeout values still use the
pool default, and `modelPressurePolicy: "reject"` remains the CLI and package
default. The new knob only bounds the already explicit `shed_non_pinned` lazy
pressure path.

Validated:

- `bun test packages/serve/src/cli.test.ts packages/serve/src/model-loading/sources.test.ts`
  (`32` tests)
- `bun run --filter '@mlxts/serve' typecheck`
- `bun run lint`
- `bun run check:file-lines`
- `bun run check:runtime-review`
- `bun test packages/serve` (`428` tests)
- `bun run validate`

## Independent Review

Lovelace performed a blocker-only review and found no code blocker. The review
confirmed the option is parsed, validated as a positive integer, rejected
outside lazy loading, forwarded through `ServeModelsOptions`, and passed into
the lazy pool as `pressureReleaseTimeoutMs`.

## Out-of-scope Drift Noticed

- Real-model `shed_non_pinned` pressure smoke coverage remains a separate
  memory-policy proof.

## Remaining Risks / Follow-ups

- `/info` does not yet expose lazy model-pool pressure policy or release
  timeout. Add that only with a broader model-pool introspection surface.
- `--model-pressure-release-timeout-ms` is accepted with lazy loading even when
  the pressure policy is the default `reject`; that pairing is currently a
  harmless no-op.
