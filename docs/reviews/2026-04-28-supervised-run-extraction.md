# Supervised Run Extraction Review

## Summary

Moved the generic supervised-run file, manager, status, event, and supervisor
plumbing into `@mlxts/train/supervised-run`. nanoGPT now keeps thin wrappers for
its GPT-specific trainer command, `.nanogpt-runs` root, runtime lock, CLI usage,
status wording, and typed GPT config parsing.

## Files Reviewed

- `examples/nanogpt/src/run/files-health.ts`
- `examples/nanogpt/src/run/files-json.ts`
- `examples/nanogpt/src/run/files-paths.ts`
- `examples/nanogpt/src/run/files-types.ts`
- `examples/nanogpt/src/run/files.ts`
- `examples/nanogpt/src/run/manager-args.ts`
- `examples/nanogpt/src/run/manager-run.ts`
- `examples/nanogpt/src/run/manager-status.ts`
- `examples/nanogpt/src/run/manager.ts`
- `examples/nanogpt/src/run/supervised-run-config.ts`
- `examples/nanogpt/src/run/supervisor-events.ts`
- `examples/nanogpt/src/run/supervisor-streams.ts`
- `examples/nanogpt/src/run/supervisor.ts`
- `packages/train/src/supervised-run/files-health.ts`
- `packages/train/src/supervised-run/files-json.ts`
- `packages/train/src/supervised-run/files-paths.ts`
- `packages/train/src/supervised-run/files-types.ts`
- `packages/train/src/supervised-run/files.ts`
- `packages/train/src/supervised-run/index.ts`
- `packages/train/src/supervised-run/manager-args.ts`
- `packages/train/src/supervised-run/manager-run.ts`
- `packages/train/src/supervised-run/manager-status.ts`
- `packages/train/src/supervised-run/manager.ts`
- `packages/train/src/supervised-run/supervisor-events.ts`
- `packages/train/src/supervised-run/supervisor-streams.ts`
- `packages/train/src/supervised-run/supervisor.ts`

## Tensor Lifetime Audit

No tensor-producing expressions, optimizer update logic, checkpoint tensor
serialization, or model execution code changed. The runtime-sensitive surface is
process supervision: atomic status/control files, JSONL event pumping,
heartbeats, stall detection, stop/cancel escalation, and detached trainer launch.

## Memory / Performance Evidence

This tranche makes no performance claim and does not alter training math,
checkpoint tensor payloads, model forwards, or optimizer steps.

Focused validation passed:

- `bun run --filter '@mlxts/train' typecheck`
- `bun run --filter nanogpt typecheck`
- `bun run --filter '@mlxts/train' build`
- `cd examples/nanogpt && bun run build`
- `bun test packages/train/src/supervised-run/supervised-run.test.ts`
- `cd examples/nanogpt && bun test src/run/files.test.ts src/run/manager.test.ts`
- `bun run --filter '@mlxts/train' test`
- `bun run --filter nanogpt test`
- `bun run lint`
- `bun run validate`

The final coverage gate reported `@mlxts/train` at 95.55% line coverage and
98.48% function coverage after the supervised-run package tests were added.

## Independent Review

Jason independently reviewed the nanoGPT run surface and recommended the same
boundary: package-owned run files, manager CLI, status payload, event state
machine, and supervisor process loop; nanoGPT-owned presets, GPT config typing,
acceptance thresholds, sample generation, `.nanogpt-runs`, and trainer command.

## Out-of-scope Drift Noticed

No out-of-scope drift was changed in this tranche.

## Remaining Risks / Follow-ups

The generic package surface is intentionally narrow around supervised-run
process control. Future SFT, DPO, or QLoRA long-run wrappers should add their
own policy modules instead of widening nanoGPT-owned defaults.
