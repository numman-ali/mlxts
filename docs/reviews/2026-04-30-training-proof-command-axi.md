# Training Proof Command AXI Boundary

## Summary

The canonical `bun run proof:training` command now parses help and usage errors
before acquiring the shared MLX runtime lock. Progress and stage logs move to
stderr, while stdout is reserved for compact structured help, success, and
error output with stable exit codes.

## Files Reviewed

- `package.json`
- `examples/train-proof/args.ts`
- `examples/train-proof/cli.ts`
- `examples/train-proof/cli.test.ts`
- `examples/train-proof/index.ts`
- `examples/train-proof/stages.ts`
- `examples/train-proof/workflow.ts`
- `examples/train-proof/README.md`

## Runtime Sensitivity

The change is a command-boundary migration around an existing heavy proof path.
It does not change LoRA, QLoRA, SFT, DPO training math, model loading, dataset
preparation, adapter save/reload/merge checks, or report verification. It
changes when the runtime lock is acquired and which output channel receives
progress versus agent-consumable command data.

## Tensor Lifetime Audit

No new tensor-producing operation or retained tensor owner is introduced. The
existing workflow keeps loaded models in `using` scopes, and the new command
wrapper only formats parsed options, report summaries, and errors.

## Memory / Performance Evidence

No performance claim is made. Help and usage errors now return before the
runtime lock and before any checkpoint or dataset work. Successful proof runs
still execute the same training workflow and still record `memory.peakBytes` in
the JSON report.

## Tests

- `bun run check:training-proofs`
- `bun run typecheck`
- `bun run lint`
- `bun run validate`

## Independent Review

Locke performed a read-only Phase 9.5 audit and ranked `proof:training` as the
highest-leverage remaining AXI tranche because it is the canonical Phase 8 root
proof command and previously locked before parsing, wrote progress to stdout,
and surfaced raw errors.

## Out-of-scope Drift Noticed

`examples/lora-finetune/index.ts` and `examples/train-proof/matrix.ts` are still
human-log-shaped command boundaries. They remain separate AXI tranches so this
commit can preserve the canonical proof's training behavior exactly.

## Remaining Risks / Follow-ups

The full official-model proof remains expensive and depends on local checkpoint
and dataset availability. Focused tests cover command contract, parsing, and
report summary formatting; the full proof remains the live Phase 8 acceptance
signal when runtime budget is available.
