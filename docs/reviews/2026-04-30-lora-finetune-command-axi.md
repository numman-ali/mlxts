# LoRA Finetune Command AXI Boundary

## Summary

The readable `examples/lora-finetune` command now parses help and usage errors
before acquiring the shared MLX runtime lock. Progress and sample logs move to
stderr, while stdout is reserved for compact structured help, success, and
error output with stable exit codes.

## Files Reviewed

- `package.json`
- `examples/lora-finetune/args.ts`
- `examples/lora-finetune/cli.ts`
- `examples/lora-finetune/cli.test.ts`
- `examples/lora-finetune/index.ts`
- `examples/lora-finetune/workflow.ts`
- `examples/lora-finetune/README.md`

## Runtime Sensitivity

The change is a command-boundary migration around the existing LoRA and QLoRA
example flow. It does not change adapter application, target resolution,
optimizer settings, dataset preparation, model loading, adapter save/reload,
merge checks, sampling, or report contents. It changes when the runtime lock is
acquired and which output channel receives progress versus agent-consumable
command data.

## Tensor Lifetime Audit

No new tensor-producing operation or retained tensor owner is introduced. The
existing workflow keeps loaded models in `using` scopes, and the new command
wrapper only formats parsed options, report summaries, and errors.

## Memory / Performance Evidence

No performance claim is made. Help and usage errors now return before the
runtime lock and before any checkpoint, quantization, dataset, or training work.
Successful example runs still execute the same workflow and still write the same
JSON report shape.

## Tests

- `bun test examples/lora-finetune/helpers.test.ts examples/lora-finetune/cli.test.ts`
- `bun run check:training-proofs`
- `bun run examples/lora-finetune/index.ts --help`
- `bun run examples/lora-finetune/index.ts --train-limit --steps`

## Independent Review

Locke performed a read-only second opinion and recommended the same
command-boundary-only shape: add a CLI wrapper, parse help and usage errors
before the runtime lock, route progress to stderr, keep stdout compact, preserve
the existing `FinetuneReport` JSON schema, and avoid any changes to LoRA math,
data preparation, runtime helpers, adapter save/load, or sampling.

## Out-of-scope Drift Noticed

`examples/train-proof/matrix.ts` still wraps heavy proof runs with inherited
child output and human-oriented progress. It remains a separate AXI tranche now
that the canonical `proof:training` and readable LoRA example commands have
structured command boundaries.

## Remaining Risks / Follow-ups

The example remains a teaching surface rather than the canonical Phase 8 proof.
The full official-model LoRA or QLoRA run remains expensive and depends on local
checkpoint and dataset availability. Focused tests cover command contract,
parsing, output channels, and report summary formatting; live model proof stays
under the heavier training-proof acceptance path.
