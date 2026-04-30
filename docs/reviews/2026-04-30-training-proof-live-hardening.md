# Training Proof Live Hardening

## Summary

The Phase 8 training proof now records machine-checkable evidence for LoRA,
QLoRA, SFT, and DPO stages. Reports include adapter output location, selected
LoRA targets, trainable/total parameter counts, peak MLX memory, and adapter
save/reload/merge sampling equality evidence for adapter-backed stages.

## Files Reviewed

- `package.json`
- `examples/train-proof/args.ts`
- `examples/train-proof/matrix.ts`
- `examples/train-proof/workflow.ts`
- `examples/train-proof/stages.ts`
- `examples/train-proof/types.ts`
- `examples/train-proof/report-schema.ts`
- `examples/train-proof/verification.ts`
- `examples/train-proof/README.md`

## Runtime Sensitivity

The changed runtime path is the explicit training-proof command, not package
serving or model generation APIs. Adapter-backed proof stages now save the
trained adapter, load it into a fresh model, sample before and after reload,
merge the reloaded adapter, and sample again before the original stage merges.
The verifier requires greedy sample equality across trained, reloaded, and
reloaded-merged adapter states. The DPO loss verifier uses the configured beta
for both before and after loss measurement.

## Tensor Lifetime Audit

The new evidence path reads model parameter trees, records MLX peak memory, and
uses existing adapter save/load/merge helpers. No new tensor-producing
primitive is introduced. The proof still keeps loaded models in `using`
bindings, and the reloaded adapter-check model is scoped inside the helper.

## Memory / Performance Evidence

The proof records `memory.peakBytes` from MLX for every stage. Adapter checks
add one fresh model load per adapter-backed stage so the report can prove
save/reload/merge behavior; this is intentional proof cost and is isolated to
the manual training-proof surface. DPO saves the trained adapter before merging
the policy, releases the policy/reference pair, then loads the adapter-check
model so the proof does not keep three full DPO models live concurrently.

## Tests

- `bun test examples/train-proof/helpers.test.ts examples/train-proof/verification.test.ts examples/lora-finetune/helpers.test.ts`
- `bun run check:training-proofs`
- `bun run typecheck`
- `bun run lint`
- `bun run proof:training -- --dataset-source tiny --train-limit 2 --eval-limit 1 --batch-size 1 --steps 1 --stages lora --report .tmp/training-proof/tiny-lora-hardening-report.json --adapter-output .tmp/training-proof/tiny-lora-hardening-adapters --quantized-output .tmp/training-proof/tiny-lora-hardening-4bit`
- `bun run examples/train-proof/verify-report.ts .tmp/training-proof/tiny-lora-hardening-report.json`
- `bun run proof:training -- --dataset-source tiny --train-limit 2 --eval-limit 1 --batch-size 1 --steps 1 --stages qlora --report .tmp/training-proof/tiny-qlora-hardening-report.json --adapter-output .tmp/training-proof/tiny-qlora-hardening-adapters --quantized-output .tmp/training-proof/meta-llama-Llama-3.2-1B-Instruct-4bit`
- `bun run examples/train-proof/verify-report.ts .tmp/training-proof/tiny-qlora-hardening-report.json`
- `bun run proof:training -- --dataset-source tiny --train-limit 2 --eval-limit 1 --batch-size 1 --steps 1 --stages sft --report .tmp/training-proof/tiny-sft-hardening-report.json --adapter-output .tmp/training-proof/tiny-sft-hardening-adapters --quantized-output .tmp/training-proof/tiny-sft-hardening-4bit`
- `bun run examples/train-proof/verify-report.ts .tmp/training-proof/tiny-sft-hardening-report.json`
- `bun run proof:training -- --dataset-source tiny --train-limit 2 --eval-limit 1 --batch-size 1 --steps 1 --stages dpo --report .tmp/training-proof/tiny-dpo-hardening-report.json --adapter-output .tmp/training-proof/tiny-dpo-hardening-adapters --quantized-output .tmp/training-proof/tiny-dpo-hardening-4bit`
- `bun run examples/train-proof/verify-report.ts .tmp/training-proof/tiny-dpo-hardening-report.json`
- `bun run validate`

## Independent Review

Avicenna reviewed the broader roadmap and recommended hardening Phase 8 live
proofs before speculative Phase 10 widening. The implementation follows that
recommendation by strengthening report evidence for LoRA, QLoRA, SFT, and DPO
without adding a trainer framework or widening model contracts.

Avicenna's follow-up diff review found that adapter reload samples needed
equality checks, DPO profile knobs needed explicit verification, and DPO reload
proof memory needed attention. The verifier now enforces greedy sample equality
and profile-specific DPO knobs, and the DPO reload check runs after the
policy/reference pair is released.

## Out-of-scope Drift Noticed

No Phase 10 multimodal or diffusion implementation belongs in this tranche.
Fine-tuning orchestration beyond the canonical proof command remains future
work after the proof evidence is stable.

## Remaining Risks / Follow-ups

The full Hugging Face proof is intentionally expensive and depends on local
checkpoint and dataset availability. The focused gates prove schema and runner
behavior; the full canonical run remains the final live acceptance signal.
