# Runtime Review: Gemma 4 MLP LoRA and Quantized Compatibility

## Summary

This change fixes a real Gemma 4 family compatibility gap that showed up during
the larger review/judge LoRA proof.

`Gemma4TextMLP.forward()` previously assumed all three MLP projections were
plain dense `Linear` modules and always passed their raw `.weight` tensors into
the compiled dense fast path. That is correct for the ordinary dense inference
surface, but it is not correct once those projections are wrapped or replaced
by `LoRALinear` or `QuantizedLinear`.

The fix keeps the existing fast path for the dense case and falls back to the
ordinary semantic projection flow for wrapped and quantized projections. The
goal is correctness and composability first, without disturbing the dense path
that matters for the normal Gemma 4 runtime story.

## Files Reviewed

- `packages/transformers/src/families/gemma4/mlp.ts`
- `packages/transformers/src/families/gemma4/mlp.test.ts`

## Tensor Lifetime Audit

The dense path still delegates to `runMlp(...)` exactly as before when the
runtime projections are all plain `Linear` modules.

The fallback path keeps native tensor ownership explicit in the same style as
the rest of the runtime-sensitive code:

- `gateProjection.forward(x)` is bound to `using gate`
- `upProjection.forward(x)` is bound to `using value`
- `gegluApprox(gate, value)` is bound to `using activated`
- `downProjection.forward(activated)` is returned directly as the owned result

No borrowed tensor view is retained across calls, and the fallback does not add
shared mutable tensor state.

## Memory / Performance Evidence

The important performance constraint here is not to disturb dense Gemma 4.

Evidence reviewed:

- focused family regression tests:
  - `bun test packages/transformers/src/families/gemma4/mlp.test.ts`
- workspace typing:
  - `bun run --filter @mlxts/transformers typecheck`
- repo-owned chat-canary coverage:
  - `bun test examples/chat-canary/dataset.test.ts examples/chat-canary/review-judge/dataset.test.ts`
- dense-generation benchmark note:
  - no new `bench:generation` or `bench:generation:parity` run was taken for
    this patch because the dense `Linear` fast path is structurally unchanged
    and this change only redirects wrapped and quantized projections onto the
    semantic fallback path
- end-to-end Gemma 4 LoRA proof that originally exposed the bug:
  - `bun run example:lora-finetune --source google/gemma-4-E2B-it --mode lora --preset all-linear --dataset-source jsonl --dataset-jsonl examples/chat-canary/review-judge/mlxts-review-judge-sft.jsonl --train-limit 200 --eval-limit 50 --batch-size 2 --steps 160 --output-dir .tmp/lora-finetune/gemma4-review-judge-all-linear --report .tmp/lora-finetune/gemma4-review-judge-all-linear-report.json`
  - result:
    - `eval_loss_before=7.1888`
    - `eval_loss_after=3.8125`
    - `target_count=19`
- stronger follow-up Gemma 4 LoRA proof on the same surface:
  - `bun run example:lora-finetune --source google/gemma-4-E2B-it --mode lora --preset all-linear --dataset-source jsonl --dataset-jsonl examples/chat-canary/review-judge/mlxts-review-judge-sft.jsonl --train-limit 200 --eval-limit 50 --batch-size 2 --steps 640 --output-dir .tmp/lora-finetune/gemma4-review-judge-all-linear-640 --report .tmp/lora-finetune/gemma4-review-judge-all-linear-640-report.json`
  - result:
    - `eval_loss_before=7.1888`
    - `eval_loss_after=3.4120`
    - `target_count=19`

Interpretation:

- the fix is necessary for LoRA and QLoRA-style Gemma 4 MLP projections to run
  correctly
- the dense fast path remains structurally present and unchanged for ordinary
  `Linear` projections
- this patch is a compatibility correction, not a new dense-performance
  experiment

## Independent Review

A separate high-capability code review was requested specifically for this
change, focusing on:

- whether the dense fast path remains intact
- whether the wrapped and quantized fallback is semantically correct
- whether the tests cover the real failure mode

That review found no critical issues with the dense/fallback split and no
missing correctness blocker for the current LoRA and quantized scope. The main
follow-up suggested by the reviewer was to add one mixed replacement test later
for the case where only one MLP projection is wrapped or quantized, rather than
all three.

## Remaining Risks / Follow-ups

- The wrapped and quantized fallback is correctness-first, not a new optimized
  quantized fast path. If Gemma 4 QLoRA inference speed becomes a product
  concern later, it should earn a dedicated quantized-safe optimization path
  rather than reusing the dense shortcut by force.
- The focused tests cover dense, fully quantized, and fully QLoRA-wrapped MLP
  projections. A mixed replacement case would make the branch coverage even
  tighter.
- The larger review/judge LoRA proof improves visible answer quality in some
  held-out prompts, but it is not yet a dramatic across-the-board repo-native
  reviewer. The stronger next lever is more focused data and, likely,
  preference-style supervision rather than pretending this one patch solved the
  full judge-model problem.
