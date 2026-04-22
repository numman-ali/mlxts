# `examples/chat-canary`

Starter repo-owned chat dataset and held-out canary prompts for a repo-specific
LoRA or SFT pass.

This directory has two different assets on purpose.

`mlxts-chat-sft.jsonl`

- 60 chat-format records in one file
- first 45 are `train`
- last 15 are `eval`
- each row keeps repo metadata, but the main `messages` field is compatible with
  the existing `examples/lora-finetune` JSONL path

`mlxts-chat-canary.jsonl`

- 15 held-out prompt/ideal-response records
- meant for side-by-side chat comparison after tuning
- focused on repo-specific judgment, explanation quality, runtime discipline,
  package placement, and LoRA/training expectations

There is also a stronger follow-up dataset under
`examples/chat-canary/review-judge/`.

That review/judge set is deliberately narrower and larger:

- 250 train/eval records for LoRA-ready supervision
- 50 held-out canary prompts
- experimental proof asset for repo-native review behavior, not a final product
  benchmark

## Use for LoRA

Train against the repo-owned JSONL file with the existing LoRA example:

```bash
bun run example:lora-finetune \
  --source google/gemma-4-E2B-it \
  --dataset-source jsonl \
  --dataset-jsonl examples/chat-canary/mlxts-chat-sft.jsonl \
  --train-limit 45 \
  --eval-limit 15 \
  --batch-size 2 \
  --steps 8
```

That uses the first 45 rows for training and the last 15 rows for evaluation.

## Use for side-by-side chat

Open the base model and the adapted model side by side with the normal chat
example, then ask prompts from `mlxts-chat-canary.jsonl`.

The tuned model should show more repo-native behavior, for example:

- it prefers canonical `@mlxts/*` package surfaces over `packages/nanogpt`
- it keeps semantic surfaces readable and hides runtime strategy underneath
- it distinguishes benchmark fairness from flattering local runs
- it treats LoRA proofs, PEFT interop, and family presets in the repo's own terms
- it explains MLX, Bun FFI, and architecture choices in plain English without
  losing correctness

## Scope

This is a starter dataset, not the final truth set.

It is intentionally narrow and opinionated:

- architecture and package-boundary judgment
- runtime and benchmark discipline
- LoRA, QLoRA, SFT, and DPO expectations
- code review and refactor judgment
- plain-English explanations for a web developer learning the stack

If the tuned model visibly improves on these prompts, that is a good sign the
LoRA is teaching repo-specific habits rather than just making the model noisier.
