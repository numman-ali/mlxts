# `examples/chat-canary/review-judge`

Larger repo-owned review and architectural-judgment dataset for a stronger
repo-native LoRA proof.

This directory keeps the same two-surface pattern as the starter
`examples/chat-canary/` set, but with a narrower and stronger task:
reviewing proposed changes in the repo's own terms.

`mlxts-review-judge-sft.jsonl`

- 250 chat-format records in one file
- first 200 are `train`
- last 50 are `eval`
- each row keeps review metadata, but the `messages` field is compatible with
  the existing `examples/lora-finetune` JSONL path

`mlxts-review-judge-canary.jsonl`

- 50 held-out review prompts with ideal answers
- meant for base-vs-adapted comparison after tuning
- focused on architectural judgment, runtime discipline, example readability,
  LoRA proof realism, and public API boundaries

## Why this set exists

The starter chat dataset was broad enough to prove the LoRA stack worked, but
too broad to produce a strong visible shift in answer quality. This set narrows
the target to one task that should be easier to learn and easier to compare:
repo-native review and judgment.

## Use for LoRA

Train against the repo-owned JSONL file with the existing LoRA example:

```bash
bun run example:lora-finetune \
  --source google/gemma-4-E2B-it \
  --mode lora \
  --preset all-linear \
  --dataset-source jsonl \
  --dataset-jsonl examples/chat-canary/review-judge/mlxts-review-judge-sft.jsonl \
  --train-limit 200 \
  --eval-limit 50 \
  --batch-size 2 \
  --steps 128
```

## Use for side-by-side comparison

Ask the same held-out prompts from
`mlxts-review-judge-canary.jsonl` to the base model and the adapted model.

The tuned model should become more decisive and more repo-native in areas like:

- package placement by generation paradigm instead of modality
- preserving `CausalLM` as the universal autoregressive contract
- keeping runtime strategy under semantic surfaces
- requiring paired `mlx-lm` evidence for performance claims
- pushing back on over-abstraction and helper sprawl in examples
- treating LoRA and QLoRA as visible transformations, not hidden magic
