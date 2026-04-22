# Runtime Review: Gemma 4 Review/Judge LoRA Proof

## Summary

This proof asked a narrow question:

Can a repo-owned Gemma 4 LoRA learn visibly better review and architectural
judgment if we stop training broad assistant behavior and instead train a
larger review/judge-specific dataset?

The answer is yes, partially.

The resulting model became more decisive and more repo-shaped on a meaningful
slice of held-out prompts, but it did not become a consistently strong
repo-native reviewer yet. The proof is real enough to keep as an experimental
asset, but not strong enough to present as a finished judge model.

## Files Reviewed

- `examples/chat-canary/README.md`
- `examples/chat-canary/dataset.test.ts`
- `examples/chat-canary/review-judge/README.md`
- `examples/chat-canary/review-judge/dataset.test.ts`
- `examples/chat-canary/review-judge/mlxts-review-judge-sft.jsonl`
- `examples/chat-canary/review-judge/mlxts-review-judge-canary.jsonl`

## Tensor Lifetime Audit

This proof does not introduce a new production tensor primitive or a new model
runtime path. The relevant tensor-lifetime question was whether the new
review/judge proof could run cleanly through the existing Gemma 4 LoRA path.

That question was answered by the paired Gemma 4 LoRA runs recorded below,
after the dedicated MLP compatibility fix landed under the separate runtime
review artifact.

The dataset assets themselves are JSONL and test surfaces only. They do not
introduce native ownership or array-lifetime behavior.

## Memory / Performance Evidence

This proof is about model behavior, not decode hot-path performance. There were
no new `bench:generation` or `bench:generation:parity` measurements because the
goal here was held-out answer quality rather than a new runtime optimization.

Evidence recorded:

- dataset validation:
  - `bun test examples/chat-canary/dataset.test.ts examples/chat-canary/review-judge/dataset.test.ts`
- workspace validation:
  - `bun run validate`

Training runs:

- `all-linear`, 160 steps:
  - command:
    `bun run example:lora-finetune --source google/gemma-4-E2B-it --mode lora --preset all-linear --dataset-source jsonl --dataset-jsonl examples/chat-canary/review-judge/mlxts-review-judge-sft.jsonl --train-limit 200 --eval-limit 50 --batch-size 2 --steps 160 --output-dir .tmp/lora-finetune/gemma4-review-judge-all-linear --report .tmp/lora-finetune/gemma4-review-judge-all-linear-report.json`
  - result:
    - `eval_loss_before=7.1888`
    - `eval_loss_after=3.8125`
    - `target_count=19`
  - held-out comparison summary:
    - improved prompts: `27 / 50`
    - tied prompts: `12 / 50`
    - worse prompts: `11 / 50`
    - mean overlap score:
      - base: `0.1465`
      - adapted: `0.1740`

- `all-linear`, 640 steps:
  - command:
    `bun run example:lora-finetune --source google/gemma-4-E2B-it --mode lora --preset all-linear --dataset-source jsonl --dataset-jsonl examples/chat-canary/review-judge/mlxts-review-judge-sft.jsonl --train-limit 200 --eval-limit 50 --batch-size 2 --steps 640 --output-dir .tmp/lora-finetune/gemma4-review-judge-all-linear-640 --report .tmp/lora-finetune/gemma4-review-judge-all-linear-640-report.json`
  - result:
    - `eval_loss_before=7.1888`
    - `eval_loss_after=3.4120`
    - `target_count=19`
  - held-out comparison summary:
    - improved prompts: `26 / 50`
    - tied prompts: `7 / 50`
    - worse prompts: `17 / 50`
    - mean overlap score:
      - base: `0.1465`
      - adapted: `0.1708`

Interpretation:

- the 160-step run is the better balanced proof result
- the 640-step run lowered eval loss further, but became more brittle on the
  held-out canary prompts
- the review/judge LoRA direction is promising, but not yet strong enough to
  present as a finished repo-native judge

## Independent Review

The proof loop itself was pressure-tested in two ways:

- the narrower review/judge dataset was built from five disjoint high-capability
  draft-generation passes and then deduplicated and normalized into one
  repo-owned train/eval surface
- the Gemma 4 MLP compatibility fix that made the proof possible was reviewed
  independently under
  `docs/reviews/2026-04-22-gemma4-mlp-lora-compatibility.md`

No independent reviewer found a blocker in the compatibility fix itself. The
remaining weakness was in the strength and consistency of the training signal,
not in the Gemma 4 LoRA machinery.

## Remaining Risks / Follow-ups

- This is still an experimental proof asset, not a finished product benchmark.
- The main bottleneck is now the training signal:
  - more narrow review/judge examples are needed
  - stronger contrast between good and bad judgments is needed
  - preference-style supervision is the most plausible next lever
- The 640-step run shows that lower eval loss alone is not enough; held-out
  behavioral comparisons still need to decide which adapter is actually better.
