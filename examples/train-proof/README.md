# `examples/train-proof`

Canonical runnable proof entrypoint for the Phase 8 training surfaces.

It uses the official `meta-llama/Llama-3.2-1B-Instruct` checkpoint as the dense anchor, generates a repo-owned 4-bit snapshot if needed, and then runs four proof stages against pinned real Hugging Face subsets by default:

- LoRA on the dense model
- QLoRA on the 4-bit snapshot
- SFT on the dense model
- DPO on the dense model

The default dataset path is:

- `HuggingFaceH4/ultrachat_200k` for LoRA, QLoRA, and SFT
- `HuggingFaceH4/ultrafeedback_binarized` for DPO

The runner keeps deterministic front-slice subsets, filters out overlong examples against a configurable token cap, and reports held-out evaluation loss plus DPO reward-aware metrics for the preference stage.

LoRA target selection is now preset-driven rather than hardcoded. The proof uses:

- `attention` for LoRA and DPO
- `all-linear` for QLoRA

That keeps the orchestration readable while letting the model-family layer own which exact projection names each family exposes.

## Run

```bash
bun run proof:training
```

That defaults to the official Meta model, the real Hugging Face dataset path, and writes the generated 4-bit snapshot and JSON report under `.tmp/training-proof/`.

The repo also keeps a dedicated manual Apple Silicon GitHub workflow for this
proof so you can run the full canonical check on a self-hosted Mac when you
want it, without making every PR or push pay that cost. That runner must
already have access to the official checkpoint and pinned datasets through
`HF_TOKEN`, `HUGGINGFACE_HUB_TOKEN`, or the standard
`~/.cache/huggingface/token` file that the pretrained loader reads.

You can also override the proof size and output locations:

```bash
bun run proof:training --source meta-llama/Llama-3.2-1B-Instruct --train-limit 32 --eval-limit 8 --batch-size 4 --steps 4 --quantized-output .tmp/training-proof/meta-llama-Llama-3.2-1B-Instruct-4bit --adapter-output .tmp/training-proof/meta-llama-Llama-3.2-1B-Instruct-adapters --report .tmp/training-proof/meta-llama-Llama-3.2-1B-Instruct-report.json
```

For faster DPO iteration, you can run only the preference stage and switch to a
more handbook-aligned adapter recipe:

```bash
bun run proof:training --stages dpo --dpo-profile handbook --train-limit 128 --eval-limit 32
```

The stage selector accepts any comma-separated subset of `lora,qlora,sft,dpo`.
The DPO profile options are:

- `canonical` — the repo's lighter proof recipe
- `handbook` — a broader `attention+mlp` LoRA recipe with lower `beta`, lower learning rate, non-zero dropout, and full-decoder targeting for faster preference-tuning iteration

For a fast local smoke, you can still force the tiny built-in corpus:

```bash
bun run proof:training --dataset-source tiny --train-limit 8 --eval-limit 4 --steps 2
```

For a broader local family sweep, run the matrix wrapper:

```bash
bun run examples/train-proof/matrix.ts --dataset-source tiny --train-limit 8 --eval-limit 4 --steps 2
```

The matrix wrapper is also finite and agent-facing. It writes child proof
progress to stderr and emits one structured matrix summary on stdout. Matrix
usage errors exit `2`; child proof failures exit `1`.

The default matrix covers:

- `meta-llama/Llama-3.2-1B-Instruct`
- `google/gemma-3-1b-it`
- `google/gemma-4-E2B-it`
- `microsoft/Phi-4-mini-instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`

The proof report records:

- dataset source and filtering notes
- held-out evaluation loss before and after each stage
- resolved LoRA preset, target counts, and selected target paths for adapter-backed stages
- trainable and total parameter counts for every stage
- peak MLX memory evidence for every stage
- adapter save, reload, resample, merge, and post-merge resample evidence for LoRA, QLoRA, and DPO
- QLoRA merge preservation of the quantized base path
- DPO held-out reward accuracy, reward margin, chosen/rejected rewards, and chosen/rejected log-probs
- supplemental raw policy-only preference accuracy for DPO debugging
- a short sample after each stage
- verification checks that fail the runner when proof evidence is missing or malformed

The command is finite and agent-facing. Help, success summaries, and structured
errors are written to stdout; model loading, data preparation, stage metrics,
and sample progress are written to stderr. Usage errors exit `2`, and runtime or
proof failures exit `1`. The JSON report remains the detailed artifact.

Existing reports can be checked without rerunning training:

```bash
bun run examples/train-proof/verify-report.ts .tmp/training-proof/meta-llama-Llama-3.2-1B-Instruct-report.json
```

The verifier is a finite agent-facing command: stdout is compact structured
status or structured error output, and usage errors exit `2`.
