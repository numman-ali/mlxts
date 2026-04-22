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

The runner keeps deterministic front-slice subsets, filters out overlong examples against a configurable token cap, and reports held-out evaluation loss plus preference accuracy for DPO.

LoRA target selection is now preset-driven rather than hardcoded. The proof uses:

- `attention` for LoRA and DPO
- `all-linear` for QLoRA

That keeps the orchestration readable while letting the model-family layer own which exact projection names each family exposes.

## Run

```bash
bun run proof:training
```

That defaults to the official Meta model, the real Hugging Face dataset path, and writes the generated 4-bit snapshot and JSON report under `.tmp/training-proof/`.

You can also override the proof size and output locations:

```bash
bun run examples/train-proof/index.ts --source meta-llama/Llama-3.2-1B-Instruct --train-limit 32 --eval-limit 8 --batch-size 4 --steps 4 --quantized-output .tmp/training-proof/meta-llama-Llama-3.2-1B-Instruct-4bit --report .tmp/training-proof/meta-llama-Llama-3.2-1B-Instruct-report.json
```

For a fast local smoke, you can still force the tiny built-in corpus:

```bash
bun run examples/train-proof/index.ts --dataset-source tiny --train-limit 8 --eval-limit 4 --steps 2
```

For a broader local family sweep, run the matrix wrapper:

```bash
bun run proof:training:matrix --dataset-source tiny --train-limit 8 --eval-limit 4 --steps 2
```

The default matrix covers:

- `meta-llama/Llama-3.2-1B-Instruct`
- `google/gemma-3-1b-it`
- `google/gemma-4-E2B-it`
- `microsoft/Phi-4-mini-instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`

The proof report records:

- dataset source and filtering notes
- held-out evaluation loss before and after each stage
- resolved LoRA preset plus target counts for adapter-backed stages
- QLoRA merge preservation of the quantized base path
- DPO held-out preference accuracy before and after
- a short sample after each stage
