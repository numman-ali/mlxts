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

The proof report records:

- dataset source and filtering notes
- held-out evaluation loss before and after each stage
- QLoRA merge preservation of the quantized base path
- DPO held-out preference accuracy before and after
- a short sample after each stage
