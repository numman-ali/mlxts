# `examples/lora-finetune`

Readable end-to-end LoRA and QLoRA fine-tuning example for supported causal decoder families.

The example keeps composition visible:

- load model and tokenizer assets
- resolve a readable LoRA target preset
- apply adapters
- train with `@mlxts/align`
- save adapters
- reload adapters into a fresh model
- merge adapters back into the base model
- compare short deterministic generations

It is intentionally example-first rather than framework-heavy. The reusable primitives still live in `@mlxts/transformers`, `@mlxts/lora`, `@mlxts/align`, and `@mlxts/train`.

## Run

Dense LoRA against a real Hugging Face chat dataset:

```bash
bun run examples/lora-finetune/index.ts --source meta-llama/Llama-3.2-1B-Instruct --mode lora --dataset-source huggingface
```

QLoRA against the same dataset:

```bash
bun run examples/lora-finetune/index.ts --source meta-llama/Llama-3.2-1B-Instruct --mode qlora
```

Fast local smoke with the built-in tiny corpus:

```bash
bun run examples/lora-finetune/index.ts --source google/gemma-3-1b-it --dataset-source tiny --train-limit 8 --eval-limit 4 --batch-size 2 --steps 2
```

Verify an existing report without rerunning training:

```bash
bun run examples/lora-finetune/verify-report.ts .tmp/lora-finetune/google-gemma-3-1b-it-report.json
```

## Important options

- `--mode lora|qlora`
- `--preset attention|attention+mlp|all-linear`
- `--adapter-format mlxts|peft`
- `--dataset-source tiny|huggingface|jsonl`
- `--dataset-jsonl /path/to/data.jsonl`
- `--source <hugging-face-model-id>`
- `--output-dir <directory>`
- `--report <path>`

Default preset behavior is intentionally simple:

- LoRA defaults to `attention`
- QLoRA defaults to `all-linear`

## Output

The command is finite and agent-facing. `--help`, success, and errors write
compact structured output to stdout. Model loading, dataset preparation,
training progress, sample text, and stack traces write to stderr. Usage errors
exit with code `2`; runtime or training failures exit with code `1`.

The report JSON records:

- the selected source, mode, preset, and adapter format
- held-out loss before and after training
- one per-step training loss entry for each configured optimizer step
- target count and selected target paths for the resolved preset
- trainable/total parameter counts and peak MLX memory
- train/eval data preparation stats
- the saved adapter directory
- adapter reload equality and QLoRA quantized-base preservation evidence
- deterministic sample text from the trained, reloaded, and merged models

The verifier checks report shape, finite loss metrics, per-step loss trace
shape and mean consistency, a positive target count, parameter/memory evidence,
data preparation counts, non-empty sample text, exact trained/reloaded
deterministic sample equality, and QLoRA base preservation when `--mode qlora`
produced the report.

When `--adapter-format peft` is used, the example writes first-pass PEFT-compatible causal LM LoRA checkpoints:

- `adapter_config.json`
- `adapter_model.safetensors`

This first pass is intentionally strict. It supports standard single-adapter causal LM LoRA only, not PEFT features such as `modules_to_save`, DoRA, RSLoRA, or multi-adapter bundles.
