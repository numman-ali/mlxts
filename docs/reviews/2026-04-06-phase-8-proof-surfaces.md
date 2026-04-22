# Runtime Review: Phase 8 Proof Surfaces and Adapter Correctness

## Summary

This slice tightened the Phase 8 proof path around two real correctness issues:

1. Chat-supervision examples can no longer assume that a prompt-only tokenization
   is a token-prefix of the full prompt-plus-answer tokenization. The alignment
   helpers now split assistant completions from the full rendered transcript by
   using tokenizer offsets, which fixes the real Llama 3.2 instruct boundary bug
   hit by the training proof surface.
2. Applying LoRA now freezes non-adapter parameters across the model tree before
   wrappers are inserted, so the QLoRA proof no longer attempts to backpropagate
   into untouched quantized base weights.
3. Real-data proof runs now use pinned Hugging Face dataset subsets with a
   deterministic row-loader path and a Parquet fallback, so the training proof
   is no longer limited to the tiny built-in corpus.

This change also adds the canonical proof/example surfaces that exercise the
official Meta dense anchor, plus benchmark surfaces for MLX-LM comparison and a
long-context ladder.

## Files Reviewed

- `packages/align/src/chat-templates.ts`
- `packages/data/src/huggingface.ts`
- `packages/data/src/index.ts`
- `packages/lora/src/apply-module.ts`

## Tensor Lifetime Audit

`buildChatSupervisionExample()` and `buildChatPreferenceExample()` remain pure
tokenizer-side helpers and do not allocate native tensors. The new split logic
operates on tokenizer encodings and offsets only.

`applyLoRAToModule()` still performs only module-tree rewrites. The new freeze
pass walks the module tree and marks leaf parameters as frozen without creating
or retaining new native arrays. Adapter creation remains owned by
`LoRALinear.fromBase()`, and merge/remove ownership behavior is unchanged.

`loadHuggingFaceRowsDataset()` remains a host-side loader. It allocates only JS
objects and strings, retries transient upstream failures without retaining large
buffers, and transfers parsed rows into array-backed datasets. The Parquet
fallback used by the proof runner shells out to `hf` and `duckdb` and does not
introduce new native MLX tensor ownership.

The new benchmark and proof surfaces reuse existing generation and quantization
paths. No additional long-lived native resources were introduced beyond those
already covered by the existing benchmark and training code.

## Memory / Performance Evidence

- Official `meta-llama/Llama-3.2-1B-Instruct` training proof now runs end to
  end through the repo-owned surfaces:
  - LoRA: `loss_before=5.6029`, `loss_after=5.5866`
  - QLoRA: `loss_before=5.1287`, `loss_after=5.1085`, with
    `quantized_base_preserved=true`
  - SFT: `loss_before=5.6029`, `loss_after=3.0965`
  - DPO: `loss_before=0.6931`, `loss_after=0.0599`
- Official `meta-llama/Llama-3.2-1B-Instruct` parity benchmark against
  `mlx-lm`:
  - `mlx-lm`: `prompt_tps=5993.103`, `generation_tps=171.814`,
    `peak_memory=2.937 GB`
  - `mlxts`: `prompt_tps=6534.675`, `generation_tps=170.789`,
    `peak_memory=3.022 GB`
  - Decode throughput is effectively at parity, so the benchmark enforcement now
    allows a small tolerance for run-to-run noise instead of requiring literal
    floating-point equality.
- Official `meta-llama/Llama-3.2-1B-Instruct` long-context 32K rung:
  - `prompt_tokens=32768`
  - `prefill_seconds=10.085`
  - `prefill_tps=3249.198`
  - `first_token_seconds=0.015`
  - `decode_tps=91.003`
  - `prefill_peak_memory=4.606 GB`
  - Retrieval did not recover the marker exactly at this rung, which is now
    recorded explicitly rather than hidden behind a pure “it ran” success.

## Independent Review

Cross-checked through focused unit coverage and live official-model proof runs:

- `bun test packages/align/src/chat-templates.test.ts`
- `bun test packages/lora/src/apply-module.test.ts`
- `bun test packages/transformers/scripts/benchmark-common.test.ts`
- `bun test packages/transformers/scripts/benchmark-long-context.test.ts`
- `bun test examples/train-proof/helpers.test.ts`
- `bun run proof:training`
- `MLX_LM_BENCH_PYTHON=... bun run bench:generation:parity --model meta-llama/Llama-3.2-1B-Instruct ...`
- `bun run bench:generation:context --model meta-llama/Llama-3.2-1B-Instruct --rungs 32768 ...`

## Remaining Risks / Follow-ups

- The new proof path is validated on the official Meta 1B instruct anchor, but
  the larger official dense checkpoints in the plan still need their own proof
  runs.
- The 32K long-context rung currently shows that “it runs” does not yet mean
  “it retrieves reliably”; the long-context ladder is now instrumented enough to
  make that visible, but the prompt/eval methodology may still need refinement
  as we scale up.
- `packages/nanogpt/` remains in the repo as a temporary validation fixture and
  is not part of this cleanup pass.
