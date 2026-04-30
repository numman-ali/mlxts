# Runtime Review: Qwen A3B MoE Parser and Proof

## Summary

This tranche makes top-level Qwen MoE conditional wrappers load through the
explicit Qwen conditional loader. The direct text `loadCausalLM()` path already
accepted `model_type: "qwen3_5_moe"` by parsing the nested text config; the
conditional loader detection also accepted that top-level model type. The parser
in between still rejected it and hardcoded the returned wrapper `modelType` to
`"qwen3_5"`.

The parser now preserves `"qwen3_5_moe"` for conditional wrappers and keeps the
unchanged `CausalLM` contract. Tests cover parser shape, tiny end-to-end
conditional MoE wrapper loading, and top-level Qwen MoE wrappers with
split-quantized `language_model.*` expert paths.

## Files Reviewed

- `packages/transformers/src/families/qwen3_5/config.ts`
- `packages/transformers/src/families/qwen3_5/config.test.ts`
- `packages/transformers/src/load.test.ts`

## Tensor Lifetime Audit

`config.ts` only parses host-side JSON config into typed config records. It does
not allocate or retain MLX tensors.

The new tests construct tiny Qwen MoE conditional fixtures, save them to a temp
snapshot, load them through the existing shard iterator, and dispose loaded and
original models with `using` / `Symbol.dispose`. The loaded forward pass follows
the existing Qwen conditional model path and compares logits after explicit
`mxEval`.

## Memory / Performance Evidence

Focused tests passed:

```bash
bun test packages/transformers/src/families/qwen3_5/config.test.ts packages/transformers/src/load.test.ts
```

Result: `42 pass`, `0 fail`.

Serving model-loading and split-quantized wrapper coverage passed:

```bash
bun test packages/transformers/src/families/qwen3_5/config.test.ts packages/transformers/src/load.test.ts packages/serve/src/model-loading/server.test.ts packages/serve/src/model-loading/sources.test.ts
```

Result: `64 pass`, `0 fail`.

Typecheck passed:

```bash
bun run typecheck
```

Coverage passed:

```bash
bun run check:coverage
```

Result: coverage thresholds satisfied across the canonical package stack.

Static gates passed:

```bash
bun run lint && bun run check:assertions && bun run check:file-lines && bun run check:tensor-lifetimes && bun run check:per-package-agents && bun run check:cross-package-imports
```

Result: all gates passed.

The real Qwen A3B checkpoint cache completed locally:

```bash
hf download unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit --include '*.safetensors' --include 'config.json' --include 'tokenizer.json' --include 'tokenizer_config.json' --include 'processor_config.json' --include 'chat_template.jinja' --include 'model.safetensors.index.json' --max-workers 5
```

The remote checkpoint is public and ungated at commit
`6700c3e5bdeb050a379c8d2a4133f43f3647f20f`; supported files total about
`20.17 GiB`. Its config advertises top-level `model_type: "qwen3_5_moe"`,
architecture `Qwen3_5MoeForConditionalGeneration`, nested
`text_config.model_type: "qwen3_5_moe_text"`, `40` text layers, `256` experts,
and `8` experts per token.

Direct checkpoint generation used `bun run bench:generation:parity`, the
generation benchmark evidence surface for `bun run bench:generation` when a
same-machine `mlx-lm` reference is not available:

```bash
bun run bench:generation:parity --model unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit --prompt-tokens 128 --generation-tokens 32 --trials 1 --memory-sample-interval 8 --skip-mlx-lm-reference
```

Result: `prompt_tps=675.349`, `generation_tps=89.171`,
`peak_memory=21.035 GB`, `active_start=20.816 GB`,
`active_end=20.816 GB`, `active_delta=0.000 GB`,
`active_slope_mb_per_token=0.01`, `evals_per_token=1.00`.

The longer direct decode proof also passed:

```bash
bun run bench:generation:parity --model unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit --prompt-tokens 128 --generation-tokens 128 --trials 1 --memory-sample-interval 16 --skip-mlx-lm-reference
```

Result: `prompt_tps=683.293`, `generation_tps=89.954`,
`peak_memory=21.035 GB`, `active_start=20.816 GB`,
`active_end=20.816 GB`, `active_delta=0.000 GB`,
`active_slope_mb_per_token=0.00`, `evals_per_token=1.00`.

Endpoint serving used the package-owned `bun run bench:serve` harness:

```bash
bun run bench:serve --model unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit --model-id qwen-a3b --rungs 128x32@1 --trials 1 --no-warmup --stream --greedy --ignore-eos --report-json .tmp/qwen36-a3b-serve-128x32.json --max-concurrent-requests 1 --max-batch-size 8 --batch-window-ms 2 --gpu-memory-utilization 0.85 --request-timeout-ms 3600000
```

Result: `routes=continuous:eligible=1`, `completion_tps=41.709`,
`mean_post_ttft_completion_tps=79.300`, `mean_ttft_ms=376.0`,
`peak_memory=21.056 GB`, `active_memory=20.746 GB`,
`active_delta=0.004 GB`, `continuous_admissions=1`,
`continuous_scheduler_phases=4`, `stream_chunks=29`,
`finish_reasons=stop`.

## Independent Review

Noether independently confirmed that direct text `loadCausalLM()` for
top-level `qwen3_5_moe` was not blocked. The only real gap was the explicit
conditional loader path used by `serveModel()` when a checkpoint advertises a
Qwen MoE conditional architecture and vision config. Noether also confirmed the
minimal fix: remove the extra conditional parser rejection and return the
parsed top-level model type.

## Remaining Risks / Follow-ups

Publishable parity still requires a paired `mlx-lm` reference run with
`--require-mlx-lm-reference`. The short qualitative chat smoke loaded the
checkpoint and template successfully, but the small output budget was spent on a
Qwen-style reasoning preamble; this tranche makes no answer-quality claim beyond
direct decode and serving stability.
