# Runtime Proof: Gemma 4 A4B MoE

## Summary

This proof records the first standalone Phase 7f real-checkpoint MoE evidence for
`mlx-community/gemma-4-26b-a4b-it-4bit`.

No production code changed in this tranche. The goal was to prove that the
landed Gemma 4 MoE and mixed-quant loading paths load a cached real checkpoint,
decode with flat memory, route through serving, and produce coherent text.

## Files Reviewed

- `packages/transformers/src/infrastructure/moe.ts`
- `packages/transformers/src/families/gemma4/config.ts`
- `packages/transformers/src/families/gemma4/block.ts`
- `packages/transformers/src/families/gemma4/moe.ts`
- `packages/transformers/src/families/gemma4/weights.ts`
- `packages/transformers/scripts/benchmark-generation-parity.ts`
- `packages/serve/src/engine/routing.ts`
- `packages/serve/scripts/benchmark-serve.ts`

## Checkpoint

- Model: `mlx-community/gemma-4-26b-a4b-it-4bit`
- Snapshot:
  `/Users/numman/.cache/huggingface/hub/models--mlx-community--gemma-4-26b-a4b-it-4bit/snapshots/695690b33533b1f8b0395c1d6b4f00dc411353ef`
- Local snapshot contents: `config.json`, tokenizer artifacts, generation
  config, chat template, safetensors index, and three safetensor shards.

## Evidence

Focused MoE, loader, and serving-route tests passed:

```bash
bun test packages/core/src/quantization.test.ts packages/transformers/src/infrastructure/moe.test.ts packages/transformers/src/families/qwen3_5/config.test.ts packages/transformers/src/families/qwen3_5/weights.test.ts packages/transformers/src/families/qwen3_5/mlp.test.ts packages/transformers/src/families/gemma4/config.test.ts packages/transformers/src/families/gemma4/moe.test.ts packages/transformers/src/families/gemma4/block.test.ts packages/transformers/src/families/gemma4/weights.test.ts packages/transformers/src/load.test.ts packages/serve/src/engine/routing.test.ts packages/serve/src/model-loading/server.test.ts packages/serve/src/model-loading/sources.test.ts
```

Result: `105 pass`, `0 fail`.

Real cached-checkpoint decode proof passed:

```bash
bun run bench:generation:parity --model mlx-community/gemma-4-26b-a4b-it-4bit --prompt-tokens 128 --generation-tokens 128 --trials 1 --memory-sample-interval 16 --skip-mlx-lm-reference
```

Result: `prompt_tps=529.659`, `generation_tps=108.604`,
`peak_memory=14.671 GB`, `active_start=14.527 GB`,
`active_end=14.527 GB`, `active_delta=-0.000 GB`,
`active_max=14.530 GB`, `active_slope_mb_per_token=-0.00`,
`evals_per_token=1.00`.

Real serve-path smoke passed:

```bash
bun run bench:serve --model mlx-community/gemma-4-26b-a4b-it-4bit --model-id gemma-a4b --rungs 128x32@1 --stream --greedy --ignore-eos --report-json .tmp/gemma-a4b-serve-smoke.json --request-timeout-ms 3600000 --no-warmup
```

Result: `routes=continuous:eligible=1`, `completion_tokens=32`,
`stream_chunks=32`, `peak_memory=14.732 GB`,
`active_memory=14.470 GB`, `active_delta=0.000 GB`,
`mean_post_ttft_completion_tps=95.470`, `max_stream_chunk_gap_ms=21.8`.

Readable chat-template generation proof passed:

```bash
printf 'Write one sentence explaining why mixture-of-experts models are useful.\nq\n' | bun run examples/chat/index.ts /Users/numman/.cache/huggingface/hub/models--mlx-community--gemma-4-26b-a4b-it-4bit/snapshots/695690b33533b1f8b0395c1d6b4f00dc411353ef --greedy --max-tokens 48
```

The model loaded the `15.6 GB` snapshot, detected the chat template, and
answered coherently:

> Mixture-of-experts models are useful because they allow for significantly increasing model capacity and knowledge without a proportional increase in computational cost, as only a small fraction of the parameters are activated for any given input.

## Tensor Lifetime Audit

No production tensor code changed. The proof exercised the existing MoE,
mixed-quant loader, generation, and serve paths. Decode evidence stayed at one
explicit eval per generated token, and active memory remained flat across the
128-token decode and the serving smoke.

## Independent Review

James performed a read-only scout before the proof. The recommendation was to
start with cached `mlx-community/gemma-4-26b-a4b-it-4bit`, then prove split
quantized Qwen A3B next. James also flagged that Mixtral is not the safest next
target because this repo does not yet have a Mixtral family registration.

## Out-of-scope Drift Noticed

`PLAN.md` still names Mixtral as the original Phase 7f exit example, but the
current implementation already has real Gemma 4 MoE support and Qwen 3.5/3.6
MoE seams. Mixtral remains future family work rather than the next proof target.

`shouldLoadQwen3_5ForConditionalGeneration()` accepts a top-level
`model_type: "qwen3_5_moe"` wrapper, while `parseQwen3_5Config()` currently
requires top-level `model_type: "qwen3_5"`. This should be fixed before claiming
conditional Qwen MoE serving for a real top-level MoE wrapper checkpoint.

## Remaining Risks / Follow-ups

Qwen A3B split-quantized MoE proof is still open. The recommended next target is
`unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit`, starting with direct text
`loadCausalLM()` generation and only then widening to conditional serving.

Publishable parity still requires an external `mlx-lm` reference run with
`--require-mlx-lm-reference`. This proof is a local runtime capability proof,
not a paired Python parity claim.
