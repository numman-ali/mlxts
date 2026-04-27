# Runtime Review: Gemma 4 31B MLX Community Load

## Summary

The canonical `mlx-community/gemma-4-31b-it-4bit` checkpoint now loads through
the existing Gemma 4 text-only CausalLM path. The upstream top-level config uses
`text_config.use_bidirectional_attention: "vision"` to mark multimodal wrapper
behavior; text-only loading can accept that marker while still rejecting
unsupported bidirectional attention modes.

The checkpoint also uses the `language_model.model.*` tensor namespace, while
the previous top-level Gemma 4 mapper only accepted `model.language_model.*`.
The mapper now accepts both observed namespace forms and continues to ignore
vision-only weights for text-only CausalLM loading.

## Files Reviewed

- `packages/transformers/src/families/gemma4/config.ts`
- `packages/transformers/src/families/gemma4/weights.ts`
- `packages/transformers/src/families/config.test.ts`

## Tensor Lifetime Audit

This change does not add new tensor operations. It only changes config parsing
and checkpoint-name translation before safetensor iteration reaches tensor
assignment. Existing quantized staging and assignment ownership rules remain
unchanged.

The 31B checkpoint has `hidden_size_per_layer_input: 0`, so the exceptional
Gemma 4 per-layer embedding loader is not exercised by this smoke.

## Memory / Performance Evidence

Focused tests:

```bash
bun test packages/transformers/src/families/config.test.ts
bun test packages/transformers/src/families/gemma4/weights.test.ts
```

Real local serving smoke:

```bash
bun run bench:serve \
  --model /Users/numman/.cache/huggingface/hub/models--mlx-community--gemma-4-31b-it-4bit/snapshots/dcb78c3f5d6becacbfce71cd4851ad98c4f08a05 \
  --protocol completions \
  --stream \
  --rungs 128x16@1 \
  --trials 1 \
  --greedy \
  --ignore-eos \
  --max-concurrent-requests 1 \
  --max-batch-size 1 \
  --prefill-step-size 512 \
  --request-timeout-ms 3600000 \
  --report-json .tmp/gemma4-31b-serve-smoke.json
```

Result: `128` prompt tokens and `16` streamed completion tokens completed with
`mean_ttft_ms=776.2`, `mean_server_prefill_tps=195.541`,
`mean_post_ttft_completion_tps=27.703`, `peak_memory=17.755 GB`,
`active_memory=17.272 GB`, and `finish_reasons=stop`.

## Independent Review

A second-opinion explorer independently confirmed the same two-part diagnosis:
`"vision"` means the bidirectional mask applies only to vision tokens while
normal text remains causal, and the mlx-community snapshot stores text weights
under `language_model.model.*`. The implementation was kept intentionally
narrow: accept only the upstream `"vision"` marker for text-only loading and add
the missing `language_model.` namespace without changing model execution
semantics.

## Remaining Risks / Follow-ups

This proves text-only completions serving for the mlx-community 4-bit 31B
checkpoint. It does not claim image/video execution for Gemma 4; that still
needs the planned multimodal composition path.

The run used a single-request smoke with `--max-batch-size 1`, so continuous
batching and long-context behavior for this 31B artifact still need explicit
follow-up benchmarks before we treat it as fully characterized.
