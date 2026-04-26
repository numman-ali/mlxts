# Runtime Review: Qwen Static Hybrid Batching

## Summary

Qwen 3.6 text greedy non-streaming serving can now use the static batch path
without pretending Qwen is a plain full-KV model. The transformer layer owns a
Qwen-specific hybrid batch cache: full-attention layers use batched KV state,
while linear-attention layers keep batched convolution and recurrent state with
left-padding masks.

Serving routes only eligible non-streaming greedy Qwen text requests to static
batching. Qwen streaming, sampled/model-native-default requests, and continuous
scheduler paths remain single-route fallbacks until their cache semantics are
implemented for real.

## Files Reviewed

- `packages/core/native/mlxts_core_ops.cpp`
- `packages/core/src/fast.ts`
- `packages/core/src/ffi/symbols.ts`
- `packages/transformers/src/families/qwen3_5/batch-cache.ts`
- `packages/transformers/src/families/qwen3_5/attention.ts`
- `packages/transformers/src/families/qwen3_5/block.ts`
- `packages/transformers/src/families/qwen3_5/gated-delta.ts`
- `packages/transformers/src/families/qwen3_5/gated-delta-recurrence.ts`
- `packages/transformers/src/families/qwen3_5/model.ts`
- `packages/transformers/src/infrastructure/cache/batch.ts`
- `packages/transformers/src/infrastructure/generation/batch.ts`
- `packages/serve/src/transformers-engine-routing.ts`

## Tensor Lifetime Audit

The new `Qwen3_5TextBatchCache` follows the existing managed-cache ownership
pattern. Full-attention KV arrays are owned by `BatchKVCache`; linear
convolution and recurrent state arrays are retained on update, disposed before
replacement, filtered by batch row, and freed on cache disposal. Extracting a
single row creates a normal `Qwen3_5TextCache` and releases temporary slices
after the single cache has retained what it needs.

Qwen full-attention batch RoPE offsets and left-padding masks are disposable
tensors and are freed around attention. Linear-attention masks are created once
per layer call, passed into the gated-delta recurrence/native helper, and freed
after the linear block completes.

The native masked gated-delta helper preserves the unmasked ABI and adds a
separate masked symbol. Masked recurrent steps write zero output and keep the
previous recurrent state, while the TypeScript fallback uses the same semantics
as an oracle.

`bun run check:tensor-lifetimes` reports no suspicious nested tensor-producing
calls, and `bun run check:assertions` reports no production type assertions.

## Memory / Performance Evidence

Focused checks run locally:

- `bun run build:native` after forcing the stale native dylib to rebuild
- `bun test packages/core/src/fast.test.ts packages/transformers/src/families/qwen3_5/cache.test.ts packages/transformers/src/families/qwen3_5/gated-delta.test.ts packages/transformers/src/families/qwen3_5/model.test.ts`
- `bun test packages/serve/src/transformers-engine.test.ts packages/serve/scripts/regression-serve-matrix.test.ts`
- `bun run --filter '@mlxts/core' typecheck`
- `bun run --filter '@mlxts/transformers' typecheck`
- `bun run --filter '@mlxts/serve' typecheck`
- `bun run check:assertions`
- `bun run check:file-lines`
- `bun run check:tensor-lifetimes`
- `bun run bench:generation --model mlx-community/Qwen3.6-27B-4bit --prompt-tokens 1024 --generation-tokens 128 --trials 1 --memory-sample-interval 16`
- `bun run bench:generation:parity --model mlx-community/Qwen3.6-27B-4bit --prompt-tokens 1024 --generation-tokens 128 --trials 1 --memory-sample-interval 16 --skip-mlx-lm-reference`
- `bun run regression:qwen-gemma -- --profile real --report-dir .tmp/qwen-gemma-regression-qwen-static`

Focused unit results:

- Qwen/core focused tests: 24 pass, 0 fail.
- Serve focused tests: 35 pass, 0 fail.
- Real Qwen/Gemma wrapper: transformer focused tests 69 pass, serve focused tests
  134 pass, then cached real endpoint rungs passed all budgets.
- Direct Qwen generation benchmark: `prompt_tps=253.410`,
  `generation_tps=28.636`, `peak_memory=17.184 GB`,
  `active_slope_mb_per_token=0.13`, `evals_per_token=1.00`.
- Qwen parity benchmark run inside the real regression wrapper:
  `prompt_tps=248.975`, `generation_tps=29.025`,
  `peak_memory=17.184 GB`, `active_slope_mb_per_token=0.14`,
  `evals_per_token=1.00`; this run intentionally skipped live `mlx-lm`
  capture because the real wrapper is a local cached smoke guardrail.

Real endpoint evidence from
`.tmp/qwen-gemma-regression-qwen-static/serve/qwen36-completions-static.json`:

- Qwen 3.6 `128x32@2` non-streaming greedy route:
  `static:eligible=2`.
- Static counters: `static_batches=1`, `static_batch_rows=2`,
  `max_generation_batch=2`.
- Continuous counters: `continuous_admissions=0`,
  `continuous_scheduler_phases=0`, `max_continuous_batch=0`.
- Completion throughput: `25.959 tok/s`, peak memory `16.244 GB`,
  active delta effectively flat.

Control evidence:

- Qwen streaming `1024x128@1` remained `single:unsupported_model_type=1`,
  with zero static and continuous counters, `28.762 tok/s` post-TTFT, and
  `17.184 GB` peak memory.
- Gemma 4 non-streaming static control remained `static:eligible=2`, one
  static batch with two rows, zero continuous counters, and `64.876 tok/s`.
- Gemma 4 streaming remained `single:sliding_window_cache=1`, with zero static
  and continuous counters.

## Independent Review

Helmholtz reviewed the Qwen hybrid batch-cache design before implementation and
recommended a Qwen-owned cache with separate full-attention KV, linear recurrent
state, linear left-padding masks, and masked gated-delta semantics. Cicero
reviewed the serving route shape and recommended adding `qwen3_5_text` to
static eligibility only, keeping Qwen out of continuous batching, and updating
real regression budgets to assert static counters with zero continuous
counters. Avicenna implemented the bounded native masked gated-delta slice,
which was reviewed and integrated here.

## Remaining Risks / Follow-ups

This is static greedy batching only. Qwen continuous batching still needs a
scheduler-aware hybrid cache that can extend, filter, and admit rows across
decode steps. Qwen streaming batching and sampled/model-native-default batch
generation remain separate work.

`Qwen3_5TextBatchCache.extend()` intentionally throws so the current scheduler
cannot accidentally treat the hybrid cache as continuous-ready. If a future
all-linear Qwen variant appears, cache length/offset semantics should be
rechecked because the current real target has full-attention layers.
