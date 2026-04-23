# Continuity

This file is a compact handoff for long-running `mlxts` work. Keep durable
doctrine in `AGENTS.md`, durable learnings in `MEMORY.md`, and use this file for
current-phase state that should survive context compaction.

## Current Focus

Qwen 3.6 27B serving/inference quality is the active critical path. The goal is
not just to finish a short benchmark; it is to keep staged parity evidence
against `mlx-lm`, long-output stability, and long-context capability visible.

## Current Qwen State

- Native gated-delta helper is implemented in `packages/core/native/mlxts_core_ops.cpp` and exposed as `fast.qwenGatedDeltaUpdate`.
- Qwen linear attention uses mixed-dtype native inputs, fp32 recurrent state,
  contiguous conv-cache tails, and the TS recurrence fallback as oracle.
- Qwen full-attention cached prefill now uses causal SDPA markers instead of
  explicit boolean masks for non-window attention, and the model hoists one
  full-attention mask per forward.
- Qwen quantized `b/a` gate projections are fused in eval mode with source-handle
  invalidation so stale fused weights are not reused.
- Long-context benchmark reads nested `text_config.max_position_embeddings`;
  Qwen 3.6 advertises `262144`.

## Latest Evidence

- `1024/128` paired: `mlx-lm generation_tps=28.899`, `mlxts generation_tps=28.999`.
- `10000/128` paired: `mlx-lm generation_tps=27.154`, `mlxts generation_tps=26.959`.
- `1024/1024` paired: `mlx-lm generation_tps=28.448`, `mlxts generation_tps=28.352`.
- `128/10000` local: `generation_tps=27.867`, `active_slope_mb_per_token=0.07`, no crash.
- `32768` long-context local: `peak_after_decode=25.995 GB`, `active_decode_slope_mb_per_token=0.00`, marker was the first generated line after disabling thinking.

## Next Work

- Run higher context rungs when machine time permits: `65536`, `131072`, then
  `262144` if memory/thermals allow.
- Consider a `20k` generated-token local stress only after the repo is committed;
  `10k` already indicates the old crash class is gone.
- Remaining Qwen gap is mostly peak memory versus `mlx-lm` and small paired-run
  variance. Next investigation should profile full-attention KV representation,
  cache-buffer accounting, and wrapper/FFI overhead rather than scattering
  micro-optimizations.
- After Qwen serving quality is committed, resume Responses API completion work,
  Anthropic API, and then Qwen/Gemma MoE plus multimodal capability.
