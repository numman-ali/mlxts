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
- `128/20000` local: `generation_tps=27.076`, `peak_memory=19.569 GB`, `active_slope_mb_per_token=0.07`, no crash.
- `32768` long-context local: `peak_after_decode=25.995 GB`, `active_decode_slope_mb_per_token=0.00`, marker was the first generated line after disabling thinking.
- `65536` long-context local: `peak_after_decode=31.410 GB`, `decode_tps=19.522`, `active_decode_slope_mb_per_token=0.00`, exact marker match.
- `131072` long-context local: `peak_after_decode=42.550 GB`, `decode_tps=16.019`, `active_decode_slope_mb_per_token=0.00`, exact marker match.
- Tiny live serve endpoint probe on cached `mlx-community/Llama-3.2-1B-Instruct-4bit`
  passed through `/v1/completions`: prompt `16`, generation `4`, concurrency
  `1,2`, greedy, no warmup. The concurrency-2 rung reported
  `admission_batches=1` and `static_batches=1`, proving the new endpoint
  benchmark can separate admission coalescing from real static batch execution.

## Next Work

- Do not brute-force `262144` on this laptop until serving has real memory
  preflight. `131072` already peaks at `42.550 GB`.
- Serving now has prefill progress telemetry, explicit prompt-token admission,
  best-effort MLX memory preflight, and cooperative cancellation propagated from
  HTTP/SSE/server shutdown into transformer generation. Use
  `--max-prompt-tokens`, `--max-total-tokens`, `--gpu-memory-utilization`, and
  `/info` admission metadata to make long-context tests deliberate rather than
  accidental.
- Remaining Qwen gap is mostly peak memory versus `mlx-lm` and small paired-run
  variance. Next investigation should profile full-attention KV representation,
  cache-buffer accounting, and wrapper/FFI overhead rather than scattering
  micro-optimizations.
- Next serving-quality tranche should prioritize true pull-driven streaming
  backpressure, Qwen-aware scheduler/cache support, per-row batch cancellation,
  and memory-pool safeguards. After that, resume Responses API completion work,
  Anthropic API, and then Qwen/Gemma MoE plus multimodal capability.
- Use `bun run bench:serve` for endpoint-level serving ladders. It defaults to
  cached/local-only checkpoints, sends exact token-array prompts through
  `/v1/completions`, preserves model-native sampling unless `--greedy` is set,
  and reports request/completion throughput, memory, finish reasons, admission
  batch events, and real static batch events.
