# Transformer Package Guidance

## Performance Workflow

- Treat generation performance as a staged research loop. Start with reference
  parity, change one bounded seam, measure, and revert losers quickly.
- Check `.reference/mlx-lm` before changing model hot paths. For Qwen-family
  work, compare against `mlx_lm/models/qwen3_5.py`, `qwen3_next.py`,
  `gated_delta.py`, `cache.py`, and `base.py`.
- Do not claim Qwen capability from `128/128` alone. Use prompt rungs, output
  rungs, and long-context retrieval before promoting a keeper.
- Keep semantic model code readable. Runtime strategy belongs behind helpers
  such as native gated-delta, mask builders, quantized projection helpers, or
  cache utilities.

## Qwen 3.5 / 3.6 Notes

- Native gated-delta is the canonical fast path when Metal and shape constraints
  allow it; the TypeScript recurrence remains the oracle/fallback.
- Non-window cached full-attention prefill should use the `"causal"` SDPA marker,
  not an explicit boolean mask.
- Qwen 3.6 advertises long context through nested
  `text_config.max_position_embeddings`, currently `262144` for the tested
  checkpoint.
- Long-context retrieval benchmarks should disable Qwen thinking and grade the
  first non-empty generated answer line while still printing the full response.
