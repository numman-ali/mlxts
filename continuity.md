# Continuity

This file is a compact handoff for long-running `mlxts` work. Keep durable
doctrine in `AGENTS.md`, durable learnings in `MEMORY.md`, and use this file for
current-phase state that should survive context compaction.

## Current Focus

Qwen 3.6 27B serving/inference quality is the active critical path. The goal is
not just to finish a short benchmark; it is to keep staged parity evidence
against `mlx-lm`, long-output stability, and long-context capability visible.
The Responses API text slice is now part of that serving-quality push: it should
work as a usable text endpoint while benchmark and scheduler work continues.

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

- Fresh post-harness serving evidence is recorded in
  `docs/reviews/2026-04-24-qwen-serve-benchmark-ladder.md`. Qwen 3.6 endpoint
  serving now has JSON-backed ladders for `128/128`, `1024/512`, `5000/128`,
  `10000/128`, `1024/1024`, `1024/2048`, `32768/128`, `65536/128`, and
  `131072/128`.
- Qwen 3.6 can serve `131072/128` streaming without crashing on the 64 GB local
  machine: `wall_ms=865021.6`, `mean_ttft_ms=858549.9`,
  `mean_post_ttft_completion_tps=19.624`, `peak_memory=42.543 GB`,
  `active_delta=0.000 GB`, `finish_reasons=length`. This is capable, but not
  yet product-grade TTFT.
- Fresh required-reference parity: `1024/128` was `mlx-lm=29.135 tok/s`,
  `mlxts=29.236 tok/s`; `10000/128` was `mlx-lm=27.332`, `mlxts=27.241`;
  `1024/1024` was `mlx-lm=28.965`, `mlxts=28.537`. Decode parity is strong;
  the remaining clear gap is peak memory versus `mlx-lm`, especially at 1k
  context.
- Gemma 4 E2B endpoint control passed through `5000/128` with flat active
  memory and post-TTFT decode around `82-88 tok/s`. LLaMA 3.2 1B continuous
  batching control showed `continuous_admission_rows=4` and
  `max_generation_batch=4` at `16x16@4`, while Qwen queued concurrency stayed
  serialized with zero real batch rows as expected.
- `1024/128` paired: `mlx-lm generation_tps=28.899`, `mlxts generation_tps=28.999`.
- `10000/128` paired: `mlx-lm generation_tps=27.154`, `mlxts generation_tps=26.959`.
- `1024/1024` paired: `mlx-lm generation_tps=28.448`, `mlxts generation_tps=28.352`.
- `128/10000` local: `generation_tps=27.867`, `active_slope_mb_per_token=0.07`, no crash.
- `128/20000` local: `generation_tps=27.076`, `peak_memory=19.569 GB`, `active_slope_mb_per_token=0.07`, no crash.
- `32768` long-context local: `peak_after_decode=25.995 GB`, `active_decode_slope_mb_per_token=0.00`, marker was the first generated line after disabling thinking.
- `65536` long-context local: `peak_after_decode=31.410 GB`, `decode_tps=19.522`, `active_decode_slope_mb_per_token=0.00`, exact marker match.
- `131072` long-context local: `peak_after_decode=42.550 GB`, `decode_tps=16.019`, `active_decode_slope_mb_per_token=0.00`, exact marker match.
- `32768` all-needle long-context local: early, middle, and late markers all exact-matched with prompt tokens `32774`, marker center fractions `0.101`, `0.500`, and `0.999`, peak `25.995 GB`, and flat active decode memory slope. Report:
  `.tmp/qwen36-context-32k-all-needles.json`.
- Tiny live serve endpoint probe on cached `mlx-community/Llama-3.2-1B-Instruct-4bit`
  passed through `/v1/completions`: prompt `16`, generation `4`, concurrency
  `1,2`, greedy, no warmup. The concurrency-2 rung reported
  `admission_batches=1` and `static_batches=1`, proving the new endpoint
  benchmark can separate admission coalescing from real static batch execution.
- Tiny live SSE serve endpoint probe on the same cached Llama model passed with
  `--stream`: prompt `16`, generation `4`, concurrency `1`, greedy, no warmup.
  It reported `mean_ttft_ms=51.1`, `stream_chunks=1`, `stream_bytes=676`, and
  full usage (`prompt_tokens=16`, `completion_tokens=4`), proving the benchmark
  can measure real `/v1/completions` streaming rather than only buffered HTTP.
- Live Qwen serve endpoint exact-length probe passed with `--ignore-eos`: prompt
  `128`, generation `128`, concurrency `1`, greedy, no warmup reported
  `completion_tps=25.373`, `completion_tokens=128`, `finish_reasons=length`,
  `peak_memory=18.481 GB`. The previously misleading `1024/128` rung now
  generated all `128` tokens instead of stopping at EOS after 4 tokens:
  `completion_tps=15.116`, `peak_memory=19.934 GB`, `active_delta=0.000 GB`.
- Qwen endpoint prompt ladder with exact-length greedy completions is stable and
  prefill-dominated as expected: `128/128` averaged `25.697` end-to-end
  completion tok/s, `1024/128` `14.162`, `5000/128` `5.031`, and `10000/128`
  `2.698`, with `active_delta=0.000 GB` on all rungs. These are not raw decode
  numbers; they include full prefill wall time.
- Qwen endpoint output ladder at `1024` prompt tokens shows the expected
  amortization toward decode speed: `128` output averaged `14.835` end-to-end
  completion tok/s, `512` output `22.797`, and `1024` output `25.540`, with flat
  active memory.
- Qwen streaming endpoint ladder exposed a real flush-cadence bug before the
  latest fix: `1024/512` emitted `64` chunks but `mean_ttft_ms=21859.9` against
  `wall_ms=21860.1`, and `1024/1024` emitted `128` chunks but
  `mean_ttft_ms=39652.2` against `wall_ms=39652.4`. The serving bridge now
  yields a macrotask after processed stream events. The post-fix multi-trial
  streaming ladder at `1024` prompt tokens reports useful TTFT/decode splits:
  `512` output averaged `mean_ttft_ms=5197.3`,
  `mean_prompt_to_first_token_tps=197.403`, and
  `mean_post_ttft_completion_tps=29.080`; `1024` output averaged
  `mean_ttft_ms=4937.8`, `mean_prompt_to_first_token_tps=207.425`, and
  `mean_post_ttft_completion_tps=29.006`.
- Gemma 4 dense endpoint prompt ladder on cached `google/gemma-4-E2B-it` is
  healthy through `32k` prompt tokens with flat active memory: `128/128`
  averaged `81.865` end-to-end completion tok/s with `9.446 GB` peak,
  `1024/128` `75.893` with `9.892 GB`, `10000/128` `44.885` with `10.351 GB`,
  and `32000/128` `19.524` with `11.192 GB`.
- Gemma 4 dense endpoint output ladder at `1024` prompt tokens is stable through
  `10000` generated tokens: `128` output reported `76.088` completion tok/s,
  `1000` output `80.590`, and `10000` output `77.984`, with peak memory holding
  around `9.892 GB` and `active_delta=0.000 GB`.
- Gemma 4 dense streaming endpoint ladder also validates the SSE flush fix:
  `1024/512` averaged `mean_ttft_ms=215.8`,
  `mean_prompt_to_first_token_tps=4744.886`, and
  `mean_post_ttft_completion_tps=82.605`; `1024/1024` averaged
  `mean_ttft_ms=217.0`, `mean_prompt_to_first_token_tps=4719.600`, and
  `mean_post_ttft_completion_tps=81.727`, with flat active memory.
- Serialized endpoint concurrency with `maxConcurrentRequests=1` was stable but
  not real batching. Qwen `1024/128` at concurrency `1,2,4` completed with
  `15.050`, `13.701`, and `14.151` aggregate completion tok/s respectively,
  `admission_batches=1` for queued rungs, `static_batches=0`, and flat active
  memory. Gemma 4 E2B `1024/128` at concurrency `1,2,4` completed with
  `75.774`, `75.710`, and `75.499` aggregate completion tok/s, also with
  `static_batches=0`.
- The first real continuous batching tranche is implemented for the full-KV
  greedy safe subset. `ContinuousBatchTokenScheduler` owns waiting/active rows,
  can admit rows between decode steps, filters mixed-length rows, removes active
  aborted rows, and is endpoint-visible through `serveLoadedModel()`. The serve
  benchmark now reports `continuous_admissions` separately from `static_batches`
  and `admission_batches`.
- Tiny live LLaMA 1B endpoint probe after the scheduler tranche:
  `prompt_tokens=16`, `generation_tokens=16`, concurrency `1,2`, greedy,
  no warmup. Concurrency 1 reported `completion_tps=127.140`,
  `continuous_admissions=1`, `static_batches=0`, `admission_batches=0`;
  concurrency 2 reported `completion_tps=261.027`, `continuous_admissions=1`,
  `static_batches=0`, `admission_batches=0`, with flat active memory.
- Qwen endpoint long-context serving needs a streaming distinction in the
  evidence ledger. Buffered `65536/128` requests repeatedly hit a several-minute
  client/HTTP timeout before returning JSON, even after server-side timeout
  controls. The streaming path was fixed with initial SSE keepalive, periodic
  heartbeat comments, and cooperative streaming prefill that yields between
  chunks. Post-fix `65536/128` streaming completed in `351716.2ms` with
  `mean_ttft_ms=346501.8`, `mean_prompt_to_first_token_tps=189.136`,
  `mean_post_ttft_completion_tps=24.356`, `peak_memory=31.406 GB`,
  `cache_memory=4.467 GB`, and `active_delta=0.013 GB`.
- `/v1/responses` text support now accepts string or text-only message-array
  input, supports semantic SSE streaming with text/reasoning deltas, preserves
  model-native sampling defaults, and keeps tools/state/multimodal explicitly
  rejected. Focused tests and typecheck pass; see
  `docs/reviews/2026-04-24-serve-responses-text-streaming.md`.
- `bench:serve` now has protocol modes. `--protocol completions` remains the
  exact token-array throughput path; `--protocol chat` and
  `--protocol responses` exercise the real wire adapters with deterministic text
  prompts. Live LLaMA 3.2 1B smokes passed for streaming chat and streaming
  Responses after the harness learned to fail streams that end without usage or
  a finish reason. See
  `docs/reviews/2026-04-24-serve-protocol-benchmarking.md`.

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
- Next serving-quality tranche should run staged endpoint ladders with the new
  `continuous_admissions` metric visible, then compare LLaMA-like full-KV
  concurrency against the old static/admission behavior. Admission micro-batching
  is no longer the claim for loaded-model serving; the scheduler-owned path is.
- The next scheduler slices are cache-semantics work, not HTTP wrappers: chunked
  prefill fairness, streaming collectors, sampled batch decode, then Qwen hybrid
  recurrent/full-attention caches and Gemma sliding/global caches. Keep each
  tranche benchmarked separately before claiming broad serving concurrency.
- Use `bun run bench:serve` for endpoint-level serving ladders. It defaults to
  cached/local-only checkpoints, sends exact token-array prompts through
  `/v1/completions`, preserves model-native sampling unless `--greedy` is set,
  and reports request/completion throughput, memory, finish reasons, admission
  batch events, and real static batch events. Add `--stream` for SSE runs; the
  harness requests usage chunks and reports mean TTFT, stream chunk count, and
  streamed bytes. Very small generation lengths may flush as a single text chunk,
  so streaming/backpressure ladders need larger output rungs.
- For huge Qwen prompt rungs, prefer `--stream`; buffered JSON can be a poor
  acceptance shape because clients may wait several minutes with no bytes before
  first token. Streaming now keeps the connection alive during chunked prefill.
- Recommended endpoint ladder order: Qwen `128,1024,5000,10000` prompt tokens at
  `128` output tokens, Qwen output rungs `128,512,1024`, then Gemma dense
  endpoint rungs, then serialized concurrency `1,2,4`, then streaming rungs with
  larger outputs. Use `--max-concurrent-requests 1` first to prove queueing
  stability before any scheduler-backed concurrency claims.
- Prefer explicit staggered endpoint rungs for overnight runs:
  `--rungs 128x128@1,1024x512@1,5000x128@2,10000x128@2` plus
  `--report-json .tmp/<model>-serve-ladder.json`. The report now includes
  p95/max request latency, batch row counters, and max observed generation batch
  size so concurrency is not confused with real batching.
- For publishable mlx-lm parity claims, use `bench:generation:parity
  --require-mlx-lm-reference`; otherwise a missing Python reference is just a
  local-only profiling run. The parity comparison now warns when peak memory is
  materially above mlx-lm, not only when decode TPS trails.
- For broad context-window recall claims, use `bench:generation:context
  --needle-placements all --report-json .tmp/<model>-context.json`. The report
  records actual prompt length plus marker token span/center and is rewritten
  after each placement so partial long-run evidence survives later failures.
- Use `--ignore-eos` only for exact-length throughput/parity ladders. Normal
  serving should still honor EOS, but Qwen/Gemma parity claims need a way to
  request the full output rung when the benchmark is measuring decode speed
  rather than chat stopping behavior.
