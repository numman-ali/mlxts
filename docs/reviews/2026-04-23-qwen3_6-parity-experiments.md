# Qwen 3.6 Parity Experiments

## Goal

Close the Qwen 3.6 27B decode gap against `mlx-lm` with the same research loop
used for Gemma 4: start from honest paired measurements, change one thing at a
time, keep only measured wins, and record rejected ideas instead of relying on
memory.

The target is not merely to look comparable to `mlx-lm`; the target is to make
the benchmark true first, then make the implementation faster than the upstream
reference without weakening the TypeScript semantic surface.

## Baseline

- Baseline commit: `80fcb2f` (`Add serve info endpoint`)
- Model: `mlx-community/Qwen3.6-27B-4bit`
- Command:
  `bun run bench:generation:parity --model mlx-community/Qwen3.6-27B-4bit --prompt-tokens 128 --generation-tokens 128 --trials 1 --memory-sample-interval 16 --mlx-lm-python .tmp/venvs/mlx-lm-bench/bin/python`
- `mlx-lm`: `prompt_tps=48.235`, `generation_tps=28.662`, `peak_memory=15.681 GB`
- `mlxts`: `prompt_tps=102.596`, `generation_tps=24.245`, `peak_memory=19.161 GB`
- Gap: `-4.417 tok/s` decode, about `-15.4%`
- `mlxts` active decode slope: effectively flat at `-0.97 MB/token`

## Summary

The first conclusion is methodological: the reported decode gap should be treated
as real, but the harness had several fairness issues that made the comparison
too easy to argue with. The first keeper therefore fixes the paired benchmark
posture before touching the model hot path.

After the first keeper, the fixed three-trial paired run still shows a real
decode gap: `mlx-lm=29.024 tok/s`, `mlxts=24.898 tok/s`, a `-14.2%` decode
deficit. That makes the next work a model/runtime hot-path investigation rather
than a benchmark-accounting issue.

The working implementation hypothesis was confirmed. Apple `mlx-lm` uses a
fused Metal `gated_delta_update` kernel for Qwen 3.5 / 3.6-style linear layers,
while the earlier `mlxts` path expressed the recurrence as many ordinary MLX ops
in TypeScript. A narrow native helper, mixed-dtype inputs, contiguous conv-cache
tails, Qwen full-attention causal mask parity, and fused quantized `b/a`
projections close the practical gap on the staged rungs without weakening the
semantic model surface.

After the final keeper tranche, staged evidence is materially stronger than the
original `128/128` smoke:

- `1024/128` paired: `mlxts` slightly ahead on decode (`28.999` vs
  `28.899 tok/s`) with near-identical prompt throughput.
- `10000/128` paired: `mlxts=26.959`, `mlx-lm=27.154`, about `0.7%` under.
- `1024/1024` paired: `mlxts=28.352`, `mlx-lm=28.448`, about `0.3%` under.
- Local `128/10000` output stress completed without crash at `27.867 tok/s`
  and `0.07 MB/token` active memory slope.
- Local `32k` retrieval prefill completed at `25.995 GB` peak with zero decode
  active-memory slope; the first generated line contained the benchmark marker,
  and the readout now grades that fixed-length output shape correctly.

## Files Reviewed

- `packages/transformers/scripts/benchmark-common.ts`
- `packages/transformers/scripts/benchmark-generation-parity.ts`
- `packages/transformers/scripts/benchmark-mlx-lm.py`
- `packages/transformers/scripts/benchmark-common.test.ts`
- `packages/transformers/scripts/benchmark-long-context.ts`
- `packages/transformers/scripts/benchmark-long-context.test.ts`
- `packages/core/native/mlxts_core_ops.cpp`
- `packages/core/src/fast.ts`
- `packages/core/src/ffi/symbols.ts`
- `packages/core/src/index.ts`
- `packages/core/src/ops/index.ts`
- `packages/core/src/ops/shape.ts`
- `packages/core/src/quantization.ts`
- `packages/nn/src/index.ts`
- `packages/nn/src/quantized-linear.ts`
- `packages/nn/src/quantized-linear.test.ts`
- `packages/transformers/src/families/qwen3_5/gated-delta.ts`
- `packages/transformers/src/families/qwen3_5/gated-delta-recurrence.ts`
- `packages/transformers/src/families/qwen3_5/gated-delta.test.ts`
- `packages/transformers/src/families/qwen3_5/attention.ts`
- `packages/transformers/src/families/qwen3_5/model.ts`
- `packages/transformers/src/infrastructure/masks.ts`
- `packages/transformers/src/infrastructure/masks.test.ts`
- `packages/transformers/src/infrastructure/cache/runtime.ts`
- `packages/core/src/fast.test.ts`
- `packages/core/src/ops/ops.test.ts`
- `scripts/runtime-sensitive-ops.ts`
- `.reference/mlx-lm/mlx_lm/models/qwen3_5.py`
- `.reference/mlx-lm/mlx_lm/models/gated_delta.py`
- `.reference/mlx-lm/mlx_lm/models/cache.py`
- `.reference/mlx-lm/mlx_lm/models/base.py`
- `.reference/mlx-lm/mlx_lm/generate.py`
- `.reference/mlx-lm/mlx_lm/benchmark.py`

## Tensor Lifetime Audit

Experiment 1 changes benchmark scripts and the Python reference helper only. It
does not alter model tensor ownership, cache lifetime, generated logits,
sampling, or runtime resource ownership in production source.

Experiment 3 wraps the existing Qwen decay-factor math in a disposable compiled
transform. The transform returns a fresh owned `MxArray` exactly where
`decayFactors()` already returned one, and all local tensor-producing
intermediates inside the transform remain visible with `using` declarations.
The recurrent state ownership path in `gatedDeltaSequence()` is unchanged.

Experiment 5 adds a narrow native Qwen gated-delta helper. The C++ helper returns
owned MLX arrays through per-call output slots and the TypeScript wrapper
immediately wraps both outputs into explicit `MxArray` handles. The fallback TS
recurrence remains as the oracle path. The helper keeps recurrent state ownership
unchanged: callers free returned `output` and `state` in the same `try/finally`
block as before.

Experiment 6 exposes `contiguous()` as a core tensor-producing op and uses it
only for the Qwen conv-cache tail. The intermediate slice view and contiguous
copy are both locally visible; the cache update retains ownership and the local
copy is freed immediately after cache assignment.

Experiment 7 hoists the Qwen full-attention mask to one model-forward-owned
value. Full-attention layers retain the mask internally when it is an `MxArray`,
and the model frees the hoisted owner once after the layer loop. Non-window
cached prefill now uses the `"causal"` marker, so most hoisted masks are not
arrays at all.

Experiment 8 fuses compatible quantized `b/a` projections by concatenating
packed quantized weight rows and auxiliary rows. The fused helper is private to
the Qwen layer, source-handle keyed, eval-only, and disposed with the layer. If
the source quantized modules change, the fused helper is rebuilt rather than
silently reusing stale weights.

## Memory / Performance Evidence

- `bench:generation:parity` baseline above showed flat `mlxts` active memory
  slope but a real-looking decode deficit.
- `bench:generation` was not rerun for the first keeper because it changes the
  paired reference harness, not the local model path.
- Two independent explorer audits agreed that harness issues exist, but do not
  explain away the `24.245` vs `28.662 tok/s` gap.
- The first required measurement was the fixed harness on the same Qwen shape
  with at least one warmup and multiple measured trials on both sides.
- Fixed-harness Qwen run:
  `bun run bench:generation:parity --model mlx-community/Qwen3.6-27B-4bit --prompt-tokens 128 --generation-tokens 128 --trials 3 --memory-sample-interval 16 --mlx-lm-python .tmp/venvs/mlx-lm-bench/bin/python`
  - `mlx-lm`: `prompt_tps=214.065`, `generation_tps=29.024`,
    `peak_memory=15.682 GB`
  - `mlxts`: `prompt_tps=105.291`, `generation_tps=24.898`,
    `peak_memory=19.161 GB`
  - `mlxts` active memory stayed flat: `active_slope_mb_per_token=-0.97`
- Compiled decay-factor Qwen run:
  `bun run bench:generation:parity --model mlx-community/Qwen3.6-27B-4bit --prompt-tokens 128 --generation-tokens 128 --trials 3 --memory-sample-interval 16 --mlx-lm-python .tmp/venvs/mlx-lm-bench/bin/python`
  - `mlx-lm`: `generation_tps=29.291`
  - `mlxts`: `generation_tps=25.945`
  - improvement over fixed-harness baseline: `+1.047 tok/s`, about `+4.2%`
- Local synthetic confirmation:
  `bun run bench:generation --model mlx-community/Qwen3.6-27B-4bit --prompt-tokens 128 --generation-tokens 128 --trials 3 --memory-sample-interval 16`
  - `mlxts`: `generation_tps=25.814`, `peak_memory=19.177 GB`
  - active memory stayed flat: `active_slope_mb_per_token=-0.00`
- Native gated-delta local confirmation:
  `bun run bench:generation:parity --model mlx-community/Qwen3.6-27B-4bit --prompt-tokens 128 --generation-tokens 128 --trials 3 --memory-sample-interval 16 --skip-mlx-lm-reference`
  - `mlxts`: `generation_tps=28.641`, `peak_memory=18.515 GB`
  - active memory stayed flat.
- Native gated-delta paired confirmation:
  `bun run bench:generation:parity --model mlx-community/Qwen3.6-27B-4bit --prompt-tokens 128 --generation-tokens 128 --trials 3 --memory-sample-interval 16 --mlx-lm-python .tmp/venvs/mlx-lm-bench/bin/python`
  - `mlx-lm`: `generation_tps=29.266`
  - `mlxts`: `generation_tps=28.192`
- Mixed-dtype native path confirmation:
  `bun run bench:generation:parity --model mlx-community/Qwen3.6-27B-4bit --prompt-tokens 5000 --generation-tokens 128 --trials 1 --memory-sample-interval 16 --mlx-lm-python .tmp/venvs/mlx-lm-bench/bin/python`
  - `mlx-lm`: `generation_tps=27.129`
  - `mlxts`: `generation_tps=25.502`
  - improved over the float32-input native path on the same long-prompt rung.
- Full-attention causal-mask and conv-cache-tail tranche:
  `bun run bench:generation:parity --model mlx-community/Qwen3.6-27B-4bit --prompt-tokens 5000 --generation-tokens 128 --trials 1 --memory-sample-interval 16 --skip-mlx-lm-reference`
  - `mlxts`: `generation_tps=27.700`, `peak_memory=21.284 GB`
  - peak dropped from the earlier `21.604 GB` rung after avoiding explicit
    cached-prefill boolean masks.
- Quantized `b/a` projection fusion profile:
  `MLXTS_RUNTIME_PROFILE=1 bun run bench:generation:parity --model mlx-community/Qwen3.6-27B-4bit --prompt-tokens 5000 --generation-tokens 32 --trials 1 --memory-sample-interval 8 --skip-mlx-lm-reference`
  - `generation_tps=29.180`
  - `quantizedMatmul` calls dropped from `14910.0/trial` to `13470.0/trial`
  - `ffi_ms_per_token` dropped from about `1.2857-1.3740` to `1.2680`
- Staged prompt ladder:
  `bun run bench:generation:parity --model mlx-community/Qwen3.6-27B-4bit --prompt-tokens 1024 --generation-tokens 128 --trials 1 --memory-sample-interval 16 --mlx-lm-python .tmp/venvs/mlx-lm-bench/bin/python`
  - `mlx-lm`: `prompt_tps=250.024`, `generation_tps=28.899`
  - `mlxts`: `prompt_tps=249.030`, `generation_tps=28.999`
  `bun run bench:generation:parity --model mlx-community/Qwen3.6-27B-4bit --prompt-tokens 10000 --generation-tokens 128 --trials 1 --memory-sample-interval 16 --mlx-lm-python .tmp/venvs/mlx-lm-bench/bin/python`
  - `mlx-lm`: `prompt_tps=223.768`, `generation_tps=27.154`
  - `mlxts`: `prompt_tps=215.891`, `generation_tps=26.959`
- Staged output ladder:
  `bun run bench:generation:parity --model mlx-community/Qwen3.6-27B-4bit --prompt-tokens 1024 --generation-tokens 512 --trials 1 --memory-sample-interval 32 --mlx-lm-python .tmp/venvs/mlx-lm-bench/bin/python`
  - `mlx-lm`: `generation_tps=28.396`
  - `mlxts`: `generation_tps=28.346`, `active_slope_mb_per_token=0.07`
  `bun run bench:generation:parity --model mlx-community/Qwen3.6-27B-4bit --prompt-tokens 1024 --generation-tokens 1024 --trials 1 --memory-sample-interval 64 --mlx-lm-python .tmp/venvs/mlx-lm-bench/bin/python`
  - `mlx-lm`: `generation_tps=28.448`
  - `mlxts`: `generation_tps=28.352`, `active_slope_mb_per_token=0.07`
- Long-output local stress:
  `bun run bench:generation:parity --model mlx-community/Qwen3.6-27B-4bit --prompt-tokens 128 --generation-tokens 10000 --trials 1 --memory-sample-interval 512 --skip-mlx-lm-reference`
  - `generation_tps=27.867`, `peak_memory=18.870 GB`
  - `active_delta=0.655 GB`, `active_slope_mb_per_token=0.07`
  `bun run bench:generation:parity --model mlx-community/Qwen3.6-27B-4bit --prompt-tokens 128 --generation-tokens 20000 --trials 1 --memory-sample-interval 1024 --skip-mlx-lm-reference`
  - `generation_tps=27.076`, `peak_memory=19.569 GB`
  - `active_delta=1.309 GB`, `active_slope_mb_per_token=0.07`
- Long-context retrieval:
  `bun run bench:generation:context --model mlx-community/Qwen3.6-27B-4bit --rungs 32768 --generation-tokens 64 --prefill-step-size 2048`
  - `prompt_tokens=32771`, `prefill_tps=214.231`, `peak_after_decode=25.995 GB`
  - `active_decode_slope_mb_per_token=0.00`
  - first generated answer line was the benchmark marker; exact-match
    normalization was updated to grade this fixed-length decode pattern.
  `bun run bench:generation:context --model mlx-community/Qwen3.6-27B-4bit --rungs 65536 --generation-tokens 64 --prefill-step-size 2048`
  - `prompt_tokens=65546`, `prefill_tps=187.497`, `decode_tps=19.522`
  - `peak_after_decode=31.410 GB`, `active_decode_slope_mb_per_token=0.00`
  - `exact_match=true`, `contains_secret=true`
  `bun run bench:generation:context --model mlx-community/Qwen3.6-27B-4bit --rungs 131072 --generation-tokens 64 --prefill-step-size 2048`
  - `prompt_tokens=131078`, `prefill_tps=150.523`, `decode_tps=16.019`
  - `peak_after_decode=42.550 GB`, `active_decode_slope_mb_per_token=0.00`
  - `exact_match=true`, `contains_secret=true`

## Independent Review

Six read-only sub-agents were used during this loop. Volta audited Qwen's model
hot path against `.reference/mlx-lm` and ranked the fused gated-delta kernel gap
as the highest-confidence cause. Newton audited benchmark fairness and found
reference isolation, matching trial protocol, memory definitions, prefill-step
forwarding, and ignored diagnostic flags as the issues to fix before optimizing
model code. Lorentz and Goodall audited native/Metal-kernel options and
recommended a narrow Qwen gated-delta helper before exposing a generic
`metalKernel` surface. Confucius identified the mixed-dtype native helper gap
and conv-cache-tail contiguity as the next candidates. Lovelace identified the
cached-prefill causal-mask mismatch and per-layer mask rebuild versus `mlx-lm`.
Meitner audited the benchmark matrix and recommended using the existing parity,
synthetic, and long-context scripts rather than creating another harness.

## Remaining Risks / Follow-ups

The remaining gap is no longer a catastrophic correctness/performance class; it
is mostly memory footprint versus `mlx-lm` and small paired-run throughput
variance. `mlxts` still reports higher peak memory than `mlx-lm` on paired rungs
because cache buffers and JS/FFI-owned wrapper state remain different. Further
work should profile full-attention KV representation, cache-buffer accounting,
and lower FFI wrapper overhead before chasing micro-ops.

The 32k, 64k, and 128k long-context runs passed marker retrieval with zero
active decode slope. The 128k rung peaked at `42.550 GB`, so 262k remains an
advertised-model-capability target but should be gated behind serving admission
and memory preflight before local live testing. The benchmark script now reads
nested `text_config.max_position_embeddings`, so it knows Qwen advertises
`262144` context tokens.

## Experiment Log

### Experiment 1: Make the paired Qwen benchmark defensible before optimizing

- Status: `kept`
- Hypothesis:
  The existing parity harness is close enough to show a real Qwen gap, but it is
  not rigorous enough for keeper decisions because the `mlx-lm` helper runs
  while the Bun model is resident, uses a different warmup/trial protocol, does
  not receive `prefill_step_size`, and ignores some parsed diagnostic flags on
  the `mlxts` side.
- Success criteria:
  - `mlx-lm` reference capture happens before the Bun model is loaded
  - both sides use one warmup and the same measured trial count
  - `prefill_step_size` is forwarded to the Python helper
  - parity benchmark output states decode schedule and cache materialization
  - `--sync-decode` and `--materialize-cache-each-token` affect parity runs
  - focused tests and typecheck pass
- Rollback rule:
  if the patch cannot run the existing benchmark/tests cleanly, remove it before
  changing model code

#### Outcome

- Result:
  kept
- Measurement:
  fixed-harness Qwen run above confirmed the decode gap remains real:
  `mlx-lm=29.024 tok/s`, `mlxts=24.898 tok/s`
- Conclusion:
  the first keeper makes the benchmark harder to dismiss, but it does not close
  the gap. The next experiment should focus on Qwen's gated-delta linear
  attention path.

### Experiment 2: Add a single-token gated-delta sequence fast path

- Status: `failed and reverted`
- Hypothesis:
  Qwen decode calls `gatedDeltaSequence()` with `sequenceLength === 1` for every
  linear-attention layer. The generic sequence loop still slices five tensors and
  stacks one output. A single-token path can preserve the same recurrent math
  while replacing slice/stack work with direct reshapes.
- Success criteria:
  - output/state shapes and simple math remain covered by unit tests
  - tensor ownership stays visible and explicit
  - `bench:generation:parity` improves or at least does not regress under the
    fixed Qwen harness
- Rollback rule:
  if Qwen parity decode does not measurably improve, revert this path before
  moving to deeper compile/native work

#### Outcome

- Result:
  rejected
- Measurement:
  fixed-harness Qwen run after the fast path:
  `mlx-lm=29.307 tok/s`, `mlxts=24.964 tok/s`
- Conclusion:
  the change moved `mlxts` from `24.898` to `24.964 tok/s`, which is noise at
  this benchmark size. The experiment was removed before moving on. The gap is
  not primarily the generic single-token slice/stack wrapper; it remains in the
  deeper gated-delta compute stage.

### Experiment 3: Compile Qwen decay-factor computation

- Status: `kept`
- Hypothesis:
  `mlx-lm` compiles `compute_g(A_log, a, dt_bias)` before the gated-delta kernel.
  `mlxts` currently rebuilds the same softplus/exp chain as ordinary MLX ops.
  Compiling this helper may reduce per-layer `astype`, `exp`, `log`, `where`,
  and `multiply` overhead without changing model semantics.
- Success criteria:
  - focused Qwen gated-delta tests and typecheck pass
  - runtime profile shows the decay-factor op cluster shrink
  - fixed Qwen parity run improves enough to distinguish from noise
- Rollback rule:
  if the paired Qwen benchmark does not materially improve, remove this
  experiment and proceed to the fused gated-delta/native-kernel seam

#### Outcome

- Result:
  kept
- Measurement:
  short profile rose from `25.076` to `26.899 tok/s` and reduced per-token
  `multiply`, `astype`, and `add` counts. The fixed paired run improved from
  `24.898` to `25.945 tok/s`.
- Conclusion:
  this is a real partial win and matches `mlx-lm`'s compile posture for
  `compute_g`, but it does not close the deeper gap.

### Experiment 4: Compile the gated-delta recurrent step

- Status: `failed and reverted`
- Hypothesis:
  `mlx-lm`'s non-Metal fallback compiles the single recurrent gated-delta step.
  Compiling our TypeScript step with `compileMany()` may reduce the remaining
  recurrence plumbing before a native Metal-kernel seam is justified.
- Success criteria:
  - simple recurrent math and lifetime tests pass
  - typecheck passes
  - short profile and fixed paired Qwen run improve enough to keep the change
- Rollback rule:
  if this is neutral or regresses, remove it and proceed to the native/fused
  gated-delta kernel plan

#### Outcome

- Result:
  rejected
- Measurement:
  `bun test packages/transformers/src/families/qwen3_5/gated-delta.test.ts`
  crashed Bun with a segmentation fault before benchmark validation.
- Conclusion:
  compiling the multi-output recurrent step through the current Bun/MLX closure
  path is not safe enough to keep. The experiment was removed immediately. The
  next deep seam should be a deliberate native/Metal kernel binding rather than
  forcing this through `compileMany()`.

### Experiment 5: Add a narrow native Qwen gated-delta helper

- Status: `kept`
- Hypothesis:
  The remaining Qwen gap is dominated by the recurrent gated-delta stage. A
  narrow native helper can mirror `mlx-lm`'s Metal kernel shape while preserving
  the TypeScript recurrence as fallback/oracle.
- Success criteria:
  - native helper tests match the TS recurrence for multi-token and continuation
    cases
  - steady decode remains one eval per token
  - paired Qwen decode improves materially without introducing active-memory
    slope
- Rollback rule:
  if correctness tests fail, Bun crashes, or paired decode does not improve,
  remove the helper before trying broader native surfaces

#### Outcome

- Result:
  kept
- Measurement:
  local `128/128` rose to `28.641 tok/s`; paired `128/128` rose to
  `28.192 tok/s` versus `mlx-lm=29.266`.
- Conclusion:
  the narrow native helper is a real keeper. It closes most of the original
  `24.245 -> 28.662 tok/s` gap without broadening the public core API.

### Experiment 6: Match upstream mixed-dtype gated-delta inputs

- Status: `kept`
- Hypothesis:
  Forcing q/k/v/g/beta to float32 before the native helper adds bandwidth and
  graph nodes that `mlx-lm` avoids. The native helper should accumulate in float
  internally while accepting model-native input dtypes and fp32 state.
- Success criteria:
  - Qwen gated-delta oracle tests pass
  - paired short and long-prompt rungs improve or stay flat
  - output/state dtype behavior remains compatible with upstream
- Rollback rule:
  if token parity changes or the long-prompt rung regresses, restore explicit
  float32 inputs

#### Outcome

- Result:
  kept
- Measurement:
  local `128/128` improved to `28.697 tok/s`; paired `128/128` improved to
  `28.532 tok/s`; paired `5000/128` improved to `25.502 tok/s` from the
  float32 native path.
- Conclusion:
  mixed dtype is the correct upstream-shaped path. Float32 remains only in the
  fallback recurrence and recurrent state.

### Experiment 7: Store the Qwen conv-cache tail contiguously

- Status: `kept`
- Hypothesis:
  `mlx-lm` stores `mx.contiguous(conv_input[:, -n_keep:, :])` for the
  linear-attention conv cache. Keeping a slice view in `mlxts` can make the next
  token's concatenate/conv path less favorable.
- Success criteria:
  - expose a small core `contiguous()` op with coverage
  - Qwen tests pass
  - memory does not regress and local decode improves outside noise
- Rollback rule:
  remove the contiguous copy if long-prompt decode or peak memory regresses

#### Outcome

- Result:
  kept
- Measurement:
  local `128/128` reached `29.074 tok/s`; the profiled `5000/32` rung remained
  flat and stable. The later mask tranche dropped the 5k peak from `21.604 GB`
  to `21.284 GB`.
- Conclusion:
  this is a small upstream-parity improvement and makes the cache state shape
  more explicit.

### Experiment 8: Use causal SDPA for cached full-attention prefill and hoist the mask

- Status: `kept`
- Hypothesis:
  For non-window full attention, `mlx-lm` returns `"causal"` even when cached
  prefill has a non-zero offset. `mlxts` was materializing boolean masks for
  cached prefill chunks and rebuilding them per full-attention layer.
- Success criteria:
  - core SDPA test proves `"causal"` aligns shorter query blocks to the end of
    cached keys
  - Qwen full-attention layers share one model-forward mask owner
  - 5k prompt peak memory improves without decode regression
- Rollback rule:
  restore explicit masks if logits/mask tests fail or long-prompt behavior
  regresses

#### Outcome

- Result:
  kept
- Measurement:
  local `5000/128` peak dropped from `21.604 GB` to `21.284 GB`; paired
  `5000/128` reached `27.304 tok/s` against `mlx-lm=27.840` in one run, with
  later local runs reaching `27.901 tok/s`.
- Conclusion:
  this matches upstream and removes unnecessary mask allocation pressure. It is
  primarily a memory/prefill-quality keeper rather than a decode-only win.

### Experiment 9: Fuse Qwen quantized `b/a` projections

- Status: `kept`
- Hypothesis:
  Qwen linear-attention layers call two tiny quantized gate projections, `b` and
  `a`, that share input shape and quantization parameters. Concatenating packed
  quantized rows can remove one quantized-matmul FFI call per linear-attention
  layer per decode token without changing math.
- Success criteria:
  - package-owned helper proves fused quantized linears equal separate outputs
  - fused helper is private, eval-only, source-handle keyed, and disposable
  - runtime profile shows fewer quantized matmul calls and decode improves
- Rollback rule:
  remove the fusion if it only moves work into slices without improving local
  decode or if stale weight hazards cannot be contained

#### Outcome

- Result:
  kept
- Measurement:
  profiled `5000/32` dropped `quantizedMatmul` calls from `14910.0/trial` to
  `13470.0/trial`, and local decode reached `29.180 tok/s`. Local `5000/128`
  reached `27.901 tok/s`.
- Conclusion:
  this is a targeted TypeScript-side win with tiny additional packed-weight
  memory and no public model-surface change.

### Experiment 10: Make long-context retrieval reporting Qwen-thinking aware

- Status: `kept`
- Hypothesis:
  The long-context retrieval benchmark should measure marker retrieval, not the
  model's default thinking preamble. Qwen chat prompts should disable thinking
  for this benchmark, and fixed-length decode should grade the first generated
  answer line while still printing the full response.
- Success criteria:
  - benchmark tests cover nested context config and first-line exact response
    normalization
  - 32k retrieval run completes without memory slope and finds the marker
- Rollback rule:
  if disabling thinking is not template-compatible or hides retrieval failures,
  remove the default and require explicit benchmark flags instead

#### Outcome

- Result:
  kept
- Measurement:
  `bench:generation:context --rungs 32768 --generation-tokens 64` completed
  with `peak_after_decode=25.995 GB`, `active_decode_slope_mb_per_token=0.00`,
  and the marker as the first generated line.
- Conclusion:
  the 32k rung is usable as a capability check. Higher rungs remain future
  staged work, not a new harness requirement.

### Experiment 11: Extend the staged capability ladder to real long-context and long-output rungs

- Status: `kept`
- Hypothesis:
  The short paired parity rungs are not enough to claim Qwen 3.6 serving
  quality. The same implementation must also survive long-context retrieval and
  long-output decode without active memory slope or marker-retrieval failures.
- Success criteria:
  - 64k and 128k retrieval rungs find the marker
  - decode active memory slope remains effectively zero after long-context
    prefill
  - a 20k generated-token decode stays near the earlier 10k active slope
  - 262k is not brute-forced if 128k peak memory shows the need for admission
    controls first
- Rollback rule:
  if long-context or long-output runs show renewed active-memory growth, return
  to the relevant cache/recurrent-state stage before adding serving API breadth.

#### Outcome

- Result:
  kept
- Measurement:
  `bench:generation:context --rungs 65536 --generation-tokens 64` completed
  with `peak_after_decode=31.410 GB`, `decode_tps=19.522`,
  `active_decode_slope_mb_per_token=0.00`, and `exact_match=true`.

  `bench:generation:context --rungs 131072 --generation-tokens 64` completed
  with `peak_after_decode=42.550 GB`, `decode_tps=16.019`,
  `active_decode_slope_mb_per_token=0.00`, and `exact_match=true`.

  `bench:generation:parity --prompt-tokens 128 --generation-tokens 20000
  --skip-mlx-lm-reference` completed with `generation_tps=27.076`,
  `peak_memory=19.569 GB`, `active_delta=1.309 GB`,
  `active_slope_mb_per_token=0.07`, and `evals_per_token=1.00`.
- Conclusion:
  the old long-decode crash class is contained for 20k generated tokens, and
  Qwen 3.6 retrieves correctly through 128k local context. The next serving
  tranche should focus on request admission, progress telemetry, cancellation,
  and Qwen-aware scheduler/cache work before attempting 262k live local tests.
