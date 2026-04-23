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

The working implementation hypothesis is Qwen's linear-attention stage. Apple
`mlx-lm` uses a fused Metal `gated_delta_update` kernel for Qwen 3.5 / 3.6-style
linear layers, while `mlxts` currently expresses the recurrence as many ordinary
MLX ops in TypeScript. Full-attention masking and KV growth are not the leading
suspects after the Qwen memory work.

## Files Reviewed

- `packages/transformers/scripts/benchmark-common.ts`
- `packages/transformers/scripts/benchmark-generation-parity.ts`
- `packages/transformers/scripts/benchmark-mlx-lm.py`
- `packages/transformers/scripts/benchmark-common.test.ts`
- `packages/transformers/src/families/qwen3_5/gated-delta.ts`
- `packages/transformers/src/families/qwen3_5/gated-delta.test.ts`
- `packages/transformers/src/families/qwen3_5/attention.ts`
- `packages/transformers/src/infrastructure/masks.ts`
- `packages/transformers/src/infrastructure/cache/runtime.ts`
- `.reference/mlx-lm/mlx_lm/models/qwen3_5.py`
- `.reference/mlx-lm/mlx_lm/models/gated_delta.py`
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

## Independent Review

Three read-only sub-agents were used during this loop. Volta audited Qwen's
model hot path against `.reference/mlx-lm` and ranked the fused gated-delta
kernel gap as the highest-confidence cause. Newton audited benchmark fairness
and found reference isolation, matching trial protocol, memory definitions,
prefill-step forwarding, and ignored diagnostic flags as the issues to fix
before optimizing model code. Lorentz audited native/Metal-kernel options and
recommended a narrow Qwen gated-delta helper before exposing a generic
`metalKernel` surface.

## Remaining Risks / Follow-ups

The leading risk is mistaking a fairer benchmark for a faster model. The first
keeper should only make the comparison more defensible; it is not expected to
close the Qwen speed gap by itself.

The likely performance path is now the deeper fused gated-delta seam. A generic
custom Metal kernel binding would be strategically useful later, but the next
measured tranche should be a narrow Qwen helper that keeps the current
TypeScript recurrence as fallback/oracle.

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
