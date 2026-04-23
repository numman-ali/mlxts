# Runtime Review: Qwen 3.6 Decode Memory

## Summary

This review covers the real crash-class memory failure seen while serving
`mlx-community/Qwen3.6-27B-4bit` through the local chat/agent loop. The issue
was reproduced with the existing generation benchmark suite, isolated away from
the HTTP server, and fixed in Qwen's recurrent linear-attention and decoder
block ownership paths.

The important outcome is that the failure is not broad evidence that every
model leaks during decode. Llama, Gemma 3, and Gemma 4 stayed flat under the
same benchmark shape. Qwen 3.6 showed a runaway active-memory slope before the
fix because each decode step leaked one large recurrent state set from its
48 linear-attention layers. A second pass then rejected the remaining
`~4 MB/token` slope as still wrong and removed smaller hidden Qwen-only
intermediate leaks until the same benchmark window measured flat.

## Files Reviewed

- `packages/transformers/src/families/qwen3_5/gated-delta.ts`
- `packages/transformers/src/families/qwen3_5/gated-delta.test.ts`
- `packages/transformers/src/families/qwen3_5/block.ts`
- `packages/transformers/src/families/qwen3_5/cache.ts`
- `packages/transformers/src/infrastructure/generation/helpers.ts`
- `packages/transformers/src/infrastructure/generation/runtime.ts`
- `packages/transformers/scripts/benchmark-model.ts`
- `packages/transformers/scripts/benchmark-common.ts`
- `packages/transformers/scripts/benchmark-common.test.ts`
- `packages/transformers/scripts/benchmark-generation.ts`
- `packages/transformers/scripts/benchmark-generation-parity.ts`
- `packages/transformers/scripts/benchmark-long-context.ts`
- `scripts/runtime-sensitive-ops.ts`

## Tensor Lifetime Audit

`gatedDeltaSequence()` now treats the local recurrent state handle and the
returned final recurrent state as two distinct ownership claims. The function
retains the final state for the caller, frees its own local state handle on the
success path, and still frees the local handle on errors. Per-step outputs are
freed in `finally` after `stack(outputs, 1)` has created the returned sequence
output.

The regression test exercises the exact ownership boundary directly: it calls
`gatedDeltaSequence()` repeatedly with a large recurrent state, evaluates the
returned output and state, frees both caller-owned results, clears the MLX
cache, and asserts active memory stays under an 8 MB increase.

The follow-up Qwen slope audit found three more ownership issues in the Qwen
decode path:

- `Qwen3_5TextDecoderLayer.run()` created an owned token-mixer output without
  `using`; unlike Llama/Gemma blocks, that output was not scoped.
- `Qwen3_5GatedDeltaNet.run()` hid `this.conv1d.forward(convInput)` inside
  `silu(...)`, leaking the raw convolution output.
- `Qwen3_5GatedDeltaNet.run()` hid `takeLastAxisRange(...)` slice views inside
  `reshape(...)`, leaving projection slices anonymous.

Those lifetimes are now explicit. The tensor-lifetime checker was also widened
to treat nested `.forward(...)` and `takeLastAxisRange(...)` calls as tracked
runtime-sensitive tensor producers, so this class is harder to reintroduce.

The benchmark changes are host-side measurement code. They add active allocator
sampling, decode-schedule selection, and long-context decode-slope reporting
without introducing new model-owned tensor handles. The diagnostic sync decode
path was kept because it is useful for separating async lookahead issues from
model-state leaks.

## Memory / Performance Evidence

Existing benchmark suite, before the fix:

- `bun run bench:generation --model mlx-community/Qwen3.6-27B-4bit --prompt-tokens 128 --generation-tokens 64 --trials 1 --memory-sample-interval 16`
- Result: active memory rose from `28.956 GB` to `38.725 GB`, a `+9.770 GB`
  delta over 64 tokens, or `152.65 MB/token`.
- The same run with `--sync-decode` showed the same slope, so one-token async
  lookahead was not the root cause.

Existing benchmark suite, after the first recurrent-state fix:

- `bun run bench:generation --model mlx-community/Qwen3.6-27B-4bit --prompt-tokens 128 --generation-tokens 64 --trials 1 --memory-sample-interval 16`
- Result: active memory rose from `18.995 GB` to `19.258 GB`, a `+0.262 GB`
  delta over 64 tokens, or `4.10 MB/token`, with peak memory `20.091 GB` and
  generation throughput around `24.7 tok/s`.
- A 128-token decode confirmation stayed linear at roughly `4.12 MB/token`
  instead of returning to the runaway `~153 MB/token` slope.

Remaining-slope investigation:

- Expected Qwen 3.6 full-attention KV growth is only about `64 KiB/token` in
  bf16, or `128 KiB/token` if fp32. In the prompt 128 / generation 64 window,
  capacity had already rounded to 256 tokens, so expected full-KV allocator
  growth was effectively zero.
- `--materialize-cache-each-token` did not move the `~4 MB/token` number, so
  the issue was not simply unevaluated recurrent cache state.
- Fixing the unscoped decoder `tokenMixer` reduced the slope to `3.36 MB/token`.
- Fixing the hidden `conv1d.forward(...)` intermediate and anonymous
  `takeLastAxisRange(...)` slice views reduced the same prompt 128 /
  generation 64 run to active delta `-0.000 GB`, slope `-0.00 MB/token`, peak
  `19.161 GB`, and generation throughput `21.647 tok/s`.
- A stronger prompt 128 / generation 128 confirmation also measured active
  delta `-0.000 GB`, slope `-0.00 MB/token`, peak `19.161 GB`, and generation
  throughput `21.373 tok/s`.

Controls under the same `bench:generation` memory instrumentation:

- `mlx-community/Llama-3.2-1B-Instruct-4bit`, prompt 64 / generation 64:
  active delta `-0.000 GB`, slope `-0.00 MB/token`, peak `1.134 GB`.
- `google/gemma-3-1b-it`, prompt 64 / generation 64: active delta
  `-0.000 GB`, slope `-0.00 MB/token`, peak `2.117 GB`.
- `google/gemma-4-E2B-it`, prompt 64 / generation 64: active delta
  `-0.000 GB`, slope `-0.00 MB/token`, peak `9.375 GB`, generation
  throughput `72.059 tok/s`.

Focused validation:

- `bun test packages/transformers/src/families/qwen3_5/gated-delta.test.ts packages/transformers/src/families/qwen3_5/model.test.ts`
- `bun test packages/transformers/src/families/qwen3_5/cache.test.ts packages/transformers/src/families/qwen3_5/gated-delta.test.ts packages/transformers/src/families/qwen3_5/model.test.ts`
- `bun test packages/core/src/ops/ops.test.ts packages/transformers/src/families/qwen3_5/cache.test.ts packages/transformers/src/families/qwen3_5/gated-delta.test.ts`
- `bun test packages/transformers/src/families/qwen3_5/gated-delta.test.ts packages/transformers/src/families/qwen3_5/cache.test.ts packages/transformers/src/families/qwen3_5/model.test.ts packages/transformers/scripts/benchmark-common.test.ts packages/core/src/ops/ops.test.ts`
- `bun run typecheck`
- `bun run check:tensor-lifetimes`
- `bun run check:runtime-review`
- `bun run check:coverage`

Real serving smoke after the fix:

- `bun run packages/serve/src/cli.ts mlx-community/Qwen3.6-27B-4bit --model-id qwen-local --port 8022 --max-generated-tokens 256 --local-files-only`
- Three-turn `@mlxts/agent` smoke with model-native thinking enabled completed without a socket close or Bun crash. The server logged normal progress/completion events and peaked at `22.7 GB`.
- A second three-turn `@mlxts/agent` smoke with `--no-thinking --max-tokens 128` returned visible assistant answers for all turns, including the tool-capability question, and the server shut down cleanly. The final request completed with peak memory `22.9 GB`.
- The thinking-enabled run showed a separate UX constraint: a `128` token cap can be consumed entirely by Qwen reasoning on some turns. That is not the crash root, but small-budget agent smoke tests should either use `--no-thinking` or a larger generation cap.

`bench:generation:parity` was not rerun as the acceptance signal for this
specific incident because the bug was an allocator-lifetime leak reproduced
inside the TypeScript benchmark before any reference-model comparison was
needed. The Qwen 3.6 text path already has token-level `mlx-lm` parity evidence
recorded in the Qwen 3.5 / Qwen 3.6 tranche review; a future performance pass
should add paired Qwen decode numbers after the fused gated-delta/native-kernel
question is taken up.

## Independent Review

`Huygens` independently identified Qwen 3.6's recurrent linear-attention state
as the likely root. The review calculated that Qwen's 48 linear layers each
carry value-head recurrent state, making one full recurrent state set roughly
150 MB. That matched the observed `~153 MB/token` active-memory slope.

`Descartes` independently recommended using the existing transformer benchmark
suite instead of creating a new harness, especially `bench:generation`,
`bench:generation:parity`, `benchmark-long-context.ts`, and cache benchmarks.
That shaped the final debugging path: first reproduce in `bench:generation`,
then add reusable active-memory slope reporting.

`Beauvoir`, `Plato`, and `Lorentz` reviewed the remaining `~4 MB/token` slope.
They agreed it was not expected KV growth. Lorentz identified the missing
Qwen block `tokenMixer` lifetime; Beauvoir and Plato independently quantified
the expected Qwen cache footprint and pointed at Qwen linear-attention
intermediate retention rather than server or allocator-cache noise.

## Remaining Risks / Follow-ups

The decode-memory slope that caused the crash is fixed in the measured
no-capacity-growth window. The remaining Qwen follow-up is performance parity:
`.reference/mlx-lm` uses fused gated-delta kernels, while this repo currently
expresses the recurrence in ordinary MLX ops. That is now a throughput/native
kernel question, not an active-memory leak acceptance blocker.

The remaining serve/agent follow-up is no longer "does it crash on the basic
multi-turn path"; that smoke now passes. The next usability follow-up is budget
policy for thinking-enabled models, because small `max_tokens` values can return
only reasoning content before the model reaches visible assistant text.
