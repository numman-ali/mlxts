# Runtime Review: Split Quantized MoE Experts

## Summary

This tranche adds the missing real-checkpoint MoE seam for Qwen 3.6 A3B and Gemma 4 A4B style checkpoints. Official HF combined expert tensors remain supported through `PackedSwitchGLUExperts`; converted MLX/Unsloth split expert tensors now replace that packed bank with switch-routed `gateProjection`, `upProjection`, and `downProjection` leaves that can receive MLX-native `weight/scales/biases` without dense staging.

The hot-path execution now binds and uses MLX's native `gather_mm` and `gather_qmm` kernels for split switch experts. This avoids materializing per-token selected expert weight banks in TypeScript and matches the mlx-lm `SwitchLinear` / `QuantizedSwitchLinear` execution pattern.

## Files Reviewed

- `packages/core/src/ffi/symbols.ts`
- `packages/core/src/index.ts`
- `packages/core/src/ops/index.ts`
- `packages/core/src/ops/linalg.ts`
- `packages/core/src/quantization.ts`
- `packages/transformers/src/families/gemma4/block.ts`
- `packages/transformers/src/families/gemma4/types.ts`
- `packages/transformers/src/families/qwen3_5/mlp.ts`
- `packages/transformers/src/families/qwen3_5/weights.ts`
- `packages/transformers/src/infrastructure/moe.ts`
- `packages/transformers/src/load-quantized.ts`

## Tensor Lifetime Audit

`gatherMm` and `gatherQmm` return one new `MxArray` result each through the existing `readResultArray` out-slot pattern. They do not take ownership of inputs, index tensors, scales, or biases.

`SwitchLinear.prepareQuantized()` frees the dense placeholder weight and any previous placeholder auxiliary arrays before installing quantized placeholder leaves. Loaded checkpoint assignment still owns the final tensors through `assignWeightPath`.

`SwitchGLUExperts.forward()` keeps all intermediate arrays in visible `using` bindings. The packed official path is preserved with the existing visible `takeAxis`, projection, activation, and weighted-sum lifetimes.

`load-quantized.ts` replaces packed expert modules only before expected parameter paths are computed. It disposes the replaced packed module immediately, then prepares switch-linear quantized leaves so `.weight/.scales/.biases` are concrete `MxArray` slots before checkpoint assignment.

## Memory / Performance Evidence

Reference audit:

- `.reference/mlx-lm/mlx_lm/models/switch_layers.py` uses `mx.gather_mm` for dense `SwitchLinear` and `mx.gather_qmm` for `QuantizedSwitchLinear`.
- `.reference/mlx-lm/mlx_lm/models/qwen3_5_moe.py` splits official `experts.gate_up_proj` into `switch_mlp.gate_proj.weight` and `switch_mlp.up_proj.weight`, while mapping `experts.down_proj` to `switch_mlp.down_proj.weight`.
- `.reference/mlx-lm/mlx_lm/models/gemma4_text.py` performs the same split for Gemma 4 `experts.switch_glu.*` weights.
- `packages/core/native/build/_deps/mlx-c-src/mlx/c/ops.h` exposes `mlx_gather_mm` and `mlx_gather_qmm`, so this is an MLX-C-first binding rather than a JavaScript workaround.

Focused validation:

- `bun test packages/core/src/quantization.test.ts packages/transformers/src/infrastructure/moe.test.ts packages/transformers/src/load.test.ts packages/transformers/src/families/qwen3_5/weights.test.ts packages/transformers/src/families/gemma4/weights.test.ts` passed: 61 tests, 492 expectations.
- The load tests cover official combined Qwen MoE, split quantized Qwen MoE, official combined Gemma 4 MoE, and split quantized Gemma 4 MoE fixtures.
- `gatherQmm` is checked against dequantized `gatherMm` with repeated expert indices, and quantized split SwitchGLU is checked against dequantized switch weights.

Benchmark scope:

- No real A3B/A4B checkpoint benchmark is claimed in this tranche. The purpose is to make the correct checkpoint topology load and run without dense expert staging. The next proof must run real cached/downloaded Qwen and Gemma MoE checkpoints under the shared MLX runtime lock and record memory/TPS/context evidence.
- `bench:generation` was not run for this tranche because no real MoE checkpoint was cached at implementation time and tiny synthetic fixtures would not produce useful throughput evidence for expert gather kernels.
- `bench:generation:parity` was not run for the same reason; paired `mlx-lm` parity belongs in the follow-up real-checkpoint proof for `unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit` and `mlx-community/gemma-4-26b-a4b-it-4bit`.

## Independent Review

Schrodinger the 2nd identified the key implementation correction before commit: do not dequantize or combine expert weights, bind `mlx_gather_qmm`, and keep official combined `gate_up_proj` support separate from split converted checkpoints.

Pasteur the 2nd independently confirmed the checkpoint truth: official HF Qwen/Gemma MoE checkpoints use combined `gate_up_proj`, while mlx-lm and converted MLX/Unsloth checkpoints split into switch `gate_proj`, `up_proj`, and `down_proj`. The final implementation follows that dual-path recommendation.

## Remaining Risks / Follow-ups

The real `unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit` and `mlx-community/gemma-4-26b-a4b-it-4bit` checkpoints still need load/decode/serve proof. This tranche makes that proof meaningful; it does not replace it.

Sorted-index switch execution from mlx-lm is not implemented yet. The unsorted native gather path is correct and avoids dense staging; sorting should be evaluated only with paired real-checkpoint measurements.

Official quantized combined `gate_up_proj` tensors are not split during exceptional loading yet. Dense official combined tensors remain supported, and split MLX quantized tensors are now supported; official combined quantized expert tensors need a separate one-input-to-two-output loader if such checkpoints become a target.
