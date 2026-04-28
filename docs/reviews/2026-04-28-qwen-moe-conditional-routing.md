# Runtime Review: Qwen MoE Conditional Routing

## Summary

This tranche makes Qwen 3.6 MoE conditional checkpoints reachable through the same honest serving path as dense Qwen conditional checkpoints. A top-level `qwen3_5_moe` wrapper that advertises `Qwen3_5MoeForConditionalGeneration` and `vision_config` now selects the conditional loader, attaches the Qwen image content adapter, and routes the text MoE model type through the Qwen hybrid continuous/static batching path.

The change is intentionally small. It does not claim real-checkpoint MoE performance parity yet; it removes a loader/routing blocker so the next tranche can run real `Qwen/Qwen3.6-35B-A3B` proof instead of falling through to the wrong generic loader.

## Files Reviewed

- `packages/serve/src/transformers-engine-routing.ts`
- `packages/transformers/src/families/qwen3_5/config.ts`
- `packages/transformers/src/families/qwen3_5/load.ts`

## Tensor Lifetime Audit

`transformers-engine-routing.ts` only updates model-type admission sets. No tensor handles are created, retained, forked, or freed there.

`qwen3_5/config.ts` extends conditional-family registration metadata so `qwen3_5_moe` top-level wrappers can be parsed by the conditional family. The existing Qwen MoE tensor ownership remains in model construction and weight assignment; this edit does not add new tensor-producing expressions.

`qwen3_5/load.ts` only widens source detection from dense conditional wrappers to dense-or-MoE conditional wrappers. It reads `config.json`, checks top-level architecture/model type/vision metadata, and returns a boolean. It does not load weights or allocate MLX arrays.

## Memory / Performance Evidence

Reference audit:

- Hugging Face model index/config shape for `Qwen/Qwen3.6-35B-A3B` uses top-level `model_type: "qwen3_5_moe"` and `architectures: ["Qwen3_5MoeForConditionalGeneration"]`, with nested language-model keys under `model.language_model.*`.
- `.reference/mlx-lm/mlx_lm/models/qwen3_5_moe.py` treats `Qwen3_5MoeForConditionalGeneration` as the conditional MoE wrapper and rewrites the top-level language model namespace before loading.
- Local `qwen3_5_moe_text` foundations already expose the Qwen hybrid batch cache, so serving route admission should include that model type alongside `qwen3_5_text`.

Focused validation:

- `bun test packages/transformers/src/load.test.ts packages/serve/src/model-server.test.ts packages/serve/src/model-sources.test.ts packages/serve/src/transformers-engine.test.ts` passed: 100 tests, 541 expectations.

Benchmark scope:

- `bench:generation` and `bench:generation:parity` were not run for this routing-only tranche because no forward pass, cache update, sampling, mask, or tensor math changed. The correct benchmark point is the next real-checkpoint MoE proof, after this loader/routing fix makes `Qwen/Qwen3.6-35B-A3B` reach the intended implementation instead of a fallback path.

## Independent Review

Raman the 2nd reviewed the MoE/product tranche independently before this implementation was finalized. The review identified real MoE checkpoint enablement as the next highest-leverage product tranche, specifically called out that `qwen3_5_moe_text` was missing from Qwen continuous/static serving routes, and recommended proving real Gemma A4B and Qwen A3B checkpoints after restoring the route.

James the 2nd and Sartre the 2nd also reviewed the adjacent Qwen image-serving tranche. Their findings were already addressed in the committed image path, but they are kept in mind here because this MoE conditional loader shares the same content-adapter attachment surface.

## Remaining Risks / Follow-ups

This is not yet a claim that `Qwen/Qwen3.6-35B-A3B` is production-proven. The next tranche must run the real checkpoint load/decode/regression path under the shared MLX runtime lock, compare against `mlx-lm` where feasible, and record actual memory/TPS/context evidence.

Gemma 4 A4B remains a separate real-checkpoint proof path. Gemma image/video support should not be inferred from this Qwen conditional routing change; it needs transformer-owned Gemma processor and prepared-prompt seams first.

