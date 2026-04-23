# Runtime Review: Qwen 3.5 / Qwen 3.6 multimodal tranche

## Summary

This tranche lands the first reusable multimodal transformer path in the repo.
It widens the shared generation/input contract for prepared prompts, adds the
first `qwen3_5` text-plus-vision family implementation, adds example-local
Apple-native image decode/resize for a one-shot smoke surface, and keeps the
package boundary clean by leaving file/image I/O outside `@mlxts/transformers`.

Follow-up debugging on the same tranche also closed the real Qwen 3.6 text
correctness gap against Apple’s reference stack. The final landed fixes were:
loader-side norm sanitation for raw-vs-converted checkpoints, direct runtime
RMSNorm semantics for Qwen text layers, the correct per-head `[query, gate]`
split for packed full-attention `q_proj` outputs, and sparse added-token decode
support for tokenizer IDs above the base vocab range.

The review covers the runtime-sensitive code that made that possible:
`inputEmbeddings`/`positionIds` support in generation, Qwen 3.5 hybrid decoder
and vision composition, the MLX-facing `Conv1d` primitive needed by the family,
and snapshot inspection updates for multimodal sidecars.

## Files Reviewed

- `packages/core/src/ffi/symbols.ts`
- `packages/core/src/index.ts`
- `packages/core/src/ops/arithmetic.ts`
- `packages/core/src/ops/index.ts`
- `packages/core/src/ops/linalg.ts`
- `packages/core/src/ops/shape.ts`
- `packages/nn/src/conv1d.ts`
- `packages/nn/src/index.ts`
- `packages/tokenizers/src/bpe-base.ts`
- `packages/transformers/src/families/gemma3/model.ts`
- `packages/transformers/src/families/gemma4/model.ts`
- `packages/transformers/src/families/llama-like/model.ts`
- `packages/transformers/src/families/qwen3_5/attention.ts`
- `packages/transformers/src/families/qwen3_5/block.ts`
- `packages/transformers/src/families/qwen3_5/cache.ts`
- `packages/transformers/src/families/qwen3_5/conditional.ts`
- `packages/transformers/src/families/qwen3_5/conditional-support.ts`
- `packages/transformers/src/families/qwen3_5/config.ts`
- `packages/transformers/src/families/qwen3_5/config-helpers.ts`
- `packages/transformers/src/families/qwen3_5/gated-delta.ts`
- `packages/transformers/src/families/qwen3_5/mlp.ts`
- `packages/transformers/src/families/qwen3_5/model.ts`
- `packages/transformers/src/families/qwen3_5/norm.ts`
- `packages/transformers/src/families/qwen3_5/preprocessing.ts`
- `packages/transformers/src/families/qwen3_5/rotary.ts`
- `packages/transformers/src/families/qwen3_5/types.ts`
- `packages/transformers/src/families/qwen3_5/vision.ts`
- `packages/transformers/src/families/qwen3_5/vision-support.ts`
- `packages/transformers/src/families/qwen3_5/weights.ts`
- `packages/transformers/src/generation.ts`
- `packages/transformers/src/index.ts`
- `packages/transformers/src/infrastructure/cache/index.ts`
- `packages/transformers/src/infrastructure/generation/helpers.ts`
- `packages/transformers/src/infrastructure/generation/runtime.ts`
- `packages/transformers/src/infrastructure/input-embeddings.ts`
- `packages/transformers/src/load-quantized.ts`
- `packages/transformers/src/pretrained/snapshot-supported-files.ts`
- `packages/transformers/src/pretrained/snapshot-inspection.ts`
- `packages/transformers/src/pretrained/snapshot.ts`
- `packages/transformers/src/pretrained/types.ts`
- `packages/transformers/src/registry.ts`
- `packages/transformers/src/types.ts`

## Tensor Lifetime Audit

I reran `bun run check:tensor-lifetimes` after the Qwen tranche landed. The
only backstop failures were in `packages/transformers/src/families/qwen3_5/`
where disposable intermediates were still nested inside `reshape()`, `sum()`,
`multiply()`, `expandDims()`, and `slice()` calls. Those sites were rewritten
to bind the inner arrays to named `using` locals first, so the lifetime of each
MLX array is visible in code and matches the repo’s ownership rules.

I also checked the new multimodal prompt path itself for ownership clarity:
`prepareQwen3_5ImagePrompt()` returns owned `inputEmbeddings` and `positionIds`
to the caller, and the example frees them explicitly after generation. The
example similarly frees `pixelValues` and `imageGridThw` after prompt
preparation/generation rather than letting them escape implicitly.

## Memory / Performance Evidence

Validated so far for this tranche:

- `bun run typecheck`
- `bun run check:tensor-lifetimes`
- `bun test packages/tokenizers/src/bpe.test.ts packages/transformers/src/families/qwen3_5/attention.test.ts packages/transformers/src/families/qwen3_5/model.test.ts packages/transformers/src/families/qwen3_5/conditional.test.ts packages/transformers/src/families/qwen3_5/weights.test.ts packages/transformers/src/load.test.ts`
- `bun test examples/qwen3_5-image/image-io.test.ts packages/transformers/src/families/qwen3_5/conditional.test.ts packages/transformers/src/families/qwen3_5/preprocessing.test.ts packages/transformers/src/families/qwen3_5/weights.test.ts packages/transformers/src/families/qwen3_5/config.test.ts packages/transformers/src/registry.test.ts packages/transformers/src/pretrained/snapshot.test.ts`
- Local `mlx-community/Qwen3.6-27B-4bit` text-only greedy generation now matches Apple `mlx-lm` token-for-token for the checked 16-token continuation from the prompt `"The capital of France is"`: `[11751, 13, 271, 248068, 271, 248069, 271, 4639, 369, 4252, 13, 11751, 369, 279, 6511, 321]`
- `examples/qwen3_5-image/index.ts` now provides the intended one-shot image-conditioned smoke surface for `mlx-community/Qwen3.6-27B-4bit`; this artifact still does not record a completed real quantized multimodal run.

Generation hot-path note:

- `bench:generation` was not treated as the headline proof for this tranche because the landed work primarily affects multimodal prompt prefill and optional prepared-input pathways, not the steady-state dense decode math alone.
- `bench:generation:parity` was likewise not used as the acceptance headline for this tranche because the primary deliverable was correct multimodal composition and prompt preparation.
- Even so, both `bench:generation` and `bench:generation:parity` remain required follow-up commands before claiming decode-speed improvements from the shared-generation changes in this tranche.

## Independent Review

Independent review came from parallel sub-agent audits during implementation:

- `Avicenna` audited preprocessing structure and later reviewed the runtime-review wording to keep the validation claims exact.
- `Nietzsche` audited the `vision.ts` and `conditional.ts` split boundaries so the landed Qwen tranche stayed readable rather than accreting a monolith, and later caught a real forwarding bug where the text-only `Qwen3_5TextCausalLM` wrapper was dropping explicit `positionIds` before final revalidation.
- `Godel` provided a second-opinion review on checkpoint choice and the shared multimodal contract surface while the real smoke target was downloading.
- The final text-parity debug loop also used direct cross-checks against Apple’s reference `qwen3_5.py` on the same cached `mlx-community/Qwen3.6-27B-4bit` snapshot to isolate the first diverging layer and confirm the repaired greedy token sequence.

Those reviews were independent from the implementation pass and directly
changed the landing plan.

## Remaining Risks / Follow-ups

- Video-conditioned prompt preparation is still intentionally out of scope for this tranche. The wrapper and snapshot layer now recognize the relevant sidecars, but the canonical prepared-prompt helper is image-only today.
- The example depends on macOS `sips` for local resize/decode. That is the right boundary for this Apple Silicon repo, but it is still example-local plumbing rather than a cross-platform media layer.
- The first smoke surface is one-shot generation, not a full multimodal chat session with mixed message-content arrays.
- A tiny top-level `qwen3_5` load fixture in `packages/transformers/src/load.test.ts` would still be valuable as a lightweight regression check alongside a future real quantized smoke run.
