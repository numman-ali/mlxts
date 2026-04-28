# Runtime Review: Serve Media Normalization and MoE Text Foundations

## Summary

This tranche adds a protocol-neutral media content shape to `@mlxts/serve`, keeps media inputs explicitly out of the text-only transformers execution path, and adds first-pass text MoE support for Qwen 3.5/3.6 MoE and Gemma 4 A4B-style configs without changing the `CausalLM` serving contract.

The serving work is intentionally not full multimodal execution yet: OpenAI Chat Completions and Responses can preserve ordered text/media parts, but the transformers engine rejects those requests with `unsupported_input` until model-family prompt preparation owns image/audio/file tensors.

## Files Reviewed

- `packages/serve/src/index.ts`
- `packages/serve/src/protocols/media-content.ts`
- `packages/serve/src/protocols/openai-chat-completions.ts`
- `packages/serve/src/protocols/openai-chat-messages.ts`
- `packages/serve/src/protocols/openai-responses-input.ts`
- `packages/serve/src/protocols/openai-responses.ts`
- `packages/serve/src/transformers-engine-generation.ts`
- `packages/serve/src/transformers-engine-routing.ts`
- `packages/serve/src/transformers-engine.ts`
- `packages/serve/src/types.ts`
- `packages/transformers/src/families/gemma4/block.ts`
- `packages/transformers/src/families/gemma4/config.ts`
- `packages/transformers/src/families/gemma4/moe.ts`
- `packages/transformers/src/families/gemma4/types.ts`
- `packages/transformers/src/families/qwen3_5/block.ts`
- `packages/transformers/src/families/qwen3_5/config-feedforward.ts`
- `packages/transformers/src/families/qwen3_5/config.ts`
- `packages/transformers/src/families/qwen3_5/mlp.ts`
- `packages/transformers/src/families/qwen3_5/types.ts`
- `packages/transformers/src/families/qwen3_5/weights.ts`
- `packages/transformers/src/infrastructure/moe.ts`

## Tensor Lifetime Audit

The new serve media parsing code is host-only and does not create tensor handles.

The new MoE tensor paths were checked for visible ownership:

- `topKFromRouterProbabilities` returns owned `indices` and `weights`; if weight selection fails, it frees `indices` before rethrowing.
- `Qwen3_5TextMoE.forward` frees routed `indices` and `weights` in a `finally` block after packed expert execution and shared-expert gating.
- `Gemma4TextRouter.route` transfers ownership of `routing.indices`, replaces `routing.weights` with scaled weights, and frees both on failure.
- `Gemma4TextDecoderBlock.runMoeFeedforwardTail` frees routed `indices` and `weights` in a `finally` block.
- `PackedSwitchGLUExperts.forward` keeps intermediate tensor handles in visible `using` declarations and returns only the final summed tensor.

Focused tests evaluate MoE outputs with `mxEval` before reading shapes or values.

## Memory / Performance Evidence

Focused validation run:

- `bun test packages/transformers/src/infrastructure/moe.test.ts packages/transformers/src/families/qwen3_5/config.test.ts packages/transformers/src/families/qwen3_5/weights.test.ts packages/transformers/src/families/qwen3_5/mlp.test.ts packages/transformers/src/families/gemma4/config.test.ts packages/transformers/src/families/gemma4/moe.test.ts packages/transformers/src/families/gemma4/block.test.ts packages/transformers/src/families/gemma4/weights.test.ts` passed in focused runs. Coverage includes fixed-weight Qwen MoE routed/shared expert mixing and fixed-weight Gemma router top-k/per-expert scaling.
- `bun test packages/serve/src/protocols/openai-chat-completions.test.ts packages/serve/src/protocols/openai-responses.test.ts packages/serve/src/transformers-engine.test.ts packages/serve/src/server.test.ts` passed: 116 tests.
- `bun run typecheck` passed across all workspaces.
- `bun run check:coverage` passed across the canonical package stack after the registry test was updated for Qwen MoE model types.

No large-model `bench:generation` or `bench:generation:parity` run is claimed for this tranche yet. The MoE expert runner is correctness-first and materializes selected expert weights for `[tokens, top_k]`; it is acceptable as a first architecture seam but is not the final long-context performance strategy.

## Independent Review

Curie reviewed the MoE architecture split before implementation: MoE remains a decoder block/feed-forward variant under `CausalLM`; Qwen replaces the dense MLP with routed plus shared experts; Gemma keeps the dense MLP branch and adds a routed branch in parallel.

Wegener reviewed the multimodal serving seam before implementation: serve should preserve ordered media content and reject it honestly until `@mlxts/transformers` owns family-specific prepared-prompt tensor construction.

McClintock performed the final working-tree review for hard-gate, tensor-lifetime, and semantic issues. That review found three bounded issues: `file_data` accepted non-data URLs, batched media requests did not emit the `media_input` route before rejection, and MoE tests were too shape/self-consistency heavy. All three findings were addressed before this artifact was finalized.

## Remaining Risks / Follow-ups

The packed expert implementation should be optimized before declaring Qwen/Gemma MoE production-fast. The likely next seam is a gathered matmul or native/compiled expert dispatch that avoids materializing full selected expert matrices per token.

Media normalization is not multimodal inference. The next product tranche should route ordered media content into Qwen/Gemma processor-owned prepared prompts inside `@mlxts/transformers`, then expose it through serve and local coding-agent clients with honest capability metadata.
