# Runtime Review: Serve Prompt Admission

## Summary

This change separates long-context serving admission into explicit prompt,
generated-output, and total-token budgets. The transformer-backed engine now
rejects tokenized prompts above `maxPromptTokens` before prefill/generation, the
CLI exposes `--max-prompt-tokens`, and `/info` reports per-model admission
metadata including checkpoint-declared context windows and effective total
limits.

The intent is operator safety and clearer failure modes. This is not a KV-cache
or scheduler change, and it does not claim that every model-advertised context
window fits local memory.

## Files Reviewed

- `packages/serve/src/cli-options.ts`
- `packages/serve/src/cli.ts`
- `packages/serve/src/index.ts`
- `packages/serve/src/model-context.ts`
- `packages/serve/src/model-server.ts`
- `packages/serve/src/model-sources.ts`
- `packages/serve/src/protocols/openai-models.ts`
- `packages/serve/src/server-info.ts`
- `packages/serve/src/transformers-engine-generation.ts`
- `packages/serve/src/transformers-engine-shared.ts`
- `packages/serve/src/transformers-engine.ts`

## Tensor Lifetime Audit

The new admission path operates on host-side token counts and model config
metadata only. It does not allocate or retain `MxArray` values.

`enforcePromptTokenLimit()` runs after prompt compilation/token counting and
before `generateTextStream()`, `generateTokens()`, `generatePreparedTokenEvents()`,
or cached prefill. A rejected long prompt therefore exits before model forward
or cache mutation. The existing total-token limit continues to run before
generation as well.

`modelContextWindow()`, `effectiveTotalTokenLimit()`, and `/info` formatting only
read `model.config.rawConfig` and configured server limits. They do not interact
with native handles, tokenizer state, model cache state, or generation streams.

## Memory / Performance Evidence

Validated with:

- `bun test packages/serve/src/model-context.test.ts packages/serve/src/transformers-engine.test.ts packages/serve/src/model-server.test.ts packages/serve/src/server.test.ts packages/serve/src/cli.test.ts packages/serve/src/model-sources.test.ts`
- `bun run check:file-lines -- packages/serve/src/model-server.ts packages/serve/src/model-context.ts packages/serve/src/model-context.test.ts packages/serve/src/transformers-engine-shared.ts packages/serve/src/cli-options.ts packages/serve/README.md packages/serve/AGENTS.md`
- `bun run typecheck`
- `bun run lint`

No new `bench:generation` or `bench:generation:parity` run is required for this
slice because the generation math, sampling path, cache representation, and
decode loop are unchanged. The change rejects unsafe requests before those paths
begin and adds host-side introspection metadata.

## Independent Review

Sub-agent Harvey completed a read-only serve admission audit. The review
recommended exactly this narrow slice: add a prompt-token cap at the
transformer-backed admission point, make the error distinct from total-token
rejection, and expose context/admission metadata through `/info` rather than
claiming memory safety from the OpenAI `/v1/models` shape.

The review also cautioned that context metadata based on
`max_position_embeddings` can miss family-specific runtime behavior, RoPE
scaling, sliding-window policy, or multimodal prompt expansion. The
implementation therefore presents the values as admission metadata, not a full
memory preflight guarantee.

## Remaining Risks / Follow-ups

This is still not memory-aware preflight. Qwen 3.6 advertises a 262144-token
context window, but local 128k evidence already peaked at 42.550 GB, so larger
contexts still need machine-specific budgeting before testing.

True high-concurrency serving still needs Qwen-aware batch/cache semantics,
prefix-cache reuse, non-streaming cancellation, memory-pool safeguards, and a
scheduler-owned decode loop. Admission micro-batching remains explicitly
separate from continuous batching.
