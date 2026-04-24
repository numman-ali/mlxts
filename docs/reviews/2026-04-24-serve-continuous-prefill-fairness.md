# Serve Continuous Prefill Fairness

## Summary

Improved the existing full-KV greedy continuous scheduler so a long waiting
prompt no longer becomes a whole-prompt admission wall while active rows are
decoding. Waiting rows are now reserved as partial-prefill rows, chunked by
`prefillStepSize`, and only merged into the active batch after the final prompt
token produces the first sampled generation token.

This is intentionally not a broader serving claim. Streaming collectors,
sampling, Qwen hybrid caches, and Gemma sliding/global caches still require
separate scheduler/cache work.

## Files Reviewed

- `packages/transformers/src/infrastructure/generation/continuous-batch.ts`
- `packages/transformers/src/infrastructure/generation/continuous-batch-helpers.ts`
- `packages/transformers/src/infrastructure/generation/continuous-batch.test.ts`
- `packages/serve/src/transformers-engine-continuous.ts`
- `packages/serve/src/transformers-engine-generation.ts`
- `packages/serve/AGENTS.md`
- `MEMORY.md`
- `continuity.md`

## Runtime Invariants

- Active rows keep the existing invariant: `BatchKVCache` contains prompt plus
  already-fed generated tokens, while `currentToken` is the sampled token waiting
  to be emitted/fed back.
- Prefilling rows count against scheduler capacity but are not active until
  their first sampled token exists.
- A prefilling row owns a one-row `BatchKVCache`; once ready, it is merged into
  the active cache through `extend()`, and the temporary cache is disposed.
- When active rows exist, only one waiting row is prefilling at a time, avoiding
  hidden multi-prompt prefill walls before active decode resumes.
- Prefill progress is surfaced through the existing serving
  `generation_prefill_progress` event path.

## Tensor Lifetime Audit

- Chunk input tensors and chunk logits use `using` declarations.
- Cache state arrays returned by `BatchKVCache.arrays()` are explicitly evaluated
  and freed during chunked prefill.
- Ready-row token tensors are either installed as `currentToken`, combined with
  the existing active token tensor, or freed on the impossible no-row fallback.
- Aborted prefilling rows dispose their one-row cache and reject before joining
  active decode.
- Scheduler failure cleanup rejects waiting, prefilling, and active requests,
  disposes prefilling caches, and frees active cache/current token handles.

## Independent Review

The implementation follows the prior sub-agent recommendation to fix chunked
prefill fairness before attempting Qwen/Gemma continuous batching. The reviewer
grounded that recommendation in `.reference/rapid-mlx` and `.reference/vllm-mlx`
scheduler structure: production schedulers treat prefill and decode as
interleavable work rather than one HTTP-layer admission batch.

Follow-up sub-agent review found no blockers. It specifically checked
`BatchKVCache` ownership, `currentToken` alignment with active row order,
prefilling-row cancellation cleanup, progress reporting, and the narrow
full-KV greedy scope. Non-blocking suggestions led to the added abort-during-
partial-prefill test and to including `prefillStepSize` in the continuous
scheduler key.

## Memory / Performance Evidence

- `bun test packages/transformers/src/infrastructure/generation/continuous-batch.test.ts`
  passed. The new scheduler test verifies a long waiting prompt is chunked as
  `2/5`, `4/5`, `5/5`, never forwarded as a full six-token prompt, and active
  decode forwards continue between prefill chunks. The test suite also now
  covers abort during partial prefill.
- `bun test packages/serve/src/transformers-engine.test.ts packages/serve/src/model-server.test.ts`
  passed.
- Tiny cached `bench:generation:parity` guard, which exercises the
  in-process `bench:generation` path without an external MLX-LM reference:
  `bun run bench:generation:parity --model mlx-community/Llama-3.2-1B-Instruct-4bit --prompt-tokens 16 --generation-tokens 16 --trials 1 --memory-sample-interval 16 --skip-mlx-lm-reference`.
  It reported `generation_tps=364.322`, `peak_memory=1.094 GB`,
  `active_delta=-0.000 GB`, `active_slope_mb_per_token=-0.00`, and
  `evals_per_token=1.00`.
- `bun run typecheck` passed.
- `bun run lint` passed.
- `bun run check:file-lines` passed after extracting helper code; the scheduler
  file remains within the 500-line production limit.
- `bun run check:tensor-lifetimes` passed.

## Remaining Risks / Follow-ups

- Initial batch-window admission can still prefill the initial eligible batch as
  one batch call. This tranche fixes fairness for waiting rows behind active
  decode, which is the path that can starve already-running requests.
- No Qwen or Gemma continuous-batching claim is made here. Their cache semantics
  still need dedicated batch-aware representations.
- Streaming still uses the single-request lane; streaming collectors are a later
  tranche.
